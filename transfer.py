import os
from copy import deepcopy
from pathlib import Path
from time import time

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import MetricCollection
from tsl import logger
from tsl.data import SpatioTemporalDataModule
from tsl.data.preprocessing import Scaler, StandardScaler
from tsl.datasets import PeMS03, PeMS04, PeMS07, PeMS08
from tsl.datasets.pems_benchmarks import _PeMS
from tsl.utils import ensure_list

from env import CACHE, HOME, REPRODUCIBLE, rng
from src.data.concat_datamodule import ConcatDataModule
from src.data.spatiotemporal_dataset import SpatioTemporalDataset
from src.data.splitters import TemporalConcatSplitter
from src.nn.models.prototypes import STGNN
from utils import maybe_add_regularization, maybe_adjust_grace_epochs, target_classname


def reset_metrics_(
    engine: LightningModule, cfg: DictConfig, post_fix: str | None = None
):
    engine._metric_attributes = None
    metrics = instantiate(cfg.engine.metrics)
    engine.test_metrics = MetricCollection(
        metrics={k: engine._check_metric(m) for k, m in metrics.items()},
        prefix="test/",
        postfix=post_fix,
    )


def get_datasets(cfg: DictConfig) -> tuple[list[_PeMS], _PeMS]:
    target = cfg.target
    datasets = {
        "pems3": PeMS03(),
        "pems4": PeMS04(),
        "pems7": PeMS07(),
        "pems8": PeMS08(),
    }
    target = datasets.pop(target)
    return list(datasets.values()), target


def get_torch_dataset(
    dataset, cfg, add_node_index: bool = True, idx_offset=0
) -> SpatioTemporalDataset:
    covariates = dict()
    if cfg.dataset.get("add_exogenous"):
        # encode time of the day and use it as exogenous variable
        day_sin_cos = dataset.datetime_encoded("day").values
        weekdays = dataset.datetime_onehot("weekday").values
        covariates.update(u=np.concatenate([day_sin_cos, weekdays], axis=-1))
    torch_dataset = SpatioTemporalDataset(
        target=dataset.dataframe(),
        mask=dataset.mask,
        connectivity=dataset.get_connectivity(**cfg.dataset.connectivity),
        covariates=covariates,
        horizon=cfg.dataset.horizon,
        window=cfg.dataset.window,
        stride=cfg.dataset.stride,
    )
    if add_node_index:
        node_index = torch.arange(
            idx_offset,
            idx_offset + torch_dataset.n_nodes,
            dtype=torch.int64,
        )
        torch_dataset.add_covariate(
            "node_idx", node_index, pattern="n", convert_precision=False
        )
    return torch_dataset


def update_cfg(
    cfg: DictConfig,
    torch_dataset: SpatioTemporalDataset,
    datamodule: ConcatDataModule | None = None,
    n_nodes: int | None = None,
) -> DictConfig:
    cfg = cfg.copy()
    n_nodes = n_nodes or torch_dataset.n_nodes
    with open_dict(cfg):
        cfg.model.update(
            n_nodes=n_nodes,
            input_size=torch_dataset.n_channels,
            exog_size=(
                torch_dataset.input_map.u.shape[-1] if "u" in torch_dataset else 0
            ),
            horizon=torch_dataset.horizon,
        )
    cfg.engine.metrics.mmre.update(dim=torch_dataset.n_channels)
    if "embedding" in cfg and cfg.embedding.get("module", None) is not None:
        with open_dict(cfg):
            cfg.embedding.module.n_nodes = n_nodes
        if (
            datamodule is not None
            and cfg.embedding.get("forgetting_scheduler", None) is not None
        ):
            num_train_batches = len(datamodule.train_dataloader())
            train_batches_limit = cfg.trainer.get("limit_train_batches", None)
            if train_batches_limit is not None:
                if isinstance(train_batches_limit, float):
                    assert (
                        0 <= train_batches_limit <= 1
                    ), "train_batches_limit must be in [0, 1] if float"
                    train_batches_limit = int(train_batches_limit * num_train_batches)
                num_train_batches = min(num_train_batches, train_batches_limit)
            to_update = {}
            for attr in ["stop_after", "dampen_for", "warmup_for", "period"]:
                val = cfg.embedding.forgetting_scheduler.get(attr, None)
                if val is not None:
                    to_update[attr] = int(val * num_train_batches)
            cfg.embedding.forgetting_scheduler.update(**to_update)
    return cfg


def train_on_source(datasets, cfg, hparams):
    torch_datasets = []
    offset = 0
    for ds in datasets:
        torch_datasets.append(
            get_torch_dataset(ds, cfg, add_node_index=True, idx_offset=offset)
        )
        offset += ds.n_nodes
    dm = ConcatDataModule(
        datasets=torch_datasets,
        scalers=dict(target=StandardScaler(axis=(0, 1))),
        splitter=TemporalConcatSplitter(**cfg.dataset.splitting),
        batch_size=cfg.dataset.batch_size,
        workers=cfg.dataset.num_workers,
    )
    dm.setup()
    scalers_dir = os.path.join(HydraConfig.get().runtime.output_dir, "scalers")
    for name, scaler in dm.scalers.items():
        filename = os.path.join(scalers_dir, name)
        scaler.save(filename)
    cfg = update_cfg(cfg, torch_datasets[0], dm, n_nodes=dm.n_nodes())
    engine = instantiate(cfg.engine)

    if cfg.get("log_run", None) is not None:
        exp_logger = dict(
            _target_="pytorch_lightning.loggers.WandbLogger",
            project=cfg.log_run,
            log_model=True,
            group="transfer.{model}.{embedding}.{regularization}.{target}".format(
                **hparams
            ),
            name=f"seed{cfg.get('seed', '')}.train.{int(time())}",
            save_dir=HydraConfig.get().runtime.output_dir,
        )
    else:
        exp_logger = dict(
            _target_="pytorch_lightning.loggers.logger.DummyLogger",
        )

    trainer = instantiate(
        cfg.trainer,
        default_root_dir=HydraConfig.get().runtime.output_dir,
        logger=exp_logger,
    )
    trainer.logger.log_hyperparams({**hparams, "phase": "train"})
    if isinstance(trainer.logger, WandbLogger):
        if os.path.exists(scalers_dir):
            trainer.logger.experiment.log_artifact(scalers_dir, type="scaler")
    trainer.fit(
        engine,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )

    model_path = os.path.join(HydraConfig.get().runtime.output_dir, "model.ckpt")
    os.rename(trainer.checkpoint_callback.best_model_path, model_path)

    engine.load_model(model_path)
    engine.freeze()

    trainer.test(engine, dataloaders=dm.test_dataloader())
    for idx, dataset in enumerate(datasets):
        logger.info(f"Evaluation on {dataset.name}")
        reset_metrics_(engine, cfg, f"/{dataset.name}")
        trainer.test(engine, dataloaders=dm.test_dataloader(idx))

    if trainer.logger:
        trainer.logger.finalize("success")
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.finish()

    return engine.model.cpu(), torch_datasets[0].scalers


def test_on_target(model, torch_dataset, dataset, cfg, test_id, hparams):
    train_len = cfg.target.splitting.train_len
    val_len = cfg.target.splitting.val_len
    assert val_len > 1

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        splitter=dataset.get_splitter(
            # add 'samples_offset' to val_len to keep intact number of samples
            val_len=val_len + torch_dataset.samples_offset,
            test_len=cfg.target.splitting.test_len,
        ),
        batch_size=cfg.dataset.batch_size,
        workers=cfg.dataset.num_workers,
    )
    dm.setup()
    assert len(dm.valset) == val_len

    # Shorten training set
    if train_len > 0:
        training_indices = dm.trainset.indices
        if train_len <= 1:
            train_len = int(train_len * len(torch_dataset))
        # set training set
        dm.trainset = training_indices[-train_len:]
        dm.splitter.set_indices(train=dm.trainset.indices)
        assert len(dm.trainset) == train_len
    else:
        dm.trainset = None

    cfg = update_cfg(cfg, torch_dataset)

    ########################################
    # Reset embeddings                     #
    ########################################

    if isinstance(model, STGNN):
        model.reset_embeddings(
            n_nodes=torch_dataset.n_nodes,
            requires_grad=train_len > 0,
            from_learned=cfg.target.get("reset_from_learned", False),
        )

    if cfg.target.tune_all and train_len > 0:
        for param in model.parameters():
            param.requires_grad = True

    ########################################
    # predictor                            #
    ########################################
    cfg = cfg.copy()
    cfg.engine.update(**cfg.target.engine)
    cfg.trainer.update(**cfg.target.trainer)
    cfg.early_stopping.update(**cfg.target.early_stopping)
    # remove forgetting and regularization in fine-tuning
    cfg.embedding.forgetting_scheduler = None
    cfg.engine.forgetting_scheduler = None
    if "embeddings_regularization" in cfg.engine:
        cfg.engine.embeddings_regularization = None
    # recreate engine to reflect updated training settings
    engine = instantiate(cfg.engine)
    # copy model
    engine.model = model
    # ensure dropout is not carried over by model
    engine.set_embeddings_dropout(0.0)

    log_dir = os.path.join(
        HydraConfig.get().runtime.output_dir, "transfer", str(test_id)
    )
    if cfg.get("log_run", None) is not None:
        seed = cfg.get("seed", "")
        exp_logger = dict(
            _target_="pytorch_lightning.loggers.WandbLogger",
            project=cfg.log_run,
            log_model=True,
            group="transfer.{model}.{embedding}.{regularization}.{target}".format(
                **hparams
            ),
            name=f"seed{seed}.transfer.t{train_len}.v{val_len}.{int(time())}",
            save_dir=log_dir,
        )
    else:
        exp_logger = dict(
            _target_="pytorch_lightning.loggers.logger.DummyLogger",
        )
    trainer = instantiate(cfg.trainer, default_root_dir=log_dir, logger=exp_logger)
    trainer.logger.log_hyperparams(
        {**hparams, "phase": "transfer", "train_len": train_len, "val_len": val_len}
    )

    ########################################
    # fine-tuning                          #
    ########################################

    if engine.trainable_parameters > 0 and train_len > 0:
        # fine-tune model
        trainer.fit(
            engine,
            train_dataloaders=dm.train_dataloader(),
            val_dataloaders=dm.val_dataloader(),
        )
        # test with parameters at the end of training
        reset_metrics_(engine, cfg, post_fix="/fine_tuning")
        trainer.test(engine, dataloaders=dm.test_dataloader())
        # test with the best parameters (early stopping)
        engine.load_model(trainer.checkpoint_callback.best_model_path)
        engine.freeze()
        reset_metrics_(engine, cfg)

    ########################################
    # testing                              #
    ########################################

    trainer.test(engine, dataloaders=dm.test_dataloader())

    # log metrics for smaller test intervals
    log_test_len = ensure_list(cfg.target.get("log_test_len", []))
    # order test_lens in descending order...
    log_test_len = sorted(log_test_len, reverse=True)
    for test_len in log_test_len:
        if test_len > len(dm.testset):
            continue
        days = int(test_len / 12 / 24)
        # ...and cut until test_len
        dm.testset = dm.testset.indices[:test_len]
        assert len(dm.testset) == test_len
        reset_metrics_(engine, cfg, f"/{days}_days")
        trainer.test(engine, dataloaders=dm.test_dataloader())

    if trainer.logger:
        trainer.logger.finalize("success")
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.finish()


def parse_hyperparams(cfg: DictConfig) -> dict[str, any]:
    if cfg.embedding.get("module", None) is not None:
        if cfg.embedding.module.get("forgetting_strategy", None) is not None:
            embedding = target_classname(cfg.embedding.module.forgetting_strategy)
            embedding = (
                f"{embedding}_{cfg.embedding.forgetting_scheduler.scheduler_class}"
            )
        else:
            embedding = target_classname(cfg.embedding.module)
        emb_size = cfg.embedding.module.emb_size
    else:
        embedding = "none"
        emb_size = 0
    regularization = cfg.get("regularization", None)
    if regularization is not None:
        regularization = regularization.method
    hyperparams = {
        "engine": target_classname(cfg.engine).lower(),
        "model": target_classname(cfg.model).lower(),
        "hidden_size": cfg.model.hidden_size,
        "embedding": embedding.lower(),
        "embedding_size": emb_size,
        "lr": cfg.engine.optim_kwargs.lr,
        "target_lr": cfg.target.engine.optim_kwargs.lr,
        "seed": cfg.get("seed", None) or "none",
        "regularization": regularization,
        "hydra_cfg": OmegaConf.to_container(cfg),
        "target": cfg.dataset.target,
    }
    return hyperparams


def load_model(path, torch_dataset, cfg):
    cfg = update_cfg(cfg, torch_dataset)
    engine = instantiate(cfg.engine)

    model_path = os.path.join(path, "model.ckpt")
    state_dict = torch.load(model_path, lambda storage, loc: storage)["state_dict"]

    ignore_modules = ["model.emb.emb"]
    for module in ignore_modules:
        if module in state_dict:
            del state_dict[module]
    # load weights
    status = engine.load_state_dict(state_dict, strict=False)
    assert set(status.missing_keys).issubset(ignore_modules), f"{status.missing_keys}"
    assert len(status.unexpected_keys) == 0

    scalers = dict()
    scalers_dir = os.path.join(path, "scalers")
    for scaler_file in os.listdir(scalers_dir):
        scaler_name = scaler_file[:-3]  # target.pt -> target
        scaler_path = os.path.join(scalers_dir, scaler_file)
        scalers[scaler_name] = Scaler.load(scaler_path)
    engine.freeze()
    return engine.model.cpu(), scalers


@hydra.main(config_path="conf", config_name="transfer", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.get("seed", None) is not None and REPRODUCIBLE:
        seed_everything(cfg.seed)
        rng.seed(cfg.seed)
    if cfg.get("num_threads", None) is not None:
        torch.set_num_threads(cfg.num_threads)
    cfg = maybe_add_regularization(cfg)
    cfg = maybe_adjust_grace_epochs(cfg)
    del cfg.trainer.logger

    source, target = get_datasets(cfg.dataset)

    hparams = parse_hyperparams(cfg)
    hparams["source_dataset"] = "_".join(sorted([d.name for d in source]))
    hparams["target_dataset"] = target.name

    torch_dataset = get_torch_dataset(target, cfg, add_node_index=False)
    if cfg.get("checkpoint", None) is not None:
        model, src_scalers = load_model(cfg.checkpoint, torch_dataset, cfg)
    else:
        model, src_scalers = train_on_source(source, cfg, hparams)
    for k, v in src_scalers.items():
        torch_dataset.add_scaler(k, v)

    train_lens = ensure_list(cfg.target.splitting.train_len)
    val_lens = ensure_list(cfg.target.splitting.val_len)
    if len(train_lens) == 1:
        train_lens = train_lens * len(val_lens)
    elif len(val_lens) == 1:
        val_lens = val_lens * len(train_lens)
    assert len(train_lens) == len(val_lens)

    # otherwise rnn will crash when calling backward in eval mode
    # "cudnn RNN backward can only be called in training mode"
    torch.backends.cudnn.enabled = False
    for test_id, (train_len, val_len) in enumerate(zip(train_lens, val_lens)):
        cfg.target.splitting.train_len = train_len
        cfg.target.splitting.val_len = val_len
        test_on_target(
            deepcopy(model), deepcopy(torch_dataset), target, cfg, test_id, hparams
        )


if __name__ == "__main__":
    map(lambda path: Path(path).mkdir(exist_ok=True, parents=True), [HOME, CACHE])
    main()
