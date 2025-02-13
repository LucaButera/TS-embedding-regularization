import ast
from time import time

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from wandb.sdk.lib.runid import generate_id

from env import CACHE
from src.data.spatiotemporal_datamodule import SpatioTemporalDataModule


def math_eval(node):
    # adapted from https://stackoverflow.com/a/9558001
    import ast
    import operator

    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }
    match node:
        case ast.Constant(value) if isinstance(value, int):
            return value  # integer
        case ast.BinOp(left, op, right):
            return operators[type(op)](math_eval(left), math_eval(right))
        case ast.UnaryOp(op, operand):  # e.g., -1
            return operators[type(op)](math_eval(operand))
        case _:
            raise TypeError(node)


def instantiate_data(cfg: DictConfig) -> SpatioTemporalDataModule:
    import numpy as np
    from hydra.utils import instantiate

    dataset = instantiate(cfg.dataset)
    torch_dataset = instantiate(
        cfg.torch_dataset,
        target=dataset.dataframe(),
        mask=dataset.mask,
        covariates=dataset.covariates,
        connectivity=dataset.get_connectivity(**cfg.connectivity),
        _convert_="partial",
    )
    if cfg.get("add_exogenous", False):
        day_sin_cos = dataset.datetime_encoded("day").values
        weekdays = dataset.datetime_onehot("weekday").values
        torch_dataset.add_covariate(
            name="u",
            value=np.concatenate([day_sin_cos, weekdays], axis=-1),
        )
    if cfg.get("mask_as_exog", False) and "u" in torch_dataset:
        torch_dataset.update_input_map(u=["u", "mask"])

    datamodule = instantiate(cfg.datamodule, dataset=torch_dataset, _convert_="partial")
    return datamodule


def update_config_from_data(
    cfg: DictConfig, datamodule: SpatioTemporalDataModule
) -> DictConfig:
    with open_dict(cfg):
        cfg.engine.model.update(
            input_size=datamodule.torch_dataset.n_channels,
            exog_size=(
                datamodule.torch_dataset.input_map.u.shape[-1]
                if "u" in datamodule.torch_dataset
                else 0
            ),
            n_nodes=datamodule.torch_dataset.n_nodes,
            horizon=datamodule.torch_dataset.horizon,
        )
        cfg.engine.scale_target = cfg.data.get("scale_target", False)
        if "embedding" in cfg and cfg.embedding.get("module", None) is not None:
            cfg.embedding.module.n_nodes = datamodule.torch_dataset.n_nodes
            if cfg.embedding.get("forgetting_scheduler", None) is not None:
                num_train_batches = len(datamodule.train_dataloader())
                train_batches_limit = cfg.trainer.get("limit_train_batches", None)
                if train_batches_limit is not None:
                    if isinstance(train_batches_limit, float):
                        assert (
                            0 <= train_batches_limit <= 1
                        ), "train_batches_limit must be in [0, 1] if float"
                        train_batches_limit = int(
                            train_batches_limit * num_train_batches
                        )
                    num_train_batches = min(num_train_batches, train_batches_limit)
                to_update = {}
                for attr in ["stop_after", "dampen_for", "warmup_for", "period"]:
                    val = cfg.embedding.forgetting_scheduler.get(attr, None)
                    if val is not None:
                        to_update[attr] = int(val * num_train_batches)
                cfg.embedding.forgetting_scheduler.update(**to_update)
        cfg.engine.metrics.mmre.update(dim=datamodule.torch_dataset.n_channels)
    return cfg


def target_classname(cfg: DictConfig) -> str:
    return cfg._target_.split(".")[-1]


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
    hyperparams = {
        "engine": target_classname(cfg.engine).lower(),
        "dataset": target_classname(cfg.data.dataset).lower(),
        "model": target_classname(cfg.model).lower(),
        "hidden_size": cfg.model.hidden_size,
        "embedding": embedding.lower(),
        "embedding_size": emb_size,
        "lr": cfg.engine.optim_kwargs.lr,
        "seed": cfg.get("seed", None) or "none",
        "hydra_cfg": OmegaConf.to_container(cfg),
    }
    return hyperparams


def maybe_update_logger_cfg(cfg: DictConfig) -> DictConfig:
    if target_classname(cfg.trainer.logger) == "WandbLogger":
        if cfg.trainer.logger.get("id", None) is None:
            cfg.trainer.logger.id = generate_id()
        if cfg.trainer.logger.get("save_dir", None) is None:
            cfg.trainer.logger.save_dir = HydraConfig.get().runtime.output_dir
        if cfg.trainer.logger.get("group", None) is None:
            hyperparams = parse_hyperparams(cfg)
            group_format = (
                "{engine}.{dataset}.{model}.h{hidden_size}."
                "{embedding}.e{embedding_size}.lr{lr}"
            )
            group = group_format.format(**hyperparams)
            if cfg.get("notes", None) is not None:
                group = f"{group}.{cfg.notes}"
            cfg.trainer.logger.group = group
        if cfg.trainer.logger.get("name", None) is None:
            cfg.trainer.logger.name = f"seed{cfg.get('seed', '')}.{int(time())}"
    return cfg


def maybe_add_regularization(cfg: DictConfig) -> DictConfig:
    regularization = cfg.get("regularization", None)
    whole_model = cfg.get("regularize_whole_model", False)
    if regularization is not None and regularization.method != "none":
        if whole_model:
            if regularization.method == "dropout":
                with open_dict(cfg):
                    cfg.model.update(
                        dropout=regularization.factor,
                    )
            elif regularization.method == "l2":
                with open_dict(cfg):
                    cfg.engine.optim_kwargs.update(
                        weight_decay=regularization.factor,
                    )
            else:
                raise ValueError(
                    f"Unknown regularization method {regularization.method}"
                )
        else:
            with open_dict(cfg):
                cfg.engine.update(embeddings_regularization=regularization)
    return cfg


def maybe_adjust_grace_epochs(cfg: DictConfig) -> DictConfig:
    embedding = cfg.get("embedding", None)
    if embedding is not None:
        forgetting_scheduler = embedding.get("forgetting_scheduler", None)
        if forgetting_scheduler is not None:
            grace_epochs = cfg.early_stopping.get("grace_epochs", 0)
            grace_epochs += (
                forgetting_scheduler.stop_after + forgetting_scheduler.warmup_for
            )
            with open_dict(cfg):
                cfg.early_stopping.grace_epochs = grace_epochs
    return cfg


#  Custom resolvers
OmegaConf.register_new_resolver("as_tuple", lambda *args: tuple(args))
OmegaConf.register_new_resolver(
    "math",
    lambda expr: math_eval(ast.parse(expr, mode="eval").body),
)
OmegaConf.register_new_resolver(
    "cache", lambda path: str(CACHE.joinpath(path).absolute())
)
