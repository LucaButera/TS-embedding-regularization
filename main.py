from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger

from env import CACHE, HOME, REPRODUCIBLE, rng
from utils import (
    instantiate_data,
    maybe_add_regularization,
    maybe_adjust_grace_epochs,
    maybe_update_logger_cfg,
    parse_hyperparams,
    update_config_from_data,
)


@hydra.main(config_path="conf", config_name="experiment", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.get("seed", None) is not None and REPRODUCIBLE:
        seed_everything(cfg.seed)
        rng.seed(cfg.seed)
    if cfg.get("num_threads", None) is not None:
        torch.set_num_threads(cfg.num_threads)
    cfg = maybe_add_regularization(cfg)
    cfg = maybe_adjust_grace_epochs(cfg)

    datamodule = instantiate_data(cfg.data)
    datamodule.setup()  # call setup as we need to know the number of batches
    cfg = update_config_from_data(cfg, datamodule)
    cfg = maybe_update_logger_cfg(cfg)
    engine = instantiate(cfg.engine)
    try:
        trainer = instantiate(cfg.trainer)
        trainer.logger.log_hyperparams(parse_hyperparams(cfg))
        getattr(trainer, cfg.mode)(engine, datamodule=datamodule)
        if cfg.mode == "fit":
            trainer.test(datamodule=datamodule)
        trainer.logger.finalize("success")
    finally:
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.finish()


if __name__ == "__main__":
    map(lambda path: Path(path).mkdir(exist_ok=True, parents=True), [HOME, CACHE])
    main()
