from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping


class EarlyStoppingWithGrace(EarlyStopping):

    def __init__(
        self,
        monitor: str,
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: float | None = None,
        divergence_threshold: float | None = None,
        check_on_train_epoch_end: bool | None = None,
        log_rank_zero_only: bool = False,
        grace_epochs: int = 0,
    ):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            strict=strict,
            check_finite=check_finite,
            stopping_threshold=stopping_threshold,
            divergence_threshold=divergence_threshold,
            check_on_train_epoch_end=check_on_train_epoch_end,
            log_rank_zero_only=log_rank_zero_only,
        )
        self.grace_epochs = grace_epochs
        self._grace_counter = 0

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        elif self._grace_counter < self.grace_epochs:
            self._grace_counter += 1
            return
        return super().on_train_epoch_end(trainer, pl_module)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        elif self._grace_counter < self.grace_epochs:
            self._grace_counter += 1
            return
        return super().on_validation_end(trainer, pl_module)
