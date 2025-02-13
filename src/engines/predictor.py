from typing import Any, Callable, Mapping, Type

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torchmetrics import Metric, MetricCollection
from tsl.engines import Predictor as _Predictor_

from src.engines.mixin import EmbeddingLoggerMixin, WithForgettingSchedulerMixin
from src.nn.layers.embedding import (
    ClusterizedNodeEmbedding,
    VariationalNodeEmbedding,
    kl_divergence_normal,
)
from src.nn.models.prototypes import STGNN


class Predictor(_Predictor_, EmbeddingLoggerMixin, WithForgettingSchedulerMixin):
    def __init__(
        self,
        model: nn.Module | None = None,
        loss_fn: Callable | None = None,
        scale_target: bool = False,
        metrics: Mapping[str, Metric] | None = None,
        *,
        model_class: Type | None = None,
        model_kwargs: Mapping | None = None,
        optim_class: Type | None = None,
        optim_kwargs: Mapping | None = None,
        scheduler_class: Any | None = None,
        scheduler_kwargs: Mapping | None = None,
        forgetting_scheduler: dict | None = None,
        log_embeddings_every: int | None = None,  # epochs
        embeddings_regularization: dict[str, int | float | str] | None = None,
    ):
        super().__init__(
            model=model,
            model_class=model_class,
            model_kwargs=model_kwargs,
            optim_class=optim_class,
            optim_kwargs=optim_kwargs,
            loss_fn=loss_fn,
            scale_target=scale_target,
            metrics=metrics,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
        )
        self.embeddings_history = []
        self.log_embeddings_every = log_embeddings_every

        if forgetting_scheduler is None:
            forgetting_scheduler = {}
        self.reset_embedding_related_params = forgetting_scheduler.pop(
            "reset_embedding_related_params", False
        )
        self.instantiate_forgetting_scheduler(**forgetting_scheduler)

        self.embeddings_regularization = embeddings_regularization
        if self.embeddings_regularization is not None:
            assert (
                self.get_embedding_module() is not None
            ), "Model must have an embedding module to regularize embeddings"
            assert self.embeddings_regularization["method"] in (
                "l2",
                "dropout",
                "l1",
            ), "Embeddings regularization method must be either 'l1', 'l2' or 'dropout'"
            assert (
                self.embeddings_regularization["factor"] >= 0
            ), "Embeddings regularization factor must be non-negative"
            if self.embeddings_regularization["method"] == "dropout":
                self.set_embeddings_dropout(self.embeddings_regularization["factor"])

    @property
    def is_regularizing_embeddings(self):
        return (
            self.embeddings_regularization is not None
            and self.embeddings_regularization["factor"] > 0
        )

    def set_embeddings_dropout(self, dropout: float):
        if isinstance(self.model, STGNN):
            self.model.embeddings_dropout = dropout
        else:
            raise ValueError(
                "Model must be an instance of STGNN to set embeddings dropout"
            )

    def _set_metrics(self, metrics):
        self.train_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in metrics.items()},
            prefix="train/",
        )
        self.val_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in metrics.items()},
            prefix="val/",
        )
        self.test_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in metrics.items()},
            prefix="test/",
        )

    def log_loss(self, name, loss, **kwargs):
        """"""
        self.log(
            name + "/loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            **kwargs,
        )

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        norm_penalty = self.get_embeddings_norm_penalty()
        regularization_loss = self.regularization_loss(batch)
        if norm_penalty > 0:
            loss += norm_penalty
            self.log(
                f"regularization/{self.embeddings_regularization['method']}_penalty",
                norm_penalty,
                on_step=False,
                on_epoch=True,
                batch_size=batch.batch_size,
            )
        if regularization_loss is not None:
            loss += regularization_loss
        return loss

    def predict_batch(
        self,
        batch,
        preprocess: bool = False,
        postprocess: bool = True,
        return_target: bool = False,
        return_embeddings: bool = False,
        **forward_kwargs,
    ):
        inputs, targets, mask, transform = self._unpack_batch(batch)
        if preprocess:
            for key, trans in transform.items():
                if key in inputs:
                    inputs[key] = trans.transform(inputs[key])

        if forward_kwargs is None:
            forward_kwargs = dict()
        out = self.forward(**inputs, **forward_kwargs)
        y_hat, emb = out if isinstance(out, tuple) else (out, None)
        # Rescale outputs
        if postprocess:
            trans = transform.get("y")
            if trans is not None:
                y_hat = trans.inverse_transform(y_hat)
        if return_target:
            y = targets.get("y")
            result = [y, y_hat, mask]
        else:
            result = [y_hat]
        if return_embeddings:
            result.append(emb)
        return tuple(result) if len(result) > 1 else result[0]

    def get_embeddings_norm_penalty(self):
        if self.is_regularizing_embeddings:
            llambda = self.embeddings_regularization["factor"]
            if self.embeddings_regularization["method"] == "l2":
                return (
                    llambda * torch.linalg.vector_norm(self.model.emb.emb, ord=2) ** 2
                )
            elif self.embeddings_regularization["method"] == "l1":
                return llambda * torch.linalg.vector_norm(self.model.emb.emb, ord=1)
        return 0.0

    def regularization_loss(self, batch) -> torch.Tensor | None:
        reg = None
        embedding_module = self.get_embedding_module()
        if isinstance(embedding_module, VariationalNodeEmbedding):
            beta = (
                0
                if self.current_epoch < embedding_module.warmup
                else embedding_module.beta
            )
            mu = embedding_module.emb
            log_var = embedding_module.log_var
            kl = kl_divergence_normal(mu, log_var)
            self.log(
                "regularization/variational/kl_divergence",
                kl,
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=False,
                batch_size=batch.batch_size,
            )
            reg = beta * kl
        elif isinstance(embedding_module, ClusterizedNodeEmbedding):
            beta = (
                0
                if self.current_epoch < embedding_module.warmup
                else embedding_module.beta
            )
            clustering_loss, avg_dist = embedding_module.clustering_loss()
            self.log(
                "regularization/cluster/clustering_loss",
                clustering_loss.detach(),
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=False,
                batch_size=batch.batch_size,
            )
            self.log(
                "regularization/cluster/assignment_temp",
                embedding_module._tau,
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=False,
            )
            self.log(
                "regularization/cluster/average_centroid_distance",
                avg_dist,
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=False,
                batch_size=batch.batch_size,
            )
            if self.log_embeddings_every is not None:
                # log average distance among centroids
                sep = torch.cdist(
                    embedding_module.centroids, embedding_module.centroids
                ).mean()
                self.log(
                    "regularization/cluster/separation_l2",
                    sep.mean(),
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    prog_bar=False,
                    batch_size=batch.batch_size,
                )
                # log entropy of the cluster assignments distribution
                entropy = embedding_module.assigment_entropy()
                self.log(
                    "regularization/cluster/entropy",
                    entropy,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    prog_bar=False,
                    batch_size=batch.batch_size,
                )
                # log number of cluster members for each cluster
                assignments = embedding_module.get_assignment(estimator="ste")
                counts = assignments.sum(0)
                for i, c in enumerate(counts):
                    self.log(
                        f"regularization/cluster/members_cluster{i}",
                        c,
                        on_step=True,
                        logger=True,
                    )
            reg = beta * clustering_loss
        return reg

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        # Log embeddings
        if (self.log_embeddings_every is not None) and (
            self.current_epoch % self.log_embeddings_every == 0
        ):
            self.log_embeddings()

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        if self.forgetting_scheduler is not None:
            n_forgotten, forgot_mask = self.forgetting_scheduler.step(return_mask=True)
            if (
                self.reset_embedding_related_params
                and n_forgotten > 0
                and isinstance(self.model, STGNN)
            ):
                self.model.reset_embedding_related_layers(emb_mask=forgot_mask)
            self.log(
                "train/forget_count",
                n_forgotten,
                on_step=False,
                on_epoch=True,
                batch_size=batch.batch_size,
            )

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        if self.is_regularizing_embeddings:
            stop_after = self.embeddings_regularization.get("stop_after", None)
            if stop_after is not None and self.current_epoch >= stop_after:
                self.embeddings_regularization["factor"] = 0.0
                if self.embeddings_regularization["method"] == "dropout":
                    self.set_embeddings_dropout(0.0)
