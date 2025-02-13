import math
from abc import ABC, abstractmethod

import torch
from sklearn.cluster import KMeans
from torch import Tensor
from torch.nn import Parameter
from tsl.nn.layers import NodeEmbedding


def kl_divergence_normal(mu: Tensor, log_var: Tensor):
    # shape: [*, n, k]
    kl = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=-1)
    return torch.mean(kl)


def get_default_initializer_params(
    initializer: str | None, emb_size: int | None = None
) -> dict:
    if initializer in ("normal", "gaussian"):
        return {"mean": 0.0, "std": 0.2 ** (1 / 2)}
    elif initializer == "uniform" or initializer is None:
        bound = 1.0 / math.sqrt(emb_size)
        return {"low": -bound, "high": bound}
    else:
        raise ValueError(f"Unknown initializer: {initializer}")


class EmbeddingStrategy(ABC):

    @abstractmethod
    def apply_strategy(
        self,
        embeddings: Tensor,
        x: Tensor | None = None,
        mask: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        raise NotImplementedError

    def setup(self, embeddings: Tensor, **kwargs) -> Tensor:
        return embeddings


class ForgettingStrategy(EmbeddingStrategy, torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initializer = None
        self.initializer_params = None

    def setup(
        self,
        embeddings: Tensor,
        initializer: str | None = "normal",
        initializer_params: dict | None = None,
        **kwargs,
    ) -> Tensor:
        if self.initializer is None and self.initializer_params is None:
            emb_size = embeddings.shape[-1]
            if initializer_params is None:
                initializer_params = get_default_initializer_params(
                    initializer, emb_size
                )
            if initializer in ("normal", "gaussian"):
                initializer_params["log_var"] = math.log(initializer_params["std"] ** 2)
                del initializer_params["std"]

                def initializer_func(
                    strat: ForgettingStrategy, embs: Tensor
                ) -> Tensor:
                    return strat.initializer_params["mean"] + torch.exp(
                        strat.initializer_params["log_var"] / 2
                    ) * torch.randn_like(embs)

            elif initializer == "uniform" or initializer is None:

                def initializer_func(
                    strat: ForgettingStrategy, embs: Tensor
                ) -> Tensor:
                    return strat.initializer_params["low"] + torch.rand_like(embs) * (
                        strat.initializer_params["high"]
                        - strat.initializer_params["low"]
                    )

            else:
                raise ValueError(f"Unknown initializer: {initializer}")
            self.initializer = initializer_func
            self.initializer_params = torch.nn.ParameterDict(
                {k: torch.full((emb_size,), v) for k, v in initializer_params.items()}
            )
        return self(embeddings=embeddings)

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.initializer(self, embeddings)

    def apply_strategy(
        self, embeddings: Tensor, mask: Tensor | None = None, **kwargs
    ) -> Tensor:
        return self(embeddings=embeddings[mask])


class IFNodeEmbedding(NodeEmbedding):

    def __init__(
        self,
        n_nodes: int,
        emb_size: int,
        forgetting_strategy: EmbeddingStrategy | None = None,
        initializer: str = "normal",
        **kwargs,
    ):
        self.initializer_params = get_default_initializer_params(initializer, emb_size)
        super().__init__(
            n_nodes=n_nodes, emb_size=emb_size, initializer=initializer, **kwargs
        )
        self.forgetting_strategy = forgetting_strategy
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        with torch.no_grad():
            if self.forgetting_strategy is not None:
                setup_emb = self.forgetting_strategy.setup(
                    embeddings=self.emb,
                    initializer=self.initializer,
                    initializer_params=self.initializer_params,
                )
                self.emb.copy_(setup_emb)

    def reset_emb(self):
        with torch.no_grad():
            if self.initializer in ("normal", "gaussian"):
                self.emb.data.normal_(**self.initializer_params)
            elif self.initializer == "uniform" or self.initializer is None:
                self.emb.data.uniform_(
                    self.initializer_params["low"], self.initializer_params["high"]
                )
            else:
                super().reset_emb()

    def forward(
        self,
        x: Tensor | None = None,
        mask: Tensor | None = None,
        expand: list[int] | None = None,
        node_index: Tensor | None = None,
        nodes_first: bool = True,
        **kwargs,
    ) -> Tensor:
        emb = self.get_emb()
        if node_index is not None:
            emb = emb[node_index]
        if not nodes_first:
            emb = emb.T
        if expand is None:
            return emb
        shape = [*emb.size()]
        view = [1 if d > 0 else shape.pop(0 if nodes_first else -1) for d in expand]
        return emb.view(*view).expand(*expand)

    def forget(self, mask: Tensor | None = None, **kwargs) -> None:
        if self.forgetting_strategy is not None:
            new_emb = self.forgetting_strategy.apply_strategy(
                embeddings=self.emb, mask=mask, **kwargs
            )
            with torch.no_grad():
                if mask is not None:
                    self.emb.data[mask] = new_emb
                else:
                    self.emb.data = new_emb


class VariationalNodeEmbedding(IFNodeEmbedding):

    def __init__(
        self, n_nodes: int, emb_size: int, warmup: int = 5, beta: float = 0.05
    ):
        self.log_var = None
        super().__init__(n_nodes, emb_size, initializer="normal")
        self.warmup = warmup
        self.beta = beta
        self.log_var = Parameter(Tensor(self.n_nodes, self.emb_size))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.log_var is not None:
            with torch.no_grad():
                self.log_var.data.fill_(math.log(self.initializer_params["std"] ** 2))

    def forward(
        self,
        x: Tensor | None = None,
        mask: Tensor | None = None,
        expand: list[int] | None = None,
        node_index: Tensor | None = None,
        nodes_first: bool = True,
        **kwargs,
    ):
        """"""
        if self.training:
            mu, std = self.emb, torch.exp(self.log_var / 2)
            if node_index is not None:
                mu, std = mu[node_index], std[node_index]
            if not nodes_first:
                mu, std = mu.T, std.T
            if expand is not None:
                shape = [*mu.size()]
                view = [
                    1 if d > 0 else shape.pop(0 if nodes_first else -1) for d in expand
                ]
                mu = mu.view(*view).expand(*expand)
                std = std.view(*view).expand(*expand)
            return mu + std * torch.randn_like(std)
        else:
            return super().forward(
                expand=expand, node_index=node_index, nodes_first=nodes_first
            )


class ClusterizedNodeEmbedding(IFNodeEmbedding):
    def __init__(
        self,
        n_nodes: int,
        emb_size: int,
        n_clusters: int,
        tau: int = 1.0,
        estimator: str = "ste",
        learned_assignments: bool = True,
        requires_grad: bool = True,
        separation_loss: bool = False,
        sep_eps: float = 1.0,
        temp_annealing: bool = False,
        temp_decay_coeff: float = 0.99,
        warmup: int = 5,
        beta: float = 0.5,
    ):
        self.centroids, self.cluster_assignment = None, None
        super().__init__(
            n_nodes=n_nodes,
            emb_size=emb_size,
            initializer="uniform",
            requires_grad=requires_grad,
        )
        self.n_clusters = n_clusters
        self.estimator = estimator
        self.learned_assignments = learned_assignments
        self._tau = tau
        self.temp_annealing = temp_annealing
        self.temp_decay_coeff = temp_decay_coeff
        self.separation_loss = separation_loss
        self.max_sep = sep_eps
        self._frozen_centroids = False
        self.warmup = warmup
        self.beta = beta

        self.centroids = Parameter(
            Tensor(self.n_clusters, self.emb_size), requires_grad=requires_grad
        )
        if self.learned_assignments:
            self.cluster_assignment = Parameter(
                Tensor(n_nodes, n_clusters), requires_grad=requires_grad
            )
        else:
            self.register_parameter("cluster_assignment", None)
        self.reset_parameters()

    @property
    def tau(self):
        return self._tau

    def step(self):
        if self.temp_annealing:
            self._tau = max(self.temp_decay_coeff * self._tau, 0.0001)

    def extra_repr(self) -> str:
        return (
            f"n_tokens={self.n_tokens}, embedding_size={self.emb_size}, "
            f"n_clusters={self.n_clusters}"
        )

    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters()
        if self.centroids is not None:
            self.centroids.data.uniform_(
                self.initializer_params["low"], self.initializer_params["high"]
            )
        if self.cluster_assignment is not None:
            self.cluster_assignment.data.uniform_(0.0, 1.0)

    def assignment_logits(self):
        if self.learned_assignments:
            return self.cluster_assignment
        return torch.cdist(self.emb, self.centroids)

    def assigment_entropy(self):
        soft_assignments = self.get_assignment(estimator="soft")
        entropy = (
            -torch.sum(soft_assignments * torch.log(soft_assignments), -1)
        ).mean()
        return entropy

    def get_assignment(self, estimator=None):
        if estimator is None:
            estimator = self.estimator
        logits = self.assignment_logits()

        if estimator == "ste":
            soft_assignment = torch.softmax(logits / self.tau, -1)
            idx = torch.argmax(soft_assignment, -1)
            hard_assignment = torch.nn.functional.one_hot(
                idx, num_classes=self.n_clusters
            )
            return hard_assignment + soft_assignment - soft_assignment.detach()
        elif estimator == "gt":
            g = -torch.log(-torch.log(torch.rand_like(logits)))
            scores = logits / self.tau + g
            return torch.softmax(scores, -1)
        elif estimator == "soft":
            return torch.softmax(logits / self.tau, -1)
        else:
            raise NotImplementedError(f"{self.estimator} is not a valid a trick.")

    def init_centroids(self):
        if not self._frozen_centroids:
            kmeans = KMeans(n_clusters=self.n_clusters)
            X = self.emb.detach().cpu().numpy()
            kmeans.fit(X)
            centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
            self.centroids.data.copy_(centroids)
        if self.learned_assignments:
            dist = torch.cdist(self.emb, self.centroids)
            self.cluster_assignment.data.copy_(dist)

    def freeze_centroids(self):
        self._frozen_centroids = True

    def unfreeze_centroids(self):
        self._frozen_centroids = False

    def clustering_loss(self):
        assignment = self.get_assignment()
        node_centroid = torch.matmul(assignment, self.centroids)
        dist = torch.norm(self.emb - node_centroid, p=2, dim=-1)
        dist = dist.mean()
        if self.separation_loss:
            sep = torch.cdist(self.centroids, self.centroids)
            return (
                dist - torch.minimum(sep, self.max_sep * torch.ones_like(sep)).mean(),
                dist,
            )
        return dist, dist
