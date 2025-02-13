from importlib.util import find_spec

import numpy as np
import wandb
from pytorch_lightning.loggers import NeptuneLogger, WandbLogger
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from tsl.ops.connectivity import convert_torch_connectivity

from src.engines.forgetting import schedulers
from src.nn.layers.embedding import IFNodeEmbedding
from src.nn.models.prototypes import STGNN


class EmbeddingLoggerMixin:

    def get_embeddings_history(self) -> list[np.ndarray]:
        return getattr(self, "embeddings_history", [])

    def get_embedding_module(self) -> IFNodeEmbedding | None:
        if isinstance(self.model, STGNN):
            return self.model.emb

    def get_numpy_embeddings(self) -> np.ndarray | None:
        emb_module = self.get_embedding_module()
        if emb_module is not None:
            return emb_module.get_emb().detach().cpu().clone().numpy()

    def log_embeddings(self, embedding_method: str = "spectral", n_components: int = 2):
        emb = self.get_numpy_embeddings()
        if hasattr(self, "embeddings_history"):
            self.embeddings_history.append(emb)
        if emb is not None and isinstance(self.logger, (NeptuneLogger, WandbLogger)):
            from matplotlib import pyplot as plt

            emb = self.reduce_embedding_dim(
                embeddings=emb,
                embedding_method=embedding_method,
                n_components=n_components,
            )
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(emb[:, 0], emb[:, 1])
            ax.set_title(f"Embeddings at step: {self.global_step:6d}")
            plt.tight_layout()
            if isinstance(self.logger, NeptuneLogger):
                self.logger.log_figure(fig, f"embeddings/step{self.global_step}")
            elif isinstance(self.logger, WandbLogger):
                if find_spec("plotly") is None:
                    fig = wandb.Image(fig)
                self.logger.experiment.log(
                    {"embeddings": fig, "step": self.global_step}
                )
            else:
                raise NotImplementedError(
                    "Only NeptuneLogger and WandbLogger are supported."
                )
            plt.close()

    def reduce_embedding_dim(
        self,
        embeddings: np.ndarray | list[np.ndarray],
        embedding_method: str = "spectral",
        n_components: int = 2,
    ):
        if not isinstance(embeddings, list):
            embeddings = [embeddings]
        assert all(
            emb.shape[-1] == embeddings[0].shape[-1] for emb in embeddings
        ), "All embeddings must have the same dimension."
        if embeddings[0].shape[-1] <= n_components:
            reduced_embeddings = embeddings
        elif embedding_method == "spectral":
            from sklearn.manifold import SpectralEmbedding

            reduced_embeddings = [
                SpectralEmbedding(n_components).fit_transform(emb) for emb in embeddings
            ]
        elif embedding_method == "tsne":
            from sklearn.manifold import TSNE

            reduced_embeddings = []
            init = "pca"
            for emb in embeddings:
                init = TSNE(n_components, init=init).fit_transform(emb)
                reduced_embeddings.append(init)
        else:
            raise NotImplementedError("Only 'spectral' and 'tsne' are supported.")
        return (
            reduced_embeddings if len(reduced_embeddings) > 1 else reduced_embeddings[0]
        )

    def plot_embeddings(
        self,
        filename: str,
        edge_index,
        embedding_method: str = "spectral",
        writer: str = "pillow",
    ):
        import networkx as nx
        from matplotlib import animation
        from matplotlib import pyplot as plt
        from matplotlib import rcParams
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        from torch_geometric.utils import to_networkx

        rcParams["font.size"] = 14
        # Reduce embeddings  ##################################################
        embeddings = self.reduce_embedding_dim(
            self.get_embeddings_history(), embedding_method, 2
        )
        # Generate graph  #####################################################
        n_iter, n_nodes = len(embeddings), len(embeddings[0])
        edge_index, _ = convert_torch_connectivity(
            edge_index, target_layout="edge_index"
        )
        edge_index, _ = remove_self_loops(edge_index)
        g = to_networkx(
            Data(edge_index=edge_index, pos=embeddings[0], num_nodes=n_nodes)
        )
        degree = [deg for node, deg in g.degree]
        # Create figure  ######################################################
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.subplots_adjust(left=0.01, right=1.05, top=0.95, bottom=0.01)
        # add colormap
        cmappable = ScalarMappable(
            norm=Normalize(min(degree), max(degree)), cmap="plasma"
        )
        fig.colorbar(
            cmappable,
            ax=ax,
            location="right",
            pad=0.02,
            label="Node degree",
            shrink=0.95,
        )
        # set boundaries
        embs = np.concatenate(embeddings, 0)
        min_x, min_y = np.min(embs, 0)
        max_x, max_y = np.max(embs, 0)
        std_x, std_y = np.std(embs, 0) * 0.2
        ax.set_xlim(min_x - std_x, max_x + std_x)
        ax.set_ylim(min_y - std_y, max_y + std_y)
        # Plot graph  #########################################################

        def update(i):
            ax.clear()
            curr_emb = embeddings[i]
            pos = dict(zip(g.nodes, curr_emb))
            nx.draw_networkx(
                g,
                pos=pos,
                ax=ax,
                font_size=6,
                width=0.1,
                cmap="plasma",
                node_color=degree,
                arrowsize=4,
                node_size=150,
                font_color="#fff",
            )
            ax.set_title(f"Embeddings at step: {i:6d}/{n_iter}")

        ani = animation.FuncAnimation(fig, update, frames=n_iter, interval=200)
        ani.save(filename, writer=writer)


class WithForgettingSchedulerMixin:

    def instantiate_forgetting_scheduler(self, **kwargs):
        emb_module = self.get_embedding_module()
        if kwargs and emb_module is not None:
            scheduler_class = schedulers[kwargs.pop("scheduler_class", "default")]
            forgetting_scheduler = scheduler_class(
                emb_module=emb_module,
                **kwargs,
            )
        else:
            forgetting_scheduler = None
        self.forgetting_scheduler = forgetting_scheduler
