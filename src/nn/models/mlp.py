from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch_geometric.typing import Adj

from src.nn.layers.embedding import IFNodeEmbedding
from src.nn.models.prototypes import STGNN


class MLPModel(STGNN):

    def __init__(
        self,
        input_size: int,
        window: int,
        horizon: int,
        n_nodes: int | None = None,
        output_size: int | None = None,
        exog_size: int = 0,
        hidden_size: int = 32,
        embedding: IFNodeEmbedding | None = None,
        add_embedding_before: str | list[str] | None = "encoding",
        use_local_weights: str | list[str] | None = None,
        n_layers: int = 1,
        activation: str = "elu",
        dropout: float = 0.0,
        embeddings_dropout: float = 0.0,
    ):
        super().__init__(
            input_size=input_size,
            horizon=horizon,
            n_nodes=n_nodes,
            output_size=output_size,
            exog_size=exog_size,
            hidden_size=hidden_size,
            embedding=embedding,
            add_embedding_before=add_embedding_before,
            use_local_weights=use_local_weights,
            activation=activation,
            dropout=dropout,
            embeddings_dropout=embeddings_dropout,
        )
        mlp = [
            Rearrange("b t n f -> b n (t f)"),
            nn.Linear(window * hidden_size, hidden_size),
        ]
        for _ in range(n_layers):
            mlp.append(nn.ReLU())
            if dropout > 0:
                mlp.append(nn.Dropout(dropout))
            mlp.append(nn.Linear(hidden_size, hidden_size))
        self.mlp = nn.Sequential(*mlp)

    def stmp(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: Tensor | None = None,
        emb: Tensor | None = None,
        node_mask: Tensor | None = None,
    ) -> Tensor:
        return self.mlp(x)
