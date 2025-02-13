from tsl.utils import ensure_list

from src.nn.layers.embedding import IFNodeEmbedding
from src.nn.layers.graph_conv import DiffConv, GraphConv
from src.nn.models.prototypes import TTS, get_time_model


class TTSIMPModel(TTS):
    def __init__(
        self,
        input_size: int,
        horizon: int,
        n_nodes: int | None = None,
        output_size: int | None = None,
        exog_size: int = 0,
        hidden_size: int = 32,
        embedding: IFNodeEmbedding | None = None,
        add_embedding_before: str | list[str] | None = "encoding",
        use_local_weights: str | list[str] | None = None,
        time_layers: int = 1,
        graph_layers: int = 1,
        root_weight: bool = True,
        norm: str = "none",
        add_backward: bool = False,
        cached: bool = False,
        activation: str = "elu",
        time_model: str = "rnn",
        dropout: float = 0.0,
        embeddings_dropout: float = 0.0,
    ):
        temporal_encoder = get_time_model(
            model_name=time_model,
            hidden_size=hidden_size,
            n_layers=time_layers,
            dropout=dropout,
        )
        self.temporal_layers = time_layers

        add_embedding_before = ensure_list(add_embedding_before)

        mp_kwargs = dict(
            root_weight=root_weight, activation=activation, dropout=dropout
        )
        if add_backward:
            assert (
                norm == "asym"
            ), "DiffConv only supports asym norm when add_backward=True."
            mp_conv = DiffConv
            mp_kwargs.update(k=1, add_backward=add_backward)
        else:
            mp_conv = GraphConv
            mp_kwargs.update(norm=norm, cached=cached)
        mp_emb_size = (
            0
            if (embedding is None or "message_passing" not in add_embedding_before)
            else embedding.emb_size
        )
        mp_layers = [
            mp_conv(hidden_size + mp_emb_size, hidden_size, **mp_kwargs)
            for _ in range(graph_layers)
        ]
        super().__init__(
            input_size=input_size,
            horizon=horizon,
            temporal_encoder=temporal_encoder,
            spatial_encoder=mp_layers,
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
