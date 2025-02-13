from src.nn.layers.embedding import IFNodeEmbedding
from src.nn.models.prototypes import TOnly, get_time_model


class RNNModel(TOnly):

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
        activation: str = "elu",
        dropout: float = 0.0,
        embeddings_dropout: float = 0.0,
    ):
        temporal_encoder = get_time_model(
            model_name="rnn",
            hidden_size=hidden_size,
            n_layers=time_layers,
            dropout=dropout,
        )
        self.temporal_layers = time_layers

        super().__init__(
            input_size=input_size,
            horizon=horizon,
            temporal_encoder=temporal_encoder,
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
