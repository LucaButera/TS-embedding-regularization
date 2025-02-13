from tsl.utils import ensure_list

from src.nn.layers.attention import SpatialAttentionLayer
from src.nn.layers.embedding import IFNodeEmbedding
from src.nn.models.prototypes import TTS, get_time_model


class TTSTransformerModel(TTS):

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
        spatial_layers: int = 1,
        num_heads: int = 1,
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
        sp_emb_size = (
            0
            if (embedding is None or "message_passing" not in add_embedding_before)
            else embedding.emb_size
        )
        # Using None defaults to input size same as hidden and no linear is used
        in_channels = hidden_size + sp_emb_size if sp_emb_size != 0 else None
        sp_layers = [
            SpatialAttentionLayer(
                in_channels=in_channels,
                embed_dim=hidden_size,
                num_heads=num_heads,
                activation=activation,
                add_time_dim=True,
                dropout=dropout,
            )
            for _ in range(spatial_layers)
        ]
        super().__init__(
            input_size=input_size,
            horizon=horizon,
            temporal_encoder=temporal_encoder,
            spatial_encoder=sp_layers,
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
