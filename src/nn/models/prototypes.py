import torch
from torch import Tensor, nn
from torch_geometric.typing import Adj
from tsl import logger
from tsl.nn.blocks import RNN
from tsl.nn.blocks.decoders import MLPDecoder
from tsl.nn.layers.multi import MultiLinear
from tsl.nn.models import BaseModel
from tsl.nn.utils import maybe_cat_exog
from tsl.utils import ensure_list

from src.nn.layers.attention import TemporalAttentionBlock
from src.nn.layers.conv import TemporalConvNet
from src.nn.layers.embedding import IFNodeEmbedding
from src.nn.layers.multi import MultiMLPDecoder
from src.nn.models.utils import maybe_cat_emb, partially_reset_linear


def get_time_model(
    model_name: str, hidden_size: int, n_layers: int, dropout: float = 0.0
) -> nn.Module:
    if model_name == "rnn":
        model = RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            return_only_last_state=True,
            cell="gru",
            dropout=dropout,
        )
    elif model_name == "tcn":
        model = TemporalConvNet(
            input_channels=hidden_size,
            hidden_channels=hidden_size,
            kernel_size=3,
            dilation=2,
            n_layers=n_layers,
            gated=True,
            exponential_dilation=True,
            pool_last_layer=True,
            pooling_window=12,
            dropout=dropout,
        )
        logger.warning(
            "TemporalConvNet encoder pools only the last 12 time-steps, "
            "ensure the receptive field is large enough."
        )
    elif model_name == "transformer":
        model = TemporalAttentionBlock(
            hidden_size=hidden_size,
            num_heads=4,
            n_layers=n_layers,
            pool_last_layer=True,
            pooling_window=12,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown time model: {model_name}")
    return model


class STGNN(BaseModel):
    available_embedding_pos = {"encoding", "decoding"}
    available_local_weights_blocks = {"encoder", "decoder"}

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
        activation: str = "elu",
        dropout: float = 0.0,
        embeddings_dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.horizon = horizon
        self.n_nodes = n_nodes
        self.output_size = output_size or input_size
        self.hidden_size = hidden_size
        self.exog_size = exog_size
        self.activation = activation
        self.dropout = dropout
        self.embeddings_dropout = embeddings_dropout
        # EMBEDDING
        self.add_embedding_before = (
            set()
            if add_embedding_before is None
            else set(ensure_list(add_embedding_before))
        )
        if not self.add_embedding_before.issubset(self.available_embedding_pos):
            raise ValueError(
                "Parameter 'add_embedding_before' must be a "
                f"subset of {self.available_embedding_pos}"
            )
        if embedding is not None:
            self.emb = embedding
            emb_size = self.emb.emb_size
        else:
            self.register_module("emb", None)
            emb_size = 0
        # ENCODER
        self.encoder_input = input_size + exog_size
        if "encoding" in self.add_embedding_before and self.emb is not None:
            self.encoder_input += emb_size
        self.use_local_weights = (
            set() if use_local_weights is None else set(ensure_list(use_local_weights))
        )
        if not self.use_local_weights.issubset(self.available_local_weights_blocks):
            raise ValueError(
                "Parameter 'use_local_weights' must be a "
                f"subset of {self.available_local_weights_blocks}"
            )
        if "encoder" in self.use_local_weights:
            self.encoder = MultiLinear(self.encoder_input, hidden_size, n_nodes)
        else:
            self.encoder = nn.Linear(self.encoder_input, hidden_size)
        if self.dropout > 0:
            self.encoder = nn.Sequential(self.encoder, nn.Dropout(p=self.dropout))
        # DECODER
        self.decoder_input = hidden_size
        if "decoding" in self.add_embedding_before and self.emb is not None:
            self.decoder_input += emb_size
        if "decoder" in self.use_local_weights:
            self.decoder = MultiMLPDecoder(
                input_size=self.decoder_input,
                n_instances=n_nodes,
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                horizon=self.horizon,
                activation=self.activation,
                dropout=self.dropout,
            )
        else:
            self.decoder = MLPDecoder(
                input_size=self.decoder_input,
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                horizon=self.horizon,
                activation=self.activation,
                dropout=self.dropout,
            )

    def reset_embeddings(
        self,
        n_nodes: int | None = None,
        requires_grad: bool = True,
        from_learned: bool = False,
    ):
        if self.emb is not None:
            if n_nodes is not None:
                with torch.no_grad():
                    self.emb.n_nodes = n_nodes
                    if from_learned:
                        emb = torch.distributions.Normal(
                            self.emb.emb.mean(dim=0),
                            self.emb.emb.std(dim=0),
                        ).sample((n_nodes,))
                        self.emb.emb = nn.Parameter(emb, requires_grad=requires_grad)
                    else:
                        self.emb.emb = nn.Parameter(
                            torch.Tensor(n_nodes, self.emb.emb_size),
                            requires_grad=requires_grad,
                        )
                        self.emb.reset_parameters()
            else:
                self.emb.reset_parameters()

    def reset_local_layers(self):
        for layer in ["encoder", "decoder"]:
            if layer in self.use_local_weights:
                getattr(self, layer).reset_parameters()

    def reset_embedding_related_layers(self, emb_mask: Tensor | None = None):
        if self.emb is None:
            return
        for stage in self.add_embedding_before:
            layer = stage[:-2] + "ing"  # change encod(decod)ing to encod(decod)er
            if layer in self.use_local_weights:
                raise NotImplementedError(
                    f"Partially resetting local {layer} is not implemented."
                )
            else:
                if stage == "encoding":
                    # reset last n==emb_size columns of weight
                    linear = self.encoder[0] if self.dropout > 0 else self.encoder
                    reset_mask = torch.zeros(linear.weight.shape[1], dtype=torch.bool)
                    reset_mask[-self.emb.emb_size :] = True
                    # works also for sequential with linear as first layer
                    partially_reset_linear(
                        self.encoder[0] if self.dropout > 0 else self.encoder,
                        in_mask=reset_mask,
                    )
                    print("a")
                elif stage == "decoding":
                    reset_mask = torch.zeros(
                        self.decoder.readout.mlp[0].affinity.weight.shape[1],
                        dtype=torch.bool,
                    )
                    reset_mask[-self.emb.emb_size :] = True
                    partially_reset_linear(
                        self.decoder.readout.mlp[0].affinity, in_mask=reset_mask
                    )
                else:
                    raise NotImplementedError(
                        f"Partially resetting stage {stage} is not implemented."
                    )

    def stmp(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: Tensor | None = None,
        emb: Tensor | None = None,
        node_mask: Tensor | None = None,
    ) -> Tensor:
        raise NotImplementedError

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: Tensor | None = None,
        u: Tensor | None = None,
        node_mask: Tensor | None = None,
        node_idx: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        # x: [batches steps nodes features]
        x = maybe_cat_exog(x, u)
        if self.emb is not None:
            emb = self.emb(
                x=x, mask=node_mask, expand=[x.size(0), -1, -1], node_index=node_idx
            )
            if self.embeddings_dropout > 0:
                emb = nn.functional.dropout(
                    emb, p=self.embeddings_dropout, training=self.training
                )
        else:
            emb = None
        if "encoding" in self.add_embedding_before and emb is not None:
            x = maybe_cat_emb(x, emb)
        # ENCODER   ###########################################################
        x = self.encoder(x)
        # SPATIOTEMPORAL MESSAGE-PASSING   ####################################
        out = self.stmp(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            emb=emb,
            node_mask=node_mask,
        )
        # DECODER   ###########################################################
        if "decoding" in self.add_embedding_before:
            out = maybe_cat_emb(out, emb)
        out = self.decoder(out)
        return out, emb


class TOnly(STGNN):

    def __init__(
        self,
        input_size: int,
        horizon: int,
        temporal_encoder: nn.Module,
        n_nodes: int | None = None,
        output_size: int | None = None,
        exog_size: int = 0,
        hidden_size: int = 32,
        embedding: IFNodeEmbedding | None = None,
        add_embedding_before: str | list[str] | None = "encoding",
        use_local_weights: str | list[str] | None = None,
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
        # STMP
        self.temporal_encoder = temporal_encoder
        self.spatial_layers = 0

    def stmp(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: Tensor | None = None,
        emb: Tensor | None = None,
        node_mask: Tensor | None = None,
    ) -> Tensor:
        # temporal encoding
        return self.temporal_encoder(x)


class TTS(STGNN):
    available_embedding_pos = {"encoding", "message_passing", "decoding"}

    def __init__(
        self,
        input_size: int,
        horizon: int,
        temporal_encoder: nn.Module,
        spatial_encoder: nn.Module | list[nn.Module],
        n_nodes: int | None = None,
        output_size: int | None = None,
        exog_size: int = 0,
        hidden_size: int = 32,
        embedding: IFNodeEmbedding | None = None,
        add_embedding_before: str | list[str] | None = "encoding",
        use_local_weights: str | list[str] | None = None,
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
        # STMP
        self.temporal_encoder = temporal_encoder
        if not isinstance(spatial_encoder, nn.ModuleList):
            spatial_encoder = nn.ModuleList(ensure_list(spatial_encoder))
        self.mp_layers = spatial_encoder
        self.spatial_layers = len(self.mp_layers)

    def stmp(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: Tensor | None = None,
        emb: Tensor | None = None,
        node_mask: Tensor | None = None,
    ) -> Tensor:
        # temporal encoding
        out = self.temporal_encoder(x)
        # spatial encoding
        for layer in self.mp_layers:
            if "message_passing" in self.add_embedding_before:
                out = maybe_cat_emb(out, emb)
            out = layer(out, edge_index, edge_weight)
        return out


class TAS(STGNN):

    def __init__(
        self,
        input_size: int,
        horizon: int,
        stmp_conv: nn.Module,
        n_nodes: int = None,
        output_size: int = None,
        exog_size: int = 0,
        hidden_size: int = 32,
        embedding: IFNodeEmbedding | None = None,
        add_embedding_before: str | list[str] | None = "encoding",
        use_local_weights: str | list[str] | None = None,
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

        # STMP
        self.stmp_conv = stmp_conv

    def stmp(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: Tensor | None = None,
        emb: Tensor | None = None,
        node_mask: Tensor | None = None,
    ) -> Tensor:
        # spatiotemporal encoding
        out = self.stmp_conv(x, edge_index, edge_weight)
        return out
