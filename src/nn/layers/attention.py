from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from tsl.nn.blocks import RNNBase
from tsl.nn.layers import (
    Activation,
    GraphGRUCellBase,
    Lambda,
    SpatialSelfAttention,
    TemporalSelfAttention,
)


class SpatioTemporalAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        in_channels: int | None = None,
        dropout: float = 0.0,
        activation: str = "elu",
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.temporal_attn = TemporalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            in_channels=in_channels,
            dropout=dropout,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.spatial_attn = SpatialSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            in_channels=None,  # channels is always embed_dim of temporal attn
            dropout=dropout,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.activation = Activation(activation)

    def forward(self, x, **kwargs):
        # x: [batch, steps, nodes, features]
        x = x + self.temporal_attn(x, need_weights=False)[0]
        x = x + self.spatial_attn(x, need_weights=False)[0]
        return self.activation(x)


class SpatioTemporalAttentionBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        input_size: int | None = None,
        dropout: float = 0.0,
        activation: str = "elu",
        n_layers: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn_layers = nn.ModuleList(
            [
                SpatioTemporalAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    in_channels=input_size if i == 0 else None,
                    dropout=dropout,
                    activation=activation,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, x, *args, **kwargs):
        for layer in self.attn_layers:
            x = layer(x)
        return x


class TemporalAttentionBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        input_size: int | None = None,
        dropout: float = 0.0,
        activation: str = "elu",
        n_layers: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
        pool_last_layer: bool = False,
        pooling_window: int = 12,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn_layers = nn.ModuleList(
            [
                TemporalAttentionLayer(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    in_channels=input_size if i == 0 else None,
                    dropout=dropout,
                    activation=activation,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                )
                for i in range(n_layers)
            ]
        )
        self.pool_last_layer = pool_last_layer
        self.pooling_window = pooling_window
        if self.pool_last_layer:
            self.pooling_layer = nn.Sequential(
                Lambda(lambda x: x[:, -pooling_window:]),
                Rearrange("b t n f -> b n (f t)"),
                nn.Linear(hidden_size * pooling_window, hidden_size),
            )

    def forward(self, x, *args, **kwargs):
        for layer in self.attn_layers:
            x = layer(x)
        if self.pool_last_layer:
            x = self.pooling_layer(x)
        return x


class SpatialAttentionLayer(SpatialSelfAttention):

    def __init__(
        self,
        embed_dim: int,
        in_channels: int,
        num_heads: int,
        dropout: float = 0.0,
        activation: str = "elu",
        bias: bool = True,
        device=None,
        dtype=None,
        add_time_dim: bool = False,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            in_channels=in_channels,
            dropout=dropout,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.activation = Activation(activation)
        self.add_time_dim = add_time_dim

    def forward(self, x, *args, **kwargs):
        added_time_dim = False
        if self.add_time_dim and x.ndim == 3:
            x = rearrange(x, "b n f -> b 1 n f")
            added_time_dim = True
        out = self.activation(super().forward(x, need_weights=False)[0])
        if added_time_dim:
            out = rearrange(out, "b 1 n f -> b n f")
        return out


class TemporalAttentionLayer(TemporalSelfAttention):

    def __init__(
        self,
        embed_dim: int,
        in_channels: int,
        num_heads: int,
        dropout: float = 0.0,
        activation: str = "elu",
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            in_channels=in_channels,
            dropout=dropout,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.activation = Activation(activation)

    def forward(self, x, *args, **kwargs):
        return self.activation(super().forward(x, need_weights=False)[0])


class TransformerGRUCell(GraphGRUCellBase):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: str = "leaky_relu",
        num_heads: int = 1,
        dropout: float = 0.0,
    ):
        self.input_size = input_size
        forget_gate = SpatialAttentionLayer(
            in_channels=input_size + hidden_size,
            embed_dim=hidden_size,
            activation=activation,
            num_heads=num_heads,
            add_time_dim=True,
            dropout=dropout,
        )
        update_gate = SpatialAttentionLayer(
            in_channels=input_size + hidden_size,
            embed_dim=hidden_size,
            activation=activation,
            num_heads=num_heads,
            add_time_dim=True,
            dropout=dropout,
        )
        candidate_gate = SpatialAttentionLayer(
            in_channels=input_size + hidden_size,
            embed_dim=hidden_size,
            activation=activation,
            num_heads=num_heads,
            add_time_dim=True,
            dropout=dropout,
        )
        super().__init__(
            hidden_size=hidden_size,
            forget_gate=forget_gate,
            update_gate=update_gate,
            candidate_gate=candidate_gate,
        )


class TransformerGRU(RNNBase):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int = 1,
        cat_states_layers: bool = False,
        return_only_last_state: bool = False,
        activation: str = "leaky_relu",
        num_heads: int = 1,
        dropout: float = 0.0,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        rnn_cells = [
            TransformerGRUCell(
                input_size=input_size if i == 0 else hidden_size,
                hidden_size=hidden_size,
                activation=activation,
                num_heads=num_heads,
                dropout=dropout,
            )
            for i in range(n_layers)
        ]
        super().__init__(
            cells=rnn_cells,
            cat_states_layers=cat_states_layers,
            return_only_last_state=return_only_last_state,
        )
