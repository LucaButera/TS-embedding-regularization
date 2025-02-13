import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from tsl.nn.blocks import RNNBase
from tsl.nn.blocks.encoders.tcn import TemporalConvNet as _TemporalConvNet_
from tsl.nn.layers import Activation, Dense, GraphGRUCellBase, Lambda

from src.nn.layers.grcnn import GraphConvGRUCell, GraphConvLSTMCell


class AMPConv(MessagePassing):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        edge_dim: int | None = None,
        activation: str = "leaky_relu",
        dropout: float = 0.0,
    ):
        super().__init__(aggr="add", node_dim=-2)

        self.in_channels = input_size
        self.out_channels = output_size
        self.dropout = dropout

        self.msg_mlp = [
            nn.Linear(2 * input_size, output_size),
            Activation(activation),
            nn.Linear(output_size, output_size),
        ]
        if self.dropout > 0:
            self.msg_mlp.insert(2, nn.Dropout(self.dropout))
        self.msg_mlp = nn.Sequential(*self.msg_mlp)

        edge_dim = edge_dim or 0
        if edge_dim > 0:
            self.lin_edge = nn.Linear(edge_dim, output_size)
            if self.dropout > 0:
                self.lin_edge = nn.Sequential(self.lin_edge, nn.Dropout(self.dropout))
        else:
            self.register_parameter("lin_edge", None)

        self.gate_mlp = Dense(
            output_size, 1, activation="sigmoid", dropout=self.dropout
        )

        self.skip_conn = nn.Linear(input_size, output_size)
        if dropout > 0:
            self.skip_conn = nn.Sequential(self.skip_conn, nn.Dropout(self.dropout))
        self.activation = Activation(activation)

    def forward(self, x, edge_index, edge_attr: Tensor | None = None):
        """"""
        if edge_attr is not None:
            if edge_attr.ndim == 1:  # accommodate for edge_index
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        out = self.activation(out + self.skip_conn(x))

        return out

    def message(self, x_i, x_j, edge_attr):
        mij = self.msg_mlp(torch.cat([x_i, x_j], -1))
        if edge_attr is not None:
            mij = mij + edge_attr
        return self.gate_mlp(mij) * mij


class AMPGRUCell(GraphGRUCellBase):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        edge_dim: int | None = None,
        activation: str = "leaky_relu",
        dropout: float = 0.0,
    ):
        self.input_size = input_size
        # instantiate gates
        forget_gate = AMPConv(
            input_size=input_size + hidden_size,
            output_size=hidden_size,
            edge_dim=edge_dim,
            activation=activation,
            dropout=dropout,
        )
        update_gate = AMPConv(
            input_size=input_size + hidden_size,
            output_size=hidden_size,
            edge_dim=edge_dim,
            activation=activation,
            dropout=dropout,
        )
        candidate_gate = AMPConv(
            input_size=input_size + hidden_size,
            output_size=hidden_size,
            edge_dim=edge_dim,
            activation=activation,
            dropout=dropout,
        )
        super().__init__(
            hidden_size=hidden_size,
            forget_gate=forget_gate,
            update_gate=update_gate,
            candidate_gate=candidate_gate,
        )


class AMPGRU(RNNBase):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        edge_dim: int | None = None,
        n_layers: int = 1,
        cat_states_layers: bool = False,
        return_only_last_state: bool = False,
        activation: str = "leaky_relu",
        dropout: float = 0.0,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        rnn_cells = [
            AMPGRUCell(
                input_size=input_size if i == 0 else hidden_size,
                hidden_size=hidden_size,
                edge_dim=edge_dim,
                activation=activation,
                dropout=self.dropout,
            )
            for i in range(n_layers)
        ]
        super().__init__(
            cells=rnn_cells,
            cat_states_layers=cat_states_layers,
            return_only_last_state=return_only_last_state,
        )


class GraphConvRNN(RNNBase):
    # Fixed for norm usage instead of bool asym_norm

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int = 1,
        cat_states_layers: bool = False,
        return_only_last_state: bool = False,
        cell: str = "gru",
        bias: bool = True,
        norm: str = "asym",
        root_weight: bool = True,
        activation: str = None,
        cached: bool = False,
        dropout: float = 0.0,
        **kwargs,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size

        if cell == "gru":
            cell = GraphConvGRUCell
        elif cell == "lstm":
            cell = GraphConvLSTMCell
        else:
            raise NotImplementedError(f'"{cell}" cell not implemented.')

        rnn_cells = [
            cell(
                input_size if i == 0 else hidden_size,
                hidden_size,
                root_weight=root_weight,
                activation=activation,
                bias=bias,
                cached=cached,
                norm=norm,
                dropout=dropout,
                **kwargs,
            )
            for i in range(n_layers)
        ]
        super().__init__(
            cells=rnn_cells,
            cat_states_layers=cat_states_layers,
            return_only_last_state=return_only_last_state,
        )


class TemporalConvNet(_TemporalConvNet_):

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation: int,
        stride: int = 1,
        exog_channels: int | None = None,
        output_channels: int | None = None,
        n_layers: int = 1,
        gated: bool = False,
        dropout: float = 0.0,
        activation: str = "relu",
        exponential_dilation: bool = False,
        weight_norm: bool = False,
        causal_padding: bool = True,
        bias: bool = True,
        channel_last: bool = True,
        pool_last_layer: bool = False,
        pooling_window: int = 12,
    ):
        super().__init__(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            exog_channels=exog_channels,
            output_channels=output_channels,
            n_layers=n_layers,
            gated=gated,
            dropout=dropout,
            activation=activation,
            exponential_dilation=exponential_dilation,
            weight_norm=weight_norm,
            causal_padding=causal_padding,
            bias=bias,
            channel_last=channel_last,
        )
        self.pool_last_layer = pool_last_layer
        self.pooling_window = pooling_window
        if self.pool_last_layer:
            self.pooling_layer = nn.Sequential(
                Lambda(lambda x: x[:, -pooling_window:]),
                Rearrange("b t n f -> b n (f t)"),
                nn.Linear(hidden_channels * pooling_window, hidden_channels),
            )

    def forward(self, x, u=None):
        x = super().forward(x=x, u=u)
        if self.pool_last_layer:
            x = self.pooling_layer(x)
        return x
