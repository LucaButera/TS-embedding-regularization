from typing import List

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul
from tsl.nn.layers.graph_convs.mixin import NormalizedAdjacencyMixin
from tsl.nn.utils import get_functional_activation
from tsl.ops.connectivity import asymmetric_norm, transpose


class GraphConv(MessagePassing, NormalizedAdjacencyMixin):
    r"""A simple graph convolutional operator where the message function is a
    simple linear projection and aggregation a simple average. In other terms:

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1} \mathbf{\tilde{A}}
        \mathbf{X} \boldsymbol{\Theta} + \mathbf{b} .

    Args:
        input_size (int): Size of the input features.
        output_size (int): Size of the output features.
        bias (bool): If :obj:`False`, then the layer will not learn an additive
            bias vector.
            (default: :obj:`True`)
        norm (str): The normalization used for edges and edge weights. If
            :obj:`'mean'`, then edge weights are normalized as
            :math:`a_{j \rightarrow i} =  \frac{a_{j \rightarrow i}} {deg_{i}}`,
            other available options are: :obj:`'gcn'`, :obj:`'asym'` and
            :obj:`'none'`.
            (default: :obj:`'mean'`)
        root_weight (bool): If :obj:`True`, then add a linear layer for the root
            node itself (a skip connection).
            (default :obj:`True`)
        activation (str, optional): Activation function to be used, :obj:`None`
            for identity function (i.e., no activation).
            (default: :obj:`None`)
        cached (bool): If :obj:`True`, then cached the normalized edge weights
            computed in the first call.
            (default :obj:`False`)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        norm: str = "mean",
        root_weight: bool = True,
        activation: str = None,
        cached: bool = False,
        dropout: float = 0.0,
    ):
        super(GraphConv, self).__init__(aggr="add", node_dim=-2)

        self.in_channels = input_size
        self.out_channels = output_size
        self.norm = norm
        self.cached = cached
        self.activation = get_functional_activation(activation)
        self.dropout = dropout

        self.lin = nn.Linear(input_size, output_size, bias=False)
        if self.dropout > 0:
            self.lin = nn.Sequential(self.lin, nn.Dropout(self.dropout))

        if root_weight:
            self.root_lin = nn.Linear(input_size, output_size, bias=False)
            if self.dropout > 0:
                self.root_lin = nn.Sequential(self.root_lin, nn.Dropout(self.dropout))
        else:
            self.register_parameter("root_lin", None)

        if bias:
            self.bias = Parameter(torch.Tensor(output_size))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.dropout > 0:
            self.lin[0].reset_parameters()
            if self.root_lin is not None:
                self.root_lin[0].reset_parameters()
        else:
            self.lin.reset_parameters()
            if self.root_lin is not None:
                self.root_lin.reset_parameters()
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(
        self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None
    ) -> Tensor:
        """"""
        out = self.lin(x)

        edge_index, edge_weight = self.normalize_edge_index(
            x, edge_index=edge_index, edge_weight=edge_weight, use_cached=self.cached
        )

        out = self.propagate(edge_index, x=out, edge_weight=edge_weight)

        if self.root_lin is not None:
            out += self.root_lin(x)

        if self.bias is not None:
            out += self.bias

        return self.activation(out)

    def message(self, x_j: Tensor, edge_weight) -> Tensor:
        """"""
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        """"""
        # adj_t: SparseTensor [nodes, nodes]
        # x: [(batch,) nodes, channels]
        return matmul(adj_t, x, reduce=self.aggr)


class DiffConv(MessagePassing):
    r"""The Diffusion Convolution Layer from the paper `"Diffusion Convolutional
    Recurrent Neural Network: Data-Driven Traffic Forecasting"
    <https://arxiv.org/abs/1707.01926>`_ (Li et al., ICLR 2018).

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        k (int): Filter size :math:`K`.
        root_weight (bool): If :obj:`True`, then add a filter also for the
            :math:`0`-order neighborhood (i.e., the root node itself).
            (default :obj:`True`)
        add_backward (bool): If :obj:`True`, then additional :math:`K` filters
            are learnt for the transposed connectivity.
            (default :obj:`True`)
        bias (bool, optional): If :obj:`True`, add a trainable additive bias.
            (default: :obj:`True`)
        activation (str, optional): Activation function to be used, :obj:`None`
            for identity function (i.e., no activation).
            (default: :obj:`None`)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        root_weight: bool = True,
        add_backward: bool = True,
        bias: bool = True,
        activation: str = None,
        dropout: float = 0.0,
    ):
        super(DiffConv, self).__init__(aggr="add", node_dim=-2)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k

        self.root_weight = root_weight
        self.add_backward = add_backward
        self.activation = get_functional_activation(activation)
        self.dropout = dropout

        n_filters = k
        if add_backward:
            n_filters *= 2
        if root_weight:
            n_filters += 1

        self.filters = nn.Linear(in_channels * n_filters, out_channels, bias=bias)
        if self.dropout > 0:
            self.filters = nn.Sequential(self.filters, nn.Dropout(self.dropout))

        self._support = None
        self.reset_parameters()

    @staticmethod
    def compute_support_index(
        edge_index: Adj,
        edge_weight: OptTensor = None,
        num_nodes: int = None,
        add_backward: bool = True,
    ) -> List:
        """Normalize the connectivity weights and (optionally) add normalized
        backward weights."""
        norm_edge_index, norm_edge_weight = asymmetric_norm(
            edge_index, edge_weight, dim=1, num_nodes=num_nodes
        )
        # Add backward matrices
        if add_backward:
            return [
                (norm_edge_index, norm_edge_weight)
            ] + DiffConv.compute_support_index(
                transpose(edge_index),
                edge_weight=edge_weight,
                num_nodes=num_nodes,
                add_backward=False,
            )
        # Normalize
        return [(norm_edge_index, norm_edge_weight)]

    def reset_parameters(self):
        if self.dropout > 0:
            self.filters[0].reset_parameters()
        else:
            self.filters.reset_parameters()
        self._support = None

    def message(self, x_j: Tensor, weight: Tensor) -> Tensor:
        """"""
        # x_j: [batch, edges, channels]
        return weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        """"""
        # adj_t: SparseTensor [nodes, nodes]
        # x: [(batch,) nodes, channels]
        return matmul(adj_t, x, reduce=self.aggr)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        cache_support: bool = False,
    ) -> Tensor:
        """"""
        # x: [batch, (steps), nodes, nodes]
        n = x.size(-2)
        if self._support is None:
            support = self.compute_support_index(
                edge_index, edge_weight, add_backward=self.add_backward, num_nodes=n
            )
            if cache_support:
                self._support = support
        else:
            support = self._support

        out = []
        if self.root_weight:
            out += [x]

        for sup_index, sup_weights in support:
            x_sup = x
            for _ in range(self.k):
                x_sup = self.propagate(sup_index, x=x_sup, weight=sup_weights)
                out += [x_sup]

        out = torch.cat(out, -1)
        return self.activation(self.filters(out))
