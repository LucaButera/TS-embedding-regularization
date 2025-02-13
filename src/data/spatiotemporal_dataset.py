from tsl.data.spatiotemporal_dataset import (
    SpatioTemporalDataset as _SpatioTemporalDataset_,
)
from tsl.ops.connectivity import reduce_graph
from tsl.typing import IndexSlice


class SpatioTemporalDataset(_SpatioTemporalDataset_):

    def reduce_(
        self, time_index: IndexSlice | None = None, node_index: IndexSlice | None = None
    ):
        # use slice to reduce known tensor
        time_slice = self._get_time_index(time_index, layout="slice")
        node_slice = self._get_node_index(node_index, layout="slice")
        # use index to reduce using index-fed functions
        time_index = self._get_time_index(time_index, layout="index")
        node_index = self._get_node_index(node_index, layout="index")
        try:
            if self.edge_index is not None and node_index is not None:
                self.edge_index, edge_mask = reduce_graph(
                    subset=node_index,
                    edge_index=self.edge_index,
                    num_nodes=self.n_nodes,
                )
                if self.edge_weight is not None:
                    self.edge_weight = self.edge_weight[edge_mask]
            self.target = self.target[time_slice, node_slice]
            if self.index is not None and time_index is not None:
                self.index = self.index[time_index.numpy()]
            if self.mask is not None:
                self.mask = self.mask[time_slice, node_slice]
            if self.trend is not None:
                self.trend = self.trend[time_slice, node_slice]
            for name, attr in self._covariates.items():
                x, scaler = self.get_tensor(
                    name, time_index=time_index, node_index=node_index
                )
                attr["value"] = x
                if scaler is not None:
                    self.scalers[name] = scaler
        except Exception as e:
            raise e
        return self
