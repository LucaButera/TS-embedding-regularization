import math

import torch
from einops import rearrange
from torch import Tensor
from torch_geometric.utils import subgraph
from tsl.nn import expand_then_cat


def maybe_cat_emb(x: Tensor, emb: Tensor | None, dim: int = -1) -> Tensor:
    if emb is not None:
        if emb.ndim < x.ndim:
            if emb.ndim == 3 and x.ndim == 4:
                emb = rearrange(emb, "b n f -> b 1 n f")
            elif emb.ndim == 2 and x.ndim == 3:
                emb = rearrange(emb, "n f -> 1 n f")
            elif emb.ndim == 2 and x.ndim == 4:
                emb = rearrange(emb, "n f -> 1 1 n f")
            else:
                raise ValueError(f"Invalid dimensions: x={x.shape} and emb={emb.shape}")
        x = expand_then_cat([x, emb], dim=dim)
    return x


def sample_visible_subgraph(batch, p: float):
    # if elements are batched we must remove only nodes visible in all batches
    visible_mask = torch.all(rearrange(batch.mask, "b t n f -> n (b t f)"), dim=-1)
    kept_mask = torch.rand(visible_mask.size(), device=visible_mask.device) > p
    batch.edge_index, batch.edge_attr = subgraph(
        subset=kept_mask,
        edge_index=batch.edge_index,
        edge_attr=batch.edge_attr,
        num_nodes=batch.num_nodes,
        relabel_nodes=True,
    )
    for key, value in batch:
        if key in ["edge_index", "edge_attr", "transform"]:
            continue
        elif key in ["batch", "ptr"]:
            batch[key] = value[visible_mask]
        else:
            batch[key] = value[..., kept_mask, :]
            if hasattr(batch, "transform") and key in batch.transform:
                batch.transform[key] = batch.transform[key].slice(node_index=kept_mask)
    return batch


def partially_reset_linear(
    layer: torch.nn.Linear,
    in_mask: Tensor | None = None,
    out_mask: Tensor | None = None,
):
    if in_mask is None and out_mask is None:
        return
    if in_mask is None:
        in_mask = torch.zeros(layer.weight.size(1), dtype=torch.bool)
    if out_mask is None:
        out_mask = torch.zeros(layer.weight.size(0), dtype=torch.bool)
    # reset full rows and columns if any of the masks requires it
    reset_mask = out_mask.unsqueeze(1) | in_mask.unsqueeze(0)
    with torch.no_grad():
        reset_weight = torch.zeros_like(layer.weight)
        torch.nn.init.kaiming_uniform_(reset_weight, a=math.sqrt(5))
        layer.weight.data[reset_mask] = reset_weight[reset_mask]
        if layer.bias is not None and out_mask.any():
            reset_bias = torch.zeros_like(layer.bias)
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(layer.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(reset_bias, -bound, bound)
            layer.bias.data[out_mask] = reset_bias[out_mask]
