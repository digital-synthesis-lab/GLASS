"""MeshGraphNets conv + Processor/Decoder.

Ported from graphite (LLNL) — https://github.com/LLNL/graphite —
specifically ``graphite/src/graphite/nn/convs/mgn.py`` and
``graphite/src/graphite/nn/models/mgn.py``. Only the pieces referenced
by glass (``Processor``, ``Decoder``, and the conv they depend on) are
kept here; glass defines its own encoders in ``lit/modules/``.
"""

import copy
from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter

from .mlp import MLP


class EdgeProcessor(nn.Module):
    def __init__(self, dims: List[int]) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(MLP(dims, act=nn.SiLU()), nn.LayerNorm(dims[-1]))

    def forward(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        out = torch.cat([x_i, x_j, edge_attr], dim=-1)
        out = self.edge_mlp(out)
        return edge_attr + out


class NodeProcessor(nn.Module):
    def __init__(self, dims: List[int]) -> None:
        super().__init__()
        self.node_mlp = nn.Sequential(MLP(dims, act=nn.SiLU()), nn.LayerNorm(dims[-1]))

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        j = edge_index[1]
        out = scatter(edge_attr, index=j, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=-1)
        out = self.node_mlp(out)
        return x + out


class MeshGraphNetsConv(nn.Module):
    """Single MeshGraphNets message-passing block.

    Reference: https://arxiv.org/pdf/2010.03409v4.pdf
    """

    def __init__(self, node_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.edge_processor = EdgeProcessor([node_dim * 2 + edge_dim] + [edge_dim] * 3)
        self.node_processor = NodeProcessor([node_dim + edge_dim] + [node_dim] * 3)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> Tuple[Tensor, Tensor]:
        i = edge_index[0]
        j = edge_index[1]
        edge_attr = self.edge_processor(x[i], x[j], edge_attr)
        x = self.node_processor(x, edge_index, edge_attr)
        return x, edge_attr

    def extra_repr(self) -> str:
        return f"node_dim={self.node_dim}, edge_dim={self.edge_dim}"


class Processor(nn.Module):
    """Stack of MeshGraphNets conv blocks."""

    def __init__(self, num_convs: int, node_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.num_convs = num_convs
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.convs = nn.ModuleList(
            [copy.deepcopy(MeshGraphNetsConv(node_dim, edge_dim)) for _ in range(num_convs)]
        )

    def forward(
        self, h_node: Tensor, edge_index: Tensor, h_edge: Tensor
    ) -> Tuple[Tensor, Tensor]:
        for conv in self.convs:
            h_node, h_edge = conv(h_node, edge_index, h_edge)
        return h_node, h_edge


class Decoder(nn.Module):
    """MLP decoder head."""

    def __init__(self, node_dim: int, out_dim: int) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.out_dim = out_dim
        self.decoder = MLP([node_dim, node_dim, out_dim], act=nn.SiLU())

    def forward(self, h_node: Tensor) -> Tensor:
        return self.decoder(h_node)
