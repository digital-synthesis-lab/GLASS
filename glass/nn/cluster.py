from typing import Tuple

import torch
from torch import Tensor


def periodic_radius_graph(
    x: Tensor, r: float, cell: Tensor, loop: bool = False
) -> Tuple[Tensor, Tensor]:
    """Graph edges under a cutoff radius with periodic boundaries (O(N^2)).

    Args:
        x: Point-cloud coordinates with shape ``(N, D)``.
        r: Cutoff radius.
        cell: Periodic cell with shape ``(D, D)``.
        loop: Whether to include self-loops.

    Returns:
        edge_index: ``(2, E)`` directed edges.
        edge_vec:   ``(E, D)`` edge vectors.

    Notes:
        - Does not handle batched inputs.
        - Assumes 3D coordinates in practice.
        - Not accurate for very oblique cells.
    """
    inv_cell = torch.linalg.pinv(cell)

    vec = x[None, :, :] - x[:, None, :]
    vec = vec - torch.round(vec @ inv_cell) @ cell
    dist = torch.linalg.norm(vec, dim=-1)
    dist += torch.eye(x.size(0), device=x.device) * (1.0 - float(loop)) * (r + 1.0)
    edge_index = torch.nonzero(dist < r).T
    i, j = edge_index
    edge_vec = vec[i, j]
    return edge_index, edge_vec
