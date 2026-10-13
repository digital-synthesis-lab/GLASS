"""Simulated-annealing post-relaxation using the Tersoff potential.

Runs a short MD-like Cartesian trajectory that mixes the Tersoff gradient
(already normalised by ``N_atoms`` and clamped inside
``TersoffEnergyGuidance``) with a decaying thermal noise. The temperature
schedule is geometric:

    T_k = T0 * (T_end / T0) ** (k / (n_steps - 1)),   k = 0..n_steps-1

Only the Tersoff potential is used — the score net is not involved. This is
the "anneal tail" that runs AFTER the reverse SDE has produced a near-physical
configuration.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from torch import Tensor


def _per_atom_norm_clamp(vec: Tensor, max_norm: float) -> Tensor:
    if max_norm is None or max_norm <= 0:
        return vec
    norms = vec.norm(dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / (norms + 1e-12), max=1.0)
    return vec * scale


def _wrap_pbc(pos: Tensor, cell: Tensor) -> Tensor:
    """Wrap Cartesian positions back into the primitive cell.

    Assumes ``cell`` rows are the lattice vectors.
    """
    cell_inv = torch.linalg.inv(cell.to(pos.dtype))
    frac = pos @ cell_inv.T
    frac = frac - torch.floor(frac)
    return frac @ cell.to(pos.dtype).T


def simulated_anneal(
    pos: Tensor,
    cell: Tensor,
    species: Tensor,
    tersoff_guidance,
    n_steps: int = 100,
    T0: float = 1e-2,
    T_end: float = 1e-5,
    lr: float = 1e-3,
    lr_clamp: float = 0.2,
    wrap: bool = True,
) -> Tensor:
    """Run ``n_steps`` of simulated annealing on ``pos``.

    Args:
        pos: (N, 3) Cartesian positions (Å). Detached from any graph.
        cell: (3, 3) cell tensor (Å).
        species: (N,) or (N, S) species tensor.
        tersoff_guidance: ``TersoffEnergyGuidance`` instance (returns
            ``-grad E / N_atoms``, already clamped).
        n_steps: Number of annealing steps.
        T0: Initial temperature (variance of the noise term).
        T_end: Final temperature. Must be > 0 for the geometric decay;
            a very small value (e.g. 1e-8) approximates T=0.
        lr: Step size on the force term.
        lr_clamp: Hard cap on per-atom displacement per step (Å).
        wrap: If True, wrap positions back into the cell each step.

    Returns:
        (N, 3) final positions tensor (same device/dtype as ``pos``).
    """
    if n_steps <= 0:
        return pos.detach()
    if T0 <= 0 or T_end <= 0:
        raise ValueError("T0 and T_end must be positive.")

    pos = pos.detach().clone()

    log_ratio = math.log(T_end / T0)
    denom = max(n_steps - 1, 1)
    for k in range(n_steps):
        T_k = T0 * math.exp(log_ratio * k / denom)
        force = tersoff_guidance(pos, cell, species)
        noise_scale = math.sqrt(2.0 * T_k)
        delta = lr * force + noise_scale * torch.randn_like(pos)
        delta = _per_atom_norm_clamp(delta, lr_clamp)
        pos = pos + delta
        if wrap:
            pos = _wrap_pbc(pos, cell)

    return pos.detach()


def make_anneal_fn(
    tersoff_guidance,
    n_steps: int = 100,
    T0: float = 1e-2,
    T_end: float = 1e-5,
    lr: float = 1e-3,
    lr_clamp: float = 0.2,
    wrap: bool = True,
) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    """Return an ``anneal_fn(pos, cell, species) -> pos`` closure."""

    def _fn(pos: Tensor, cell: Tensor, species: Tensor) -> Tensor:
        return simulated_anneal(
            pos=pos,
            cell=cell,
            species=species,
            tersoff_guidance=tersoff_guidance,
            n_steps=n_steps,
            T0=T0,
            T_end=T_end,
            lr=lr,
            lr_clamp=lr_clamp,
            wrap=wrap,
        )

    return _fn
