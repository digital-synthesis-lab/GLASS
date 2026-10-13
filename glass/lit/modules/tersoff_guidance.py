"""Tersoff-energy guidance for the reverse SDE.

Produces a per-atom "force-like" score vector (-grad E / N_atoms) that can be
added to the diffusion-model score to bias sampling toward configurations with
lower Tersoff potential energy. Analogous in spirit to EPSR (Empirical
Potential Structure Refinement), but embedded directly inside the reverse SDE.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from glass.potentials.tersoff import TersoffParameters, TorchTersoff


def _default_si_parameters() -> dict:
    return {
        ("Si", "Si", "Si"): TersoffParameters(
            A=3264.7,
            B=95.373,
            lambda1=3.2394,
            lambda2=1.3258,
            lambda3=1.3258,
            beta=0.33675,
            gamma=1.00,
            m=3.00,
            n=22.956,
            c=4.8381,
            d=2.0417,
            h=0.0000,
            R=3.00,
            D=0.20,
        )
    }


class TersoffEnergyGuidance(nn.Module):
    """Wrap TorchTersoff as a per-atom guidance signal for the reverse SDE.

    The returned tensor is ``-grad_pos E(pos) / N_atoms`` (i.e. the Tersoff
    force divided by the atom count). It points downhill in energy, so it is
    safe to ADD to the score when running the reverse SDE.

    Args:
        cutoff: Accepted for symmetry with other guidance modules. The
            Tersoff neighbour cutoff is determined internally by the
            parameter set (``R + D``); this value is not used to override it.
        clamp_norm: Per-atom gradient-norm clamp. Any atom whose guidance
            vector has magnitude larger than ``clamp_norm`` is rescaled so
            that its magnitude equals ``clamp_norm``. Prevents blow-ups at
            high noise levels where atoms can be arbitrarily close.
        parameters: Tersoff parameter dict keyed by ``(species, species,
            species)`` tuples. Defaults to the single-species Si tutorial
            parameters.
        dtype: Working dtype for the Tersoff evaluation.
    """

    def __init__(
        self,
        cutoff: float = 3.77,
        clamp_norm: float = 10.0,
        parameters: dict | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.clamp_norm = clamp_norm
        self.dtype = dtype
        self.potential = TorchTersoff(
            parameters if parameters is not None else _default_si_parameters(),
            dtype=dtype,
        )

    @staticmethod
    def _check_single_species(species: Tensor) -> None:
        if species.dim() == 1:
            unique = torch.unique(species)
            if unique.numel() != 1:
                raise ValueError(
                    "TersoffEnergyGuidance only supports a single atomic "
                    f"species; got {unique.numel()} distinct species."
                )
        elif species.dim() == 2:
            idx = species.argmax(dim=-1)
            unique = torch.unique(idx)
            if unique.numel() != 1:
                raise ValueError(
                    "TersoffEnergyGuidance only supports a single atomic "
                    f"species; got {unique.numel()} distinct species."
                )
        else:
            raise ValueError(
                f"Unexpected species tensor rank {species.dim()} "
                "(expected 1 or 2)."
            )

    def _clamp_per_atom_norm(self, vec: Tensor) -> Tensor:
        if self.clamp_norm is None or self.clamp_norm <= 0:
            return vec
        norms = vec.norm(dim=-1, keepdim=True)
        scale = torch.clamp(self.clamp_norm / (norms + 1e-12), max=1.0)
        return vec * scale

    def forward(
        self,
        pos: Tensor,
        cell: Tensor,
        species: Tensor,
    ) -> Tensor:
        """Compute ``-grad E / N_atoms`` for ``pos``.

        Args:
            pos: (N, 3) Cartesian positions in Angstrom.
            cell: (3, 3) periodic cell in Angstrom.
            species: (N,) integer tensor or (N, S) one-hot tensor identifying
                atomic species. Only homogeneous (single-species) inputs are
                accepted.

        Returns:
            (N, 3) guidance tensor (float, same device as ``pos``).
        """
        self._check_single_species(species)

        device = pos.device
        leaf = pos.detach().to(self.dtype).clone().requires_grad_(True)
        cell_t = cell.detach().to(self.dtype)

        with torch.enable_grad():
            energy = self.potential.energy(leaf, cell_t, pbc=(True, True, True))
            (grad,) = torch.autograd.grad(energy, leaf)

        n_atoms = pos.shape[0]
        guidance = (-grad / max(n_atoms, 1)).to(device=device, dtype=pos.dtype)
        # Sanitise before clamping in case the potential produced NaN/Inf.
        guidance = torch.nan_to_num(guidance, nan=0.0, posinf=0.0, neginf=0.0)
        guidance = self._clamp_per_atom_norm(guidance)
        return guidance.detach()


class TersoffSchedule:
    """Time-dependent weight ``lambda(t)`` for the Tersoff guidance term.

    Supports three shapes:

    - ``constant``: ``lambda_0`` regardless of ``t``.
    - ``linear``: ``lambda_0 * max(0, 1 - t / tmax)`` — zero at high noise,
      full weight as the trajectory approaches clean data.
    - ``sigmoid``: ``lambda_0 * sigmoid(-k * (t - t_gate))`` — activates
      smoothly below a noise-level threshold.
    """

    def __init__(
        self,
        schedule: str = "linear",
        lambda_0: float = 0.05,
        tmax: float = 1.0,
        t_gate: float = 0.3,
        k: float = 20.0,
    ) -> None:
        if schedule not in ("constant", "linear", "sigmoid"):
            raise ValueError(
                f"Unknown schedule '{schedule}'. "
                "Expected one of: constant, linear, sigmoid."
            )
        self.schedule = schedule
        self.lambda_0 = float(lambda_0)
        self.tmax = float(tmax)
        self.t_gate = float(t_gate)
        self.k = float(k)

    def __call__(self, t) -> float:
        if isinstance(t, Tensor):
            t_val = float(t.detach().flatten()[0].item())
        else:
            t_val = float(t)

        if self.schedule == "constant":
            return self.lambda_0
        if self.schedule == "linear":
            ramp = 1.0 - t_val / max(self.tmax, 1e-12)
            return self.lambda_0 * max(0.0, min(1.0, ramp))
        # sigmoid
        return self.lambda_0 * _sigmoid(-self.k * (t_val - self.t_gate))


def _sigmoid(x: float) -> float:
    # Plain-Python sigmoid; avoids tensor allocations for the scalar case.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
