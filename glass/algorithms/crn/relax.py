"""Relaxation under the Keating potential.

Two flavours, both used by the BM driver:

  * `relax_full` — all atoms move under the full force field.
  * `relax_local` — only atoms inside `atom_set` (the 3rd neighbour
    shell of a recent bond swap) move. Forces are still computed
    globally for correctness; the speedup comes from the small basin
    of motion.

Both share a FIRE integrator. `relax_with_threshold` wraps either of
them with the BM "early-reject" check: if at any step the predicted
final energy exceeds a Metropolis-derived threshold, abort and return
``accepted=False`` so the caller can revert the swap without paying
for full convergence.

NOTE: This module uses the original Numba-jitted Keating implementation
from pywww for performance. The PyTorch version in glass.potentials.keating
is available for gradient-based applications but is too slow for the
intensive FIRE relaxation loops in CRN generation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Import the fast Numba-based Keating implementation
# This is orders of magnitude faster than the PyTorch version for
# the thousands of energy/force evaluations in FIRE relaxation
from glass.algorithms.crn._keating_numba import ALPHA, BETA, keating_forces

from glass.algorithms.crn.network import Network


# Standard FIRE parameters (Bitzek et al. 2006).
_F_INC = 1.1
_F_DEC = 0.5
_ALPHA_FIRE = 0.1
_F_ALPHA = 0.99
_N_MIN = 5
_DT_MAX_FACTOR = 10.0


def _fire_step(positions, velocities, forces, dt, alpha_fire):
    # Mixing: v ← (1-α) v + α |v| F̂
    fnorm = np.linalg.norm(forces)
    if fnorm > 0:
        vnorm = np.linalg.norm(velocities)
        velocities[:] = (1.0 - alpha_fire) * velocities + alpha_fire * forces * (
            vnorm / fnorm
        )
    velocities += dt * forces
    positions += dt * velocities
    return positions, velocities


def relax_fire(
    net: Network,
    *,
    atom_mask: Optional[np.ndarray] = None,
    max_iter: int = 200,
    ftol: float = 1e-3,
    dt_init: float = 0.05,
    alpha: float = ALPHA,
    beta: float = BETA,
    d: Optional[float] = None,
    early_reject_threshold: Optional[float] = None,
    early_reject_cf: float = 0.5,
    skip_first_n_for_threshold: int = 5,
) -> tuple[float, bool, int]:
    """FIRE relaxation; mutates ``net.positions``.

    Parameters
    ----------
    atom_mask : optional (N,) bool array. If given, only those atoms
        are advanced; forces on others are zeroed for the integrator.
    early_reject_threshold : if set, abort and return
        ``accepted=False`` as soon as ``E − cf · |F|² > threshold``.
        Skips the first ``skip_first_n_for_threshold`` steps to allow
        anharmonic settling (BM section II.2).

    Returns
    -------
    energy, accepted, n_steps
    """
    n = net.n_atoms
    velocities = np.zeros((n, 3), dtype=np.float64)
    dt = dt_init
    dt_max = dt_init * _DT_MAX_FACTOR
    alpha_fire = _ALPHA_FIRE
    n_pos = 0

    d_use = net.d if d is None else d
    energy = float("nan")
    for it in range(max_iter):
        # Compute energy and forces using fast Numba Keating
        energy, forces = keating_forces(net, alpha=alpha, beta=beta, d=d_use)

        if atom_mask is not None:
            forces[~atom_mask] = 0.0
        f2 = float(np.sum(forces * forces))
        fmax = float(np.max(np.abs(forces))) if forces.size else 0.0

        if early_reject_threshold is not None and it >= skip_first_n_for_threshold:
            if energy - early_reject_cf * f2 > early_reject_threshold:
                return energy, False, it

        if fmax < ftol:
            return energy, True, it

        # FIRE control
        power = float(np.sum(forces * velocities))
        if power > 0:
            n_pos += 1
            if n_pos > _N_MIN:
                dt = min(dt * _F_INC, dt_max)
                alpha_fire *= _F_ALPHA
        else:
            n_pos = 0
            dt *= _F_DEC
            velocities[:] = 0.0
            alpha_fire = _ALPHA_FIRE

        _fire_step(net.positions, velocities, forces, dt, alpha_fire)

    return energy, True, max_iter


def relax_full(net: Network, max_iter: int = 200, ftol: float = 1e-3) -> float:
    e, _, _ = relax_fire(net, max_iter=max_iter, ftol=ftol)
    return e


def relax_local(
    net: Network, atom_set: set[int], max_iter: int = 10, ftol: float = 1e-3
) -> float:
    mask = np.zeros(net.n_atoms, dtype=bool)
    for i in atom_set:
        mask[i] = True
    e, _, _ = relax_fire(net, atom_mask=mask, max_iter=max_iter, ftol=ftol)
    return e


def relax_with_threshold(
    net: Network,
    *,
    atom_set: Optional[set[int]] = None,
    e_threshold: float,
    local_steps: int = 10,
    full_max_iter: int = 200,
    ftol: float = 1e-3,
    cf: float = 0.5,
) -> tuple[float, bool]:
    """BM-style early-reject relaxation.

    First ``local_steps`` iterations are confined to ``atom_set`` (3rd
    neighbour shell of the swap). Then full relaxation continues with
    the same threshold check. Returns ``(energy, accepted)``.
    """
    if atom_set is not None:
        mask = np.zeros(net.n_atoms, dtype=bool)
        for i in atom_set:
            mask[i] = True
        e, ok, _ = relax_fire(
            net,
            atom_mask=mask,
            max_iter=local_steps,
            ftol=ftol,
            early_reject_threshold=e_threshold,
            early_reject_cf=cf,
        )
        if not ok:
            return e, False

    e, ok, _ = relax_fire(
        net,
        atom_mask=None,
        max_iter=full_max_iter,
        ftol=ftol,
        early_reject_threshold=e_threshold,
        early_reject_cf=cf,
    )
    return e, ok
