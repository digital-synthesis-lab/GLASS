"""Keating potential — bond-stretching + bond-bending energy and forces.

Reference: P. N. Keating, Phys. Rev. 145, 637 (1966); used by
Wooten-Winer-Weaire (1985) and Barkema-Mousseau (2000).

E = (3/16)(α/d²) Σ_{<ij>}      (r_ij · r_ij − d²)²
  + (3/8) (β/d²) Σ_{<jik>}     (r_ij · r_ik + d²/3)²

with default constants α=2.965 eV/Å², β=0.285·α, d=2.35 Å. PBC is
orthorhombic via minimum-image displacement.
"""

from __future__ import annotations

import numpy as np
from numba import njit


# Default constants from Barkema-Mousseau eq. (2)
ALPHA = 2.965     # eV/Å²  bond-stretch
BETA = 0.285 * ALPHA  # eV/Å²  bond-bend
D0 = 2.35         # Å       diamond Si bond length


@njit(cache=True, fastmath=True)
def _mic_disp(rj, ri, box):
    """Minimum-image displacement r_j − r_i in an orthorhombic cell."""
    dx = rj[0] - ri[0]
    dy = rj[1] - ri[1]
    dz = rj[2] - ri[2]
    dx -= box[0] * np.round(dx / box[0])
    dy -= box[1] * np.round(dy / box[1])
    dz -= box[2] * np.round(dz / box[2])
    return dx, dy, dz


@njit(cache=True, fastmath=True)
def _energy_jit(positions, box, bonds, neigh, degree, alpha, beta, d):
    n = positions.shape[0]
    inv_d2 = 1.0 / (d * d)
    cs = (3.0 / 16.0) * alpha * inv_d2  # coefficient on bond term
    cb = (3.0 / 8.0) * beta * inv_d2    # coefficient on angle term
    third_d2 = (d * d) / 3.0

    e_bond = 0.0
    for b in range(bonds.shape[0]):
        i = bonds[b, 0]
        j = bonds[b, 1]
        dx, dy, dz = _mic_disp(positions[j], positions[i], box)
        r2 = dx * dx + dy * dy + dz * dz
        diff = r2 - d * d
        e_bond += cs * diff * diff

    e_ang = 0.0
    for i in range(n):
        di = degree[i]
        for a in range(di):
            j = neigh[i, a]
            for b in range(a + 1, di):
                k = neigh[i, b]
                jx, jy, jz = _mic_disp(positions[j], positions[i], box)
                kx, ky, kz = _mic_disp(positions[k], positions[i], box)
                dot = jx * kx + jy * ky + jz * kz
                s = dot + third_d2
                e_ang += cb * s * s

    return e_bond + e_ang


@njit(cache=True, fastmath=True)
def _forces_jit(positions, box, bonds, neigh, degree, alpha, beta, d):
    n = positions.shape[0]
    inv_d2 = 1.0 / (d * d)
    cs = (3.0 / 16.0) * alpha * inv_d2
    cb = (3.0 / 8.0) * beta * inv_d2
    third_d2 = (d * d) / 3.0

    forces = np.zeros((n, 3), dtype=np.float64)
    e_bond = 0.0
    for b in range(bonds.shape[0]):
        i = bonds[b, 0]
        j = bonds[b, 1]
        dx, dy, dz = _mic_disp(positions[j], positions[i], box)
        r2 = dx * dx + dy * dy + dz * dz
        diff = r2 - d * d
        e_bond += cs * diff * diff
        # dE/dr_j = 4 cs diff r_ij ; dE/dr_i = -4 cs diff r_ij
        coef = 4.0 * cs * diff
        forces[i, 0] += coef * dx
        forces[i, 1] += coef * dy
        forces[i, 2] += coef * dz
        forces[j, 0] -= coef * dx
        forces[j, 1] -= coef * dy
        forces[j, 2] -= coef * dz

    e_ang = 0.0
    for i in range(n):
        di = degree[i]
        for a in range(di):
            j = neigh[i, a]
            for b in range(a + 1, di):
                k = neigh[i, b]
                jx, jy, jz = _mic_disp(positions[j], positions[i], box)
                kx, ky, kz = _mic_disp(positions[k], positions[i], box)
                dot = jx * kx + jy * ky + jz * kz
                s = dot + third_d2
                e_ang += cb * s * s
                # dE/dr_j = 2 cb s r_ik ; dE/dr_k = 2 cb s r_ij
                # dE/dr_i = -2 cb s (r_ij + r_ik)
                coef = 2.0 * cb * s
                fjx = coef * kx
                fjy = coef * ky
                fjz = coef * kz
                fkx = coef * jx
                fky = coef * jy
                fkz = coef * jz
                forces[j, 0] -= fjx
                forces[j, 1] -= fjy
                forces[j, 2] -= fjz
                forces[k, 0] -= fkx
                forces[k, 1] -= fky
                forces[k, 2] -= fkz
                forces[i, 0] += fjx + fkx
                forces[i, 1] += fjy + fky
                forces[i, 2] += fjz + fkz

    return e_bond + e_ang, forces


def _box_diag(cell: np.ndarray) -> np.ndarray:
    if not (np.allclose(cell - np.diag(np.diag(cell)), 0.0)):
        raise ValueError("Only orthorhombic cells supported in v1")
    return np.ascontiguousarray(np.diag(cell), dtype=np.float64)


def keating_energy(net, alpha: float = ALPHA, beta: float = BETA,
                   d: float | None = None) -> float:
    box = _box_diag(net.cell)
    d_use = net.d if d is None else d
    return float(_energy_jit(net.positions, box, net.bonds, net.neigh, net.degree,
                             alpha, beta, d_use))


def keating_forces(net, alpha: float = ALPHA, beta: float = BETA,
                   d: float | None = None):
    box = _box_diag(net.cell)
    d_use = net.d if d is None else d
    e, f = _forces_jit(net.positions, box, net.bonds, net.neigh, net.degree,
                       alpha, beta, d_use)
    return float(e), f
