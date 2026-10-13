"""Random initial network — Barkema-Mousseau §A loop expansion.

Atoms are placed first by Poisson-disk packing (using glass.utils.packing.pack),
then a tetravalent bond list is built by growing a closed loop that
eventually visits every atom exactly twice (so every atom has degree
4). The resulting network has zero crystalline memory by construction
but is heavily strained — the WWW driver relaxes it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from glass.algorithms.crn.network import (
    MAX_DEGREE,
    Network,
    bond_length_for_density,
    cubic_cell_for_si,
)
from glass.utils.packing import pack as glass_pack


# ---------------------------------------------------------------------- #
# Atom placement
# ---------------------------------------------------------------------- #


def _pack_atoms(
    n_atoms: int, cell: np.ndarray, min_distance: float, rng: np.random.Generator
) -> np.ndarray:
    """Use glass.utils.packing.pack for Poisson-disk + MC soft-core packing."""
    return glass_pack(
        n_atoms=n_atoms, cell=cell, min_distance=min_distance, rng=rng, verbose=False
    )


# ---------------------------------------------------------------------- #
# Loop expansion
# ---------------------------------------------------------------------- #


def grow_4coord_loop(
    positions: np.ndarray,
    cell: np.ndarray,
    rng: np.random.Generator,
    *,
    r_c_init: float = 3.0,
    r_c_max: float = 6.0,
    r_c_step: float = 0.1,
) -> np.ndarray:
    """Build a 4-regular bond list by BM-style loop expansion.

    Returns ``bonds`` array of shape (2N, 2). Raises if it cannot reach
    full 4-coordination by ``r_c_max``.
    """
    n = positions.shape[0]
    box = np.diag(cell)
    degree = np.zeros(n, dtype=np.int64)
    neigh = np.full((n, MAX_DEGREE), -1, dtype=np.int64)
    bonds: list[tuple[int, int]] = []

    def has_bond(i: int, j: int) -> bool:
        for k in range(degree[i]):
            if neigh[i, k] == j:
                return True
        return False

    def add_bond(i: int, j: int) -> None:
        neigh[i, degree[i]] = j
        neigh[j, degree[j]] = i
        degree[i] += 1
        degree[j] += 1
        bonds.append((i, j))

    def remove_bond(i: int, j: int) -> None:
        for k in range(degree[i]):
            if neigh[i, k] == j:
                neigh[i, k] = neigh[i, degree[i] - 1]
                neigh[i, degree[i] - 1] = -1
                degree[i] -= 1
                break
        for k in range(degree[j]):
            if neigh[j, k] == i:
                neigh[j, k] = neigh[j, degree[j] - 1]
                neigh[j, degree[j] - 1] = -1
                degree[j] -= 1
                break
        for idx, (a, b) in enumerate(bonds):
            if {a, b} == {i, j}:
                bonds.pop(idx)
                return

    # Pre-compute MIC squared distances (N is small; 216² is trivial).
    diffs = positions[:, None, :] - positions[None, :, :]
    diffs -= box * np.round(diffs / box)
    dist2 = np.sum(diffs * diffs, axis=-1)
    np.fill_diagonal(dist2, np.inf)

    # ---- Seed: a 4-cycle of close neighbours ------------------------- #
    r_c = r_c_init
    seed_ok = False
    while not seed_ok and r_c <= r_c_max:
        rc2 = r_c * r_c
        # Random A; pick B,C among A's r_c-neighbours that are mutual r_c-neighbours
        order = rng.permutation(n)
        for a in order:
            cand = np.where(dist2[a] <= rc2)[0]
            if cand.size < 3:
                continue
            for b in rng.permutation(cand):
                for c in rng.permutation(cand):
                    if c == b or c == a:
                        continue
                    if dist2[b, c] > rc2:
                        continue
                    # Need a 4th atom D bonded to both A and C (loop A-B-C-D-A)
                    cand_d = np.where((dist2[a] <= rc2) & (dist2[c] <= rc2))[0]
                    cand_d = cand_d[(cand_d != a) & (cand_d != b) & (cand_d != c)]
                    if cand_d.size == 0:
                        continue
                    d = int(rng.choice(cand_d))
                    add_bond(int(a), int(b))
                    add_bond(int(b), int(c))
                    add_bond(int(c), int(d))
                    add_bond(int(d), int(a))
                    seed_ok = True
                    break
                if seed_ok:
                    break
            if seed_ok:
                break
        if not seed_ok:
            r_c += r_c_step

    if not seed_ok:
        raise RuntimeError(f"could not seed initial 4-loop within r_c={r_c_max} Å")

    # ---- Expansion: replace bond (b,c) with (a,b)+(a,c) ------------- #
    target_bonds = 2 * n
    while len(bonds) < target_bonds:
        rc2 = r_c * r_c
        progressed = True
        while progressed and len(bonds) < target_bonds:
            progressed = False
            # Iterate atoms with degree < 4 in random order.
            under = np.where(degree < MAX_DEGREE)[0]
            rng.shuffle(under)
            for a in under:
                if degree[a] >= MAX_DEGREE:
                    continue
                # candidates = atoms within r_c of a, not bonded to a
                cand = np.where(dist2[a] <= rc2)[0]
                if cand.size == 0:
                    continue
                # Filter unbonded
                mask = np.array(
                    [not has_bond(int(a), int(x)) for x in cand], dtype=bool
                )
                cand = cand[mask]
                if cand.size < 2:
                    continue
                rng.shuffle(cand)
                placed = False
                for b in cand:
                    if degree[b] == 0:
                        continue
                    # Try to find c ∈ cand with bond (b,c) existing
                    for k in range(degree[b]):
                        c = int(neigh[b, k])
                        if c == a:
                            continue
                        if dist2[a, c] > rc2:
                            continue
                        if has_bond(int(a), c):
                            continue
                        # Valid: replace (b,c) with (a,b)+(a,c)
                        remove_bond(int(b), c)
                        add_bond(int(a), int(b))
                        add_bond(int(a), c)
                        progressed = True
                        placed = True
                        break
                    if placed:
                        break
        if len(bonds) < target_bonds:
            if r_c >= r_c_max:
                break
            r_c = min(r_c + r_c_step, r_c_max)

    if len(bonds) != target_bonds:
        deg_hist = np.bincount(degree, minlength=MAX_DEGREE + 1)
        raise RuntimeError(
            f"loop expansion reached only {len(bonds)}/{target_bonds} bonds "
            f"at r_c={r_c} Å; degree histogram={deg_hist.tolist()}"
        )

    return np.asarray(bonds, dtype=np.int64)


# ---------------------------------------------------------------------- #
# High-level entry
# ---------------------------------------------------------------------- #


def random_initial_network(
    n_atoms: int,
    *,
    density: float = 2.33,
    seed: int = 0,
    min_distance: Optional[float] = None,
    d: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Network:
    """Return a random 4-coordinated `Network` per Barkema-Mousseau §A.

    The bond length ``d`` defaults to the isotropic-scaling value for
    ``density``. Packer min-distance and loop-expansion radii scale with
    ``d`` so this works at densities away from crystalline Si.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if d is None:
        d = bond_length_for_density(density)
    if min_distance is None:
        # Use ~98% of d so the packer doesn't choke at high density.
        min_distance = 0.98 * d
    cell = cubic_cell_for_si(n_atoms, density)
    positions = _pack_atoms(n_atoms, cell, min_distance, rng)
    # Loop-expansion radii scale with d (default tuned for d=2.35 Å).
    r_c_init = 3.0 * (d / 2.35)
    r_c_max = 6.0 * (d / 2.35)
    bonds = grow_4coord_loop(
        positions, cell, rng, r_c_init=r_c_init, r_c_max=r_c_max
    )
    return Network.from_bonds(positions, cell, bonds, d=d)
