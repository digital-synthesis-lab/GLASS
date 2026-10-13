"""Hard-sphere / soft-core atomic packing for initial structures.

Two-stage strategy:

1. ``poisson_disk_pack`` — batched Bridson-style Poisson-disk sampling.
   Fast path (orthorhombic-diagonal cells) uses ``scipy.spatial.cKDTree``
   with ``boxsize`` for periodic neighbour queries. Triclinic cells fall
   through to a manual 3D bucket grid with explicit minimum-image
   distance checks.

2. ``mc_soft_pack`` — Metropolis Monte-Carlo anneal with a
   Weeks-Chandler-Andersen pair potential. Engaged as a fallback when
   the packing fraction is too high for Poisson-disk to converge.

``pack(...)`` is the top-level dispatcher: try Poisson-disk first; on
exhaustion, harvest the partial set, fill the rest uniformly, and anneal.
If the final minimum pairwise distance still falls below
``0.95 * min_distance`` it emits a ``UserWarning`` but returns the
structure (per user spec).

No torch, no packmol. Pure numpy + scipy.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Cell geometry helpers
# ---------------------------------------------------------------------------


def _is_orthorhombic_diagonal(cell: np.ndarray, atol: float = 1e-8) -> bool:
    """Return True when ``cell`` is diagonal with positive entries."""
    if cell.shape != (3, 3):
        return False
    off = cell - np.diag(np.diag(cell))
    if np.max(np.abs(off)) > atol:
        return False
    diag = np.diag(cell)
    return bool(np.all(diag > 0))


def min_image_distance_sq(
    diff: np.ndarray, cell: np.ndarray, cell_inv: np.ndarray
) -> np.ndarray:
    """Squared minimum-image distance for a batch of displacement vectors.

    Parameters
    ----------
    diff : (..., 3) array
        Cartesian displacements ``r_j - r_i``.
    cell : (3, 3) array
        Lattice vectors as rows.
    cell_inv : (3, 3) array
        Pre-computed inverse of ``cell``.

    Returns
    -------
    r2 : (...) array
        Squared minimum-image distances.
    """
    frac = diff @ cell_inv
    frac -= np.round(frac)
    disp = frac @ cell
    return np.einsum("...i,...i->...", disp, disp)


def min_pairwise_distance(positions: np.ndarray, cell: np.ndarray) -> float:
    """Return the minimum pairwise minimum-image distance in ``positions``."""
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    n = positions.shape[0]
    if n < 2:
        return float("inf")

    if _is_orthorhombic_diagonal(cell):
        diag = np.diag(cell)
        wrapped = np.mod(positions, diag)
        tree = cKDTree(wrapped, boxsize=diag)
        # Smallest non-zero distance across all pairs. query k=2 gives
        # each point's nearest neighbour.
        d, _ = tree.query(wrapped, k=2)
        return float(d[:, 1].min())

    # Triclinic: full broadcast MIC.
    cell_inv = np.linalg.inv(cell)
    diff = positions[:, None, :] - positions[None, :, :]
    r2 = min_image_distance_sq(diff, cell, cell_inv)
    # Exclude diagonal (self).
    idx = np.arange(n)
    r2[idx, idx] = np.inf
    return float(np.sqrt(r2.min()))


# ---------------------------------------------------------------------------
# Poisson-disk sampling (primary)
# ---------------------------------------------------------------------------


def _wrap_frac(frac: np.ndarray) -> np.ndarray:
    return frac - np.floor(frac)


def _poisson_ortho(
    n_atoms: int,
    diag: np.ndarray,
    min_distance: float,
    rng: np.random.Generator,
    max_passes: int,
) -> Tuple[np.ndarray, int]:
    """Orthorhombic-diagonal fast path using periodic cKDTree."""
    accepted = np.empty((0, 3), dtype=np.float64)
    for _ in range(max_passes):
        remaining = n_atoms - accepted.shape[0]
        if remaining <= 0:
            break
        batch = min(max(4 * remaining, 64), 4096)
        frac = rng.random((batch, 3))
        cand = frac * diag  # Cartesian in [0, a_i)

        # Reject candidates that collide with the already-accepted set.
        if accepted.shape[0] > 0:
            tree_acc = cKDTree(accepted, boxsize=diag)
            # query_ball_point returns list of index arrays — use k+len
            # approach via count_neighbors would also work; query with
            # k=1 gives nearest, sufficient for rejection.
            d, _ = tree_acc.query(cand, k=1)
            cand = cand[d >= min_distance]
        if cand.shape[0] == 0:
            continue

        # Greedy within-pass: process in random order, pick points whose
        # nearest predecessor in the pass is ≥ min_distance away.
        order = rng.permutation(cand.shape[0])
        cand_shuf = cand[order]
        keep = np.ones(cand_shuf.shape[0], dtype=bool)
        # Iterate: for each kept point, mark every later point within
        # min_distance as rejected. Use cKDTree for the within-pass
        # query but only among survivors discovered so far.
        # Faster: run query_pairs once, then drop the higher index in
        # each pair (which, by the shuffle above, is random).
        tree_pass = cKDTree(cand_shuf, boxsize=diag)
        pairs = tree_pass.query_pairs(r=min_distance, output_type="ndarray")
        if pairs.size > 0:
            # Remove the higher index of each pair.
            # Iterating is safe because we only ever set True->False.
            drop_idx = pairs[:, 1]
            keep[drop_idx] = False
        picked = cand_shuf[keep]

        # Defensive re-check of picked against already-accepted (in case
        # the pass dedup interacted badly with the accepted-set rejection
        # above — should be a no-op but cheap insurance).
        if accepted.shape[0] > 0 and picked.shape[0] > 0:
            tree_acc = cKDTree(accepted, boxsize=diag)
            d, _ = tree_acc.query(picked, k=1)
            picked = picked[d >= min_distance]

        if picked.shape[0] == 0:
            continue

        # Clip to how many we still need.
        if picked.shape[0] > remaining:
            picked = picked[:remaining]

        accepted = np.vstack([accepted, picked])

    return accepted, accepted.shape[0]


def _poisson_triclinic(
    n_atoms: int,
    cell: np.ndarray,
    min_distance: float,
    rng: np.random.Generator,
    max_passes: int,
) -> Tuple[np.ndarray, int]:
    """Triclinic-cell path using a fractional-coord bucket grid with MIC."""
    cell_inv = np.linalg.inv(cell)
    # Bucket grid in fractional coordinates. Require each bucket's
    # Cartesian extent to be ≤ min_distance along every axis, so a
    # 3x3x3 neighbour shell covers the min-image radius.
    # Per-axis lattice length:
    axis_len = np.linalg.norm(cell, axis=1)  # (3,)
    nbucket = np.maximum(np.floor(axis_len / min_distance).astype(int), 1)
    buckets: dict = {}

    def _bucket_of(frac: np.ndarray) -> Tuple[int, int, int]:
        b = np.floor(_wrap_frac(frac) * nbucket).astype(int)
        # Safety: np.floor of 0.999999 * n can still equal n in float edge
        # cases; clamp.
        b = np.minimum(b, nbucket - 1)
        return int(b[0]), int(b[1]), int(b[2])

    neighbour_shell = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]

    accepted_cart = []
    accepted_frac = []

    min_dist_sq = min_distance ** 2

    def _is_far_from_accepted(frac: np.ndarray, pos_cart: np.ndarray) -> bool:
        bx, by, bz = _bucket_of(frac)
        for dx, dy, dz in neighbour_shell:
            key = (
                (bx + dx) % nbucket[0],
                (by + dy) % nbucket[1],
                (bz + dz) % nbucket[2],
            )
            bucket = buckets.get(key)
            if not bucket:
                continue
            others = np.asarray(bucket)  # (k, 3) cart
            r2 = min_image_distance_sq(
                others - pos_cart[None, :], cell, cell_inv
            )
            if r2.min() < min_dist_sq:
                return False
        return True

    for _ in range(max_passes):
        remaining = n_atoms - len(accepted_cart)
        if remaining <= 0:
            break
        batch = min(max(4 * remaining, 64), 4096)
        frac_batch = rng.random((batch, 3))
        cart_batch = frac_batch @ cell

        # Shuffle so within-pass priority is random.
        order = rng.permutation(batch)
        for i in order:
            if n_atoms - len(accepted_cart) <= 0:
                break
            frac = frac_batch[i]
            pos = cart_batch[i]
            if not _is_far_from_accepted(frac, pos):
                continue
            # Commit.
            accepted_cart.append(pos)
            accepted_frac.append(frac)
            bx, by, bz = _bucket_of(frac)
            buckets.setdefault((bx, by, bz), []).append(pos)

    return np.asarray(accepted_cart, dtype=np.float64), len(accepted_cart)


def poisson_disk_pack(
    n_atoms: int,
    cell: np.ndarray,
    min_distance: float,
    *,
    rng: np.random.Generator,
    max_passes: int = 200,
    return_partial: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, int], None]:
    """Batched Poisson-disk sampling in a periodic cell.

    Returns the full ``(n_atoms, 3)`` array on success. On exhaustion:

    - if ``return_partial`` is False: returns ``None``.
    - if ``return_partial`` is True: returns ``(partial_positions, n_placed)``
      where ``partial_positions`` has shape ``(n_placed, 3)`` with
      ``n_placed < n_atoms``.
    """
    cell = np.asarray(cell, dtype=np.float64)
    if cell.shape != (3, 3):
        raise ValueError(f"cell must be (3, 3); got {cell.shape}.")
    if min_distance <= 0:
        raise ValueError(f"min_distance must be > 0; got {min_distance}.")

    if _is_orthorhombic_diagonal(cell):
        diag = np.diag(cell)
        positions, n_placed = _poisson_ortho(
            n_atoms, diag, min_distance, rng, max_passes
        )
    else:
        positions, n_placed = _poisson_triclinic(
            n_atoms, cell, min_distance, rng, max_passes
        )

    if n_placed >= n_atoms:
        return positions[:n_atoms]
    if return_partial:
        return positions, n_placed
    return None


# ---------------------------------------------------------------------------
# MC soft-core fallback
# ---------------------------------------------------------------------------


def _wca_energy_cutoff(sigma: float) -> float:
    return sigma * (2.0 ** (1.0 / 6.0))


def _wca_pair_energy(r2: np.ndarray, sigma: float, epsilon: float) -> np.ndarray:
    """Per-pair WCA energy for an array of squared distances.

    ``U(r) = 4 ε [(σ/r)^12 - (σ/r)^6] + ε`` for ``r < 2^(1/6) σ``, else 0.
    """
    r_cut = _wca_energy_cutoff(sigma)
    mask = (r2 > 0.0) & (r2 < r_cut * r_cut)
    if not np.any(mask):
        return np.zeros_like(r2)
    inv_r2 = np.zeros_like(r2)
    inv_r2[mask] = sigma * sigma / r2[mask]
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    out = np.zeros_like(r2)
    out[mask] = 4.0 * epsilon * (inv_r12[mask] - inv_r6[mask]) + epsilon
    return out


def mc_soft_pack(
    initial_positions: np.ndarray,
    cell: np.ndarray,
    min_distance: float,
    *,
    rng: np.random.Generator,
    n_sweeps: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, float]:
    """Metropolis MC anneal with a WCA pair potential.

    Parameters
    ----------
    initial_positions : (N, 3) array
        Starting configuration (may violate ``min_distance``).
    cell : (3, 3) array
    min_distance : float
        Target minimum pairwise distance.
    rng : numpy Generator
    n_sweeps : int, optional
        Total MC sweeps. Defaults to ``max(20 * N, 2000)``.
    verbose : bool
        If True, print progress.

    Returns
    -------
    positions : (N, 3) array
    final_min_distance : float
        The minimum pairwise distance in the returned configuration.
    """
    positions = np.asarray(initial_positions, dtype=np.float64).copy()
    cell = np.asarray(cell, dtype=np.float64)
    cell_inv = np.linalg.inv(cell)
    N = positions.shape[0]
    if N < 2:
        return positions, float("inf")

    if n_sweeps is None:
        n_sweeps = max(8 * N, 1000)

    sigma = min_distance / (2.0 ** (1.0 / 6.0))
    epsilon = 1.0
    r_cut = _wca_energy_cutoff(sigma)  # == min_distance
    nl_radius = 1.3 * r_cut

    ortho = _is_orthorhombic_diagonal(cell)
    diag = np.diag(cell) if ortho else None

    if ortho:
        positions = np.mod(positions, diag)

    def _build_nbrs() -> list:
        if ortho:
            tree = cKDTree(positions, boxsize=diag)
            # query_ball_tree against self; remove self-index from each list
            nbrs = tree.query_ball_tree(tree, r=nl_radius)
            for i in range(N):
                # Sorted lists; remove self.
                if nbrs[i] and i in nbrs[i]:
                    nbrs[i] = [j for j in nbrs[i] if j != i]
            return nbrs
        # Triclinic: broadcast; cheap at N ≤ few thousand.
        diff = positions[:, None, :] - positions[None, :, :]
        r2 = min_image_distance_sq(diff, cell, cell_inv)
        within = r2 < (nl_radius * nl_radius)
        np.fill_diagonal(within, False)
        return [list(np.flatnonzero(within[i])) for i in range(N)]

    def _atom_energy(idx: int, pos: np.ndarray, nbrs: list) -> float:
        js = nbrs[idx]
        if not js:
            return 0.0
        others = positions[js]
        diff = others - pos[None, :]
        r2 = min_image_distance_sq(diff, cell, cell_inv)
        return float(_wca_pair_energy(r2, sigma, epsilon).sum())

    nbrs = _build_nbrs()

    T_hot, T_cold = 2.0 * epsilon, 1e-2 * epsilon
    step_hot, step_cold = 0.3, 0.05

    rebuild_every = 50
    consecutive_good = 0
    early_stop_thresh = 50
    log_T = math.log(T_cold / T_hot)
    log_S = math.log(step_cold / step_hot)

    for sweep in range(n_sweeps):
        # Geometric anneal.
        prog = sweep / max(n_sweeps - 1, 1)
        T = T_hot * math.exp(log_T * prog)
        step = step_hot * math.exp(log_S * prog)

        order = rng.permutation(N)
        for idx in order:
            old_pos = positions[idx].copy()
            e_old = _atom_energy(idx, old_pos, nbrs)
            trial = old_pos + rng.normal(scale=step, size=3)
            if ortho:
                trial = np.mod(trial, diag)
            e_new = _atom_energy(idx, trial, nbrs)
            de = e_new - e_old
            if de <= 0.0 or rng.random() < math.exp(-de / max(T, 1e-12)):
                positions[idx] = trial

        if (sweep + 1) % rebuild_every == 0:
            nbrs = _build_nbrs()

        # Early-stop probe (cheap: reuse the neighbour list, take the
        # minimum r across all pairs within nl_radius).
        if (sweep + 1) % 10 == 0:
            if ortho:
                tree = cKDTree(positions, boxsize=diag)
                d, _ = tree.query(positions, k=2)
                dmin = float(d[:, 1].min())
            else:
                dmin = min_pairwise_distance(positions, cell)
            if dmin >= min_distance:
                consecutive_good += 1
            else:
                consecutive_good = 0
            if consecutive_good >= early_stop_thresh // 10:
                if verbose:
                    print(
                        f"[mc_soft_pack] early stop at sweep {sweep+1}/"
                        f"{n_sweeps}; dmin={dmin:.3f}"
                    )
                break
            if verbose and (sweep + 1) % 100 == 0:
                print(
                    f"[mc_soft_pack] sweep {sweep+1}/{n_sweeps} "
                    f"T={T:.3e} step={step:.3f} dmin={dmin:.3f}"
                )

    final_min = min_pairwise_distance(positions, cell)
    return positions, final_min


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def pack(
    n_atoms: int,
    cell: np.ndarray,
    min_distance: float,
    *,
    rng: np.random.Generator,
    verbose: bool = False,
    echo: Optional[Callable[[str], None]] = None,
    max_passes: int = 200,
) -> np.ndarray:
    """Produce ``n_atoms`` positions in ``cell`` with minimum pairwise
    distance ≥ ``min_distance`` where possible.

    Tries Poisson-disk sampling first. On exhaustion, falls back to a
    Metropolis-Monte-Carlo anneal with a WCA repulsive potential, seeded
    with the partial Poisson-disk set plus uniform-random fill.

    If the MC fallback cannot reach ``min_distance`` everywhere, a
    ``UserWarning`` is emitted. If the final minimum distance falls below
    ``0.95 * min_distance`` the warning is escalated with "below tolerance".
    """
    cell = np.asarray(cell, dtype=np.float64)
    _emit = echo if echo is not None else (lambda _msg: None)

    result = poisson_disk_pack(
        n_atoms,
        cell,
        min_distance,
        rng=rng,
        max_passes=max_passes,
        return_partial=True,
    )

    if isinstance(result, np.ndarray) and result.shape[0] == n_atoms:
        return result

    # Exhausted: result is (partial, n_placed).
    partial, n_placed = result  # type: ignore[misc]
    _emit(
        f"Poisson-disk placed {n_placed}/{n_atoms}; "
        "engaging MC soft-core anneal fallback."
    )

    # Uniform-random fill for the remainder (ignores overlaps; MC will
    # relax them).
    remaining = n_atoms - n_placed
    if remaining > 0:
        frac_fill = rng.random((remaining, 3))
        cart_fill = frac_fill @ cell
        seed_positions = np.vstack([partial, cart_fill])
    else:
        seed_positions = partial

    positions, final_min = mc_soft_pack(
        seed_positions, cell, min_distance, rng=rng, verbose=verbose
    )

    if final_min < 0.95 * min_distance:
        warnings.warn(
            f"MC fallback converged to min pairwise distance {final_min:.3f} "
            f"Å < 0.95 * {min_distance:.3f} = {0.95 * min_distance:.3f} Å. "
            "Requested packing fraction may exceed the feasible regime; "
            "consider relaxing min_distance or lowering density.",
            stacklevel=2,
        )
    elif final_min < min_distance:
        warnings.warn(
            f"MC fallback settled at min pairwise distance {final_min:.3f} "
            f"Å, slightly below min_distance={min_distance:.3f} Å.",
            stacklevel=2,
        )

    return positions
