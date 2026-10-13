"""WWW driver — anneal/quench cycles per Barkema-Mousseau (2000).

Procedure:

1.  Build a random 4-coordinated initial network (loop expansion).
2.  Full FIRE relaxation — drops the angular spread from ~30° to ~13°.
3.  Repeat ``n_cycles`` times:
      a. Anneal at ``kT`` for ``n_anneal_per_atom · N`` proposed bond
         transpositions, each followed by local-then-full FIRE
         relaxation with BM early-reject.
      b. T=0 quench: keep proposing transpositions and accept only if
         they lower the energy after relaxation, until no swap from a
         large random pool succeeds.
4.  Optional: remove any 4-membered rings introduced during quenching.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from glass.algorithms.crn.initialize import random_initial_network
from glass.algorithms.crn.network import Network
from glass.algorithms.crn.relax import relax_full, relax_with_threshold
from glass.algorithms.crn.transposition import (
    apply_swap,
    detect_4rings,
    propose_swap,
    revert_swap,
    shells_around,
)

# Use fast Numba-based Keating for performance
from glass.algorithms.crn._keating_numba import keating_energy


def _keating_energy(net: Network) -> float:
    """Compute Keating energy for the network."""
    return keating_energy(net)


@dataclass
class WWWStats:
    initial_energy: float = 0.0
    final_energy: float = 0.0
    n_proposed: int = 0
    n_accepted: int = 0
    n_quench_accepted: int = 0
    cycle_energies: list[float] = field(default_factory=list)


def generate_crn(
    n_atoms: int = 216,
    *,
    density: float = 2.33,
    seed: int = 0,
    n_cycles: int = 5,
    n_anneal_per_atom: int = 50,
    kT: float = 0.25,
    quench_attempts_per_atom: int = 18,
    relax_local_steps: int = 10,
    relax_full_max_iter: int = 200,
    ftol: float = 1e-3,
    cf_threshold: float = 0.5,
    allow_4rings: bool = True,
    final_4ring_removal: bool = True,
    rng: Optional[np.random.Generator] = None,
    log: Optional[Callable[[str], None]] = None,
) -> tuple[Network, WWWStats]:
    """Generate a CRN. Returns (network, stats).

    Defaults are set for fast iteration on N=216. The original BM paper
    uses ``n_anneal_per_atom=100`` plus very many cycles; tighten those
    for higher quality.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    _log = log if log is not None else (lambda _: None)

    _log(f"[init] generating random initial network (N={n_atoms}, ρ={density})")
    t0 = time.time()
    net = random_initial_network(n_atoms, density=density, rng=rng)
    _log(
        f"[init] loop expansion done in {time.time() - t0:.1f}s; "
        f"degrees={np.unique(net.degree, return_counts=True)}"
    )

    _log("[init] initial full relaxation")
    t0 = time.time()
    e0 = relax_full(net, max_iter=relax_full_max_iter * 5, ftol=ftol)
    _log(
        f"[init] E0 = {e0:.4f} eV ({e0 / n_atoms:.4f} eV/atom) "
        f"in {time.time() - t0:.1f}s"
    )
    stats = WWWStats(initial_energy=e0)

    # ---- Anneal/quench cycles -------------------------------------- #
    for cycle in range(n_cycles):
        n_proposed = 0
        n_accepted = 0
        n_per_cycle = n_anneal_per_atom * n_atoms
        t0 = time.time()
        for _ in range(n_per_cycle):
            n_proposed += 1
            stats.n_proposed += 1
            move = propose_swap(net, rng, allow_4rings=allow_4rings)
            if move is None:
                continue
            a, b, c, d = move
            atom_set = shells_around(net, (a, b, c, d), depth=3)

            e_b = _keating_energy(net)
            u = float(rng.random())
            # Avoid log(0) — clamp.
            u = max(u, 1e-12)
            e_t = e_b - kT * math.log(u)

            saved_pos = net.positions.copy()
            apply_swap(net, a, b, c, d)
            e_f, accepted = relax_with_threshold(
                net,
                atom_set=atom_set,
                e_threshold=e_t,
                local_steps=relax_local_steps,
                full_max_iter=relax_full_max_iter,
                ftol=ftol,
                cf=cf_threshold,
            )
            if not accepted or e_f > e_t:
                revert_swap(net, a, b, c, d)
                net.positions[:] = saved_pos
            else:
                n_accepted += 1
                stats.n_accepted += 1

        e_after_anneal = _keating_energy(net)
        accept_pct = 100.0 * n_accepted / max(n_proposed, 1)
        _log(
            f"[cycle {cycle + 1}/{n_cycles}] anneal: "
            f"{n_accepted}/{n_proposed} accepted ({accept_pct:.2f}%); "
            f"E={e_after_anneal:.3f} eV ({e_after_anneal / n_atoms:.4f} eV/atom); "
            f"{time.time() - t0:.1f}s"
        )

        # ---- T=0 quench ------------------------------------------- #
        t0 = time.time()
        n_q_accepted = _quench(
            net,
            rng,
            n_attempts=quench_attempts_per_atom * n_atoms,
            relax_local_steps=relax_local_steps,
            relax_full_max_iter=relax_full_max_iter,
            ftol=ftol,
            cf_threshold=cf_threshold,
            allow_4rings=allow_4rings,
        )
        stats.n_quench_accepted += n_q_accepted
        e_after_quench = _keating_energy(net)
        stats.cycle_energies.append(e_after_quench)
        _log(
            f"[cycle {cycle + 1}/{n_cycles}] quench: {n_q_accepted} swaps; "
            f"E={e_after_quench:.3f} eV ({e_after_quench / n_atoms:.4f} eV/atom); "
            f"{time.time() - t0:.1f}s"
        )

    # ---- Final cleanup -------------------------------------------- #
    if final_4ring_removal:
        rings = detect_4rings(net)
        if rings:
            _log(f"[final] removing {len(rings)} 4-membered ring(s)")
            _remove_4rings(
                net, rng, relax_full_max_iter=relax_full_max_iter, ftol=ftol
            )
        else:
            _log("[final] no 4-membered rings present")

    relax_full(net, max_iter=relax_full_max_iter * 5, ftol=ftol)
    stats.final_energy = _keating_energy(net)
    _log(
        f"[done] E_final = {stats.final_energy:.3f} eV "
        f"({stats.final_energy / n_atoms:.4f} eV/atom); "
        f"accept rate = {100.0 * stats.n_accepted / max(stats.n_proposed, 1):.2f}%"
    )
    return net, stats


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _quench(
    net: Network,
    rng: np.random.Generator,
    *,
    n_attempts: int,
    relax_local_steps: int,
    relax_full_max_iter: int,
    ftol: float,
    cf_threshold: float,
    allow_4rings: bool,
) -> int:
    """T=0 quench: accept only if energy strictly decreases after relax.

    Tries ``n_attempts`` random transpositions; we don't enumerate the
    full 18N space (that'd be slower than a few sweeps for the gains).
    """
    accepted = 0
    for _ in range(n_attempts):
        move = propose_swap(net, rng, allow_4rings=allow_4rings)
        if move is None:
            continue
        a, b, c, d = move
        atom_set = shells_around(net, (a, b, c, d), depth=3)
        e_b = _keating_energy(net)
        saved_pos = net.positions.copy()
        apply_swap(net, a, b, c, d)
        e_f, ok = relax_with_threshold(
            net,
            atom_set=atom_set,
            e_threshold=e_b,  # threshold = current energy → only accept on decrease
            local_steps=relax_local_steps,
            full_max_iter=relax_full_max_iter,
            ftol=ftol,
            cf=cf_threshold,
        )
        if not ok or e_f >= e_b:
            revert_swap(net, a, b, c, d)
            net.positions[:] = saved_pos
        else:
            accepted += 1
    return accepted


def _remove_4rings(
    net: Network, rng: np.random.Generator, *, relax_full_max_iter: int, ftol: float
) -> None:
    """Remove 4-rings one at a time by trying each transposition that
    breaks the ring; keep the lowest-energy successful swap."""
    rings = detect_4rings(net)
    while rings:
        ring = rings[0]
        best = None
        best_e = float("inf")
        # All 4 edges of the ring are candidates for AB.
        atoms = list(ring)
        edges = []
        for i in range(4):
            for j in range(i + 1, 4):
                if net.has_bond(atoms[i], atoms[j]):
                    edges.append((atoms[i], atoms[j]))
        candidates: list[tuple[int, int, int, int]] = []
        for ab in edges:
            for a, b in (ab, (ab[1], ab[0])):
                for k in range(int(net.degree[b])):
                    c = int(net.neigh[b, k])
                    if c == a:
                        continue
                    for l_idx in range(int(net.degree[c])):
                        d = int(net.neigh[c, l_idx])
                        if d == b or d == a:
                            continue
                        if net.has_bond(a, c) or net.has_bond(b, d):
                            continue
                        candidates.append((a, b, c, d))
        for a, b, c, d in candidates:
            saved_pos = net.positions.copy()
            saved_bonds = net.bonds.copy()
            saved_neigh = net.neigh.copy()
            saved_deg = net.degree.copy()
            apply_swap(net, a, b, c, d)
            if detect_4rings(net):
                net.bonds = saved_bonds
                net.neigh = saved_neigh
                net.degree = saved_deg
                net.positions = saved_pos
                continue
            e = relax_full(net, max_iter=relax_full_max_iter, ftol=ftol)
            if e < best_e:
                best_e = e
                best = (
                    a,
                    b,
                    c,
                    d,
                    net.positions.copy(),
                    net.bonds.copy(),
                    net.neigh.copy(),
                    net.degree.copy(),
                )
            net.bonds = saved_bonds
            net.neigh = saved_neigh
            net.degree = saved_deg
            net.positions = saved_pos
        if best is None:
            # Couldn't remove this ring without making another.
            break
        _, _, _, _, pos, bonds, neigh, deg = best
        net.positions = pos
        net.bonds = bonds
        net.neigh = neigh
        net.degree = deg
        rings = detect_4rings(net)
