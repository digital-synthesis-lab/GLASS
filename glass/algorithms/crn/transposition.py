"""WWW bond transposition primitives.

Given atoms (A, B, C, D) such that bonds (A-B), (B-C), (C-D) all exist,
the WWW move replaces (A-B) and (C-D) with (A-C) and (B-D). The (B-C)
bond is unaffected. All four atoms keep degree 4.

Selection follows BM eq. count: N · 4 · 3 · 3 / 2 = 18N enumerations
(B, then A∈N(B), then C∈N(B)\{A}, then D∈N(C)\{B}; divide by 2 because
(A,B,C,D) ≡ (D,C,B,A)).
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Optional

import numpy as np

from glass.algorithms.crn.network import MAX_DEGREE, Network


# ---------------------------------------------------------------------- #
# Selection
# ---------------------------------------------------------------------- #


def propose_swap(
    net: Network, rng: np.random.Generator, *, allow_4rings: bool = True
) -> Optional[tuple[int, int, int, int]]:
    """Return (A, B, C, D) for a valid WWW move, or None after rejecting.

    Single attempt — caller should retry. Forbids self-loops, duplicate
    bonds (AC or BD already present), and optionally 4-rings.
    """
    n = net.n_atoms
    b = int(rng.integers(0, n))
    db = int(net.degree[b])
    if db == 0:
        return None

    a = int(net.neigh[b, rng.integers(0, db)])

    # C ∈ N(B) \ {A}
    if db < 2:
        return None
    while True:
        c = int(net.neigh[b, rng.integers(0, db)])
        if c != a:
            break

    dc = int(net.degree[c])
    if dc < 2:
        return None
    while True:
        d = int(net.neigh[c, rng.integers(0, dc)])
        if d != b:
            break

    if a == d:
        return None
    if net.has_bond(a, c):
        return None
    if net.has_bond(b, d):
        return None
    if not allow_4rings and _creates_4ring(net, a, b, c, d):
        return None
    return a, b, c, d


def _creates_4ring(net: Network, a: int, b: int, c: int, d: int) -> bool:
    """After the swap, the new edges are A-C and B-D. A 4-ring forms
    iff A-C-?-?-A or B-D-?-?-B is closed by old edges."""
    # AC ring: need A-C-x-y-A with x, y atoms. After swap, A's
    # neighbours become old(N(A))\{B} ∪ {C}. Check if C and any of A's
    # post-swap neighbours share a common neighbour besides A and B.
    a_post = _post_swap_neighbors(net, a, drop=b, add=c)
    c_post = _post_swap_neighbors(net, c, drop=d, add=a)
    if _share_short_path(a, c, a_post, c_post, net, exclude={a, c, b, d}):
        return True
    b_post = _post_swap_neighbors(net, b, drop=a, add=d)
    d_post = _post_swap_neighbors(net, d, drop=c, add=b)
    if _share_short_path(b, d, b_post, d_post, net, exclude={a, b, c, d}):
        return True
    return False


def _post_swap_neighbors(net: Network, i: int, drop: int, add: int) -> set[int]:
    di = int(net.degree[i])
    s = set(int(net.neigh[i, k]) for k in range(di))
    s.discard(drop)
    s.add(add)
    return s


def _share_short_path(
    i: int, j: int, ni: set[int], nj: set[int], net: Network, exclude: set[int]
) -> bool:
    # 4-ring i-x-y-j-i needs x ∈ ni, y ∈ nj, with x-y bonded.
    for x in ni - exclude - {j}:
        for k in range(int(net.degree[x])):
            y = int(net.neigh[x, k])
            if y in nj and y != i and y not in exclude:
                return True
    return False


# ---------------------------------------------------------------------- #
# Apply / revert
# ---------------------------------------------------------------------- #


def apply_swap(net: Network, a: int, b: int, c: int, d: int) -> None:
    """Mutate `net` in place: drop (a,b) and (c,d), add (a,c) and (b,d)."""
    net.remove_bond(a, b)
    net.remove_bond(c, d)
    net.add_bond(a, c)
    net.add_bond(b, d)


def revert_swap(net: Network, a: int, b: int, c: int, d: int) -> None:
    """Undo `apply_swap(net, a, b, c, d)`."""
    net.remove_bond(a, c)
    net.remove_bond(b, d)
    net.add_bond(a, b)
    net.add_bond(c, d)


# ---------------------------------------------------------------------- #
# Topology helpers
# ---------------------------------------------------------------------- #


def shells_around(net: Network, seeds: Iterable[int], depth: int = 3) -> set[int]:
    """BFS up to `depth` bonds away from any seed atom (inclusive)."""
    out: set[int] = set()
    frontier = deque((int(s), 0) for s in seeds)
    visited: set[int] = set(int(s) for s in seeds)
    out.update(visited)
    while frontier:
        u, h = frontier.popleft()
        if h == depth:
            continue
        for k in range(int(net.degree[u])):
            v = int(net.neigh[u, k])
            if v not in visited:
                visited.add(v)
                out.add(v)
                frontier.append((v, h + 1))
    return out


def detect_4rings(net: Network) -> list[tuple[int, int, int, int]]:
    """Return one canonical 4-ring per ring (sorted tuple)."""
    rings: set[tuple[int, int, int, int]] = set()
    n = net.n_atoms
    for i in range(n):
        di = int(net.degree[i])
        for a in range(di):
            j = int(net.neigh[i, a])
            if j <= i:
                continue
            for b in range(int(net.degree[j])):
                k = int(net.neigh[j, b])
                if k == i:
                    continue
                for c in range(int(net.degree[k])):
                    l = int(net.neigh[k, c])
                    if l == j or l == i:
                        continue
                    if net.has_bond(l, i):
                        ring = tuple(sorted((i, j, k, l)))
                        rings.add(ring)
    return list(rings)
