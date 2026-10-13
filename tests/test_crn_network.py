"""Loop-expansion (BM §A) initialization tests."""

from __future__ import annotations

import numpy as np
import pytest

from glass.algorithms.crn.initialize import random_initial_network
from glass.algorithms.crn.network import MAX_DEGREE


@pytest.mark.parametrize("n_atoms,seed", [(64, 0), (64, 11), (216, 3)])
def test_loop_expansion_4_regular(n_atoms: int, seed: int) -> None:
    """Test that loop expansion produces valid 4-coordinated networks."""
    net = random_initial_network(n_atoms, seed=seed)
    assert net.n_atoms == n_atoms
    assert np.all(net.degree == MAX_DEGREE), (
        f"degrees not all 4: histogram = {np.bincount(net.degree)}"
    )
    assert net.bonds.shape == (2 * n_atoms, 2)

    # No self-loops or duplicate bonds.
    canon = np.sort(net.bonds, axis=1)
    uniq = {tuple(b) for b in canon}
    assert len(uniq) == 2 * n_atoms
    assert all(a != b for a, b in canon)

    # Each bond appears in both endpoints' adjacency.
    for i, j in net.bonds:
        assert net.has_bond(int(i), int(j))
        assert net.has_bond(int(j), int(i))


def test_apply_revert_swap_is_identity() -> None:
    """Test that apply_swap followed by revert_swap is identity."""
    from glass.algorithms.crn.transposition import apply_swap, propose_swap, revert_swap

    rng = np.random.default_rng(0)
    net = random_initial_network(64, seed=0, rng=rng)

    # find a valid swap
    for _ in range(200):
        move = propose_swap(net, rng)
        if move is not None:
            break
    assert move is not None
    a, b, c, d = move

    bonds_before = np.sort(net.bonds, axis=1)
    bonds_before = bonds_before[np.lexsort(bonds_before.T[::-1])]
    deg_before = net.degree.copy()
    neigh_sets_before = [
        set(net.neigh[i, : net.degree[i]].tolist()) for i in range(net.n_atoms)
    ]

    apply_swap(net, a, b, c, d)
    revert_swap(net, a, b, c, d)

    bonds_after = np.sort(net.bonds, axis=1)
    bonds_after = bonds_after[np.lexsort(bonds_after.T[::-1])]
    assert np.array_equal(bonds_before, bonds_after)
    assert np.array_equal(deg_before, net.degree)
    for i in range(net.n_atoms):
        assert set(net.neigh[i, : net.degree[i]].tolist()) == neigh_sets_before[i]
