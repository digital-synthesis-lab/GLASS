"""Keating energy/force unit tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from glass.algorithms.crn.network import Network
from glass.potentials.keating import KeatingParameters, TorchKeating


def diamond_supercell(n_cells: int = 2) -> Network:
    """Build an n×n×n cubic-diamond Si supercell with bonds."""
    d0 = 2.35  # Default Keating bond length
    a = d0 * 4.0 / np.sqrt(3.0)  # cubic lattice constant for ideal Si
    basis = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.50, 0.50, 0.00],
            [0.50, 0.00, 0.50],
            [0.00, 0.50, 0.50],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ]
    )
    positions = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                shift = np.array([i, j, k], dtype=float)
                positions.append((basis + shift) * a)
    positions = np.vstack(positions)
    cell = np.diag([n_cells * a] * 3).astype(np.float64)

    # Build bonds via min-image distance < 1.1·d.
    box = np.diag(cell)
    n = positions.shape[0]
    bonds = []
    cutoff = 1.1 * d0
    for i in range(n):
        for j in range(i + 1, n):
            d = positions[j] - positions[i]
            d -= box * np.round(d / box)
            if np.dot(d, d) < cutoff * cutoff:
                bonds.append((i, j))
    bonds = np.asarray(bonds, dtype=np.int64)
    return Network.from_bonds(positions, cell, bonds, d=d0)


def test_diamond_energy_is_zero():
    """Test that ideal diamond Si has zero Keating energy."""
    net = diamond_supercell(n_cells=2)
    params = KeatingParameters(d=net.d)
    calc = TorchKeating(params, dtype=torch.float64)

    energy = calc.energy(
        torch.tensor(net.positions, dtype=torch.float64),
        torch.tensor(net.cell, dtype=torch.float64),
        torch.tensor(net.bonds, dtype=torch.int64),
        torch.tensor(net.neigh, dtype=torch.int64),
        torch.tensor(net.degree, dtype=torch.int64),
        pbc=True,
    )

    assert np.all(net.degree == 4), f"diamond not 4-coord: {np.bincount(net.degree)}"
    assert abs(energy.item()) < 1e-8, f"E should be 0 for ideal diamond, got {energy.item()}"


def test_diamond_forces_are_zero():
    """Test that ideal diamond Si has zero forces."""
    net = diamond_supercell(n_cells=2)
    params = KeatingParameters(d=net.d)
    calc = TorchKeating(params, dtype=torch.float64)

    energy, forces = calc.energy_and_forces_autograd(
        torch.tensor(net.positions, dtype=torch.float64),
        torch.tensor(net.cell, dtype=torch.float64),
        torch.tensor(net.bonds, dtype=torch.int64),
        torch.tensor(net.neigh, dtype=torch.int64),
        torch.tensor(net.degree, dtype=torch.int64),
        pbc=True,
    )

    max_force = torch.max(torch.abs(forces)).item()
    assert max_force < 1e-7, f"|F|_max = {max_force}"


def test_finite_difference_forces():
    """Test analytical forces against numerical finite differences."""
    net = diamond_supercell(n_cells=2)
    rng = np.random.default_rng(7)
    net.positions += rng.normal(0, 0.05, size=net.positions.shape)

    params = KeatingParameters(d=net.d)
    calc = TorchKeating(params, dtype=torch.float64)

    _, f_analytical = calc.energy_and_forces_autograd(
        torch.tensor(net.positions, dtype=torch.float64),
        torch.tensor(net.cell, dtype=torch.float64),
        torch.tensor(net.bonds, dtype=torch.int64),
        torch.tensor(net.neigh, dtype=torch.int64),
        torch.tensor(net.degree, dtype=torch.int64),
        pbc=True,
    )
    f_analytical = f_analytical.numpy()

    eps = 1e-5
    # Sample a few atoms to keep the test fast.
    for i in [0, 5, 17]:
        for k in range(3):
            net.positions[i, k] += eps
            e_plus = calc.energy(
                torch.tensor(net.positions, dtype=torch.float64),
                torch.tensor(net.cell, dtype=torch.float64),
                torch.tensor(net.bonds, dtype=torch.int64),
                torch.tensor(net.neigh, dtype=torch.int64),
                torch.tensor(net.degree, dtype=torch.int64),
                pbc=True,
            ).item()

            net.positions[i, k] -= 2 * eps
            e_minus = calc.energy(
                torch.tensor(net.positions, dtype=torch.float64),
                torch.tensor(net.cell, dtype=torch.float64),
                torch.tensor(net.bonds, dtype=torch.int64),
                torch.tensor(net.neigh, dtype=torch.int64),
                torch.tensor(net.degree, dtype=torch.int64),
                pbc=True,
            ).item()

            net.positions[i, k] += eps
            f_num = -(e_plus - e_minus) / (2 * eps)
            assert abs(f_num - f_analytical[i, k]) < 1e-3, (
                f"force mismatch at atom {i} dim {k}: "
                f"analytical={f_analytical[i, k]}, numerical={f_num}"
            )
