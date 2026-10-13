"""Variable-density CRN generation tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from glass.algorithms.crn.initialize import random_initial_network
from glass.algorithms.crn.network import (
    MAX_DEGREE,
    Network,
    bond_length_for_density,
    cubic_cell_for_si,
)
from glass.algorithms.crn.www import generate_crn
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


def _scaled_diamond(d: float, n_cells: int = 2) -> Network:
    """Diamond supercell whose ideal bond length is `d`."""
    base = diamond_supercell(n_cells=n_cells)
    scale = d / base.d
    base.positions *= scale
    base.cell *= scale
    base.d = d
    return base


def _keating_energy(net: Network, d: float | None = None) -> float:
    """Compute Keating energy."""
    d_use = net.d if d is None else d
    params = KeatingParameters(d=d_use)
    calc = TorchKeating(params, dtype=torch.float64)
    energy = calc.energy(
        torch.tensor(net.positions, dtype=torch.float64),
        torch.tensor(net.cell, dtype=torch.float64),
        torch.tensor(net.bonds, dtype=torch.int64),
        torch.tensor(net.neigh, dtype=torch.int64),
        torch.tensor(net.degree, dtype=torch.int64),
        pbc=True,
    )
    return float(energy.item())


def _keating_forces(net: Network, d: float | None = None):
    """Compute Keating energy and forces."""
    d_use = net.d if d is None else d
    params = KeatingParameters(d=d_use)
    calc = TorchKeating(params, dtype=torch.float64)
    energy, forces = calc.energy_and_forces_autograd(
        torch.tensor(net.positions, dtype=torch.float64),
        torch.tensor(net.cell, dtype=torch.float64),
        torch.tensor(net.bonds, dtype=torch.int64),
        torch.tensor(net.neigh, dtype=torch.int64),
        torch.tensor(net.degree, dtype=torch.int64),
        pbc=True,
    )
    return float(energy.item()), forces.numpy()


@pytest.mark.parametrize("density", [2.10, 2.33, 2.55])
def test_bond_length_scaling_inverse_cube_root(density: float) -> None:
    """Test that density scaling formula is self-consistent."""
    d = bond_length_for_density(density)
    # Round-trip: diamond at this density should have unit-cell-volume
    # consistent with d = a√3/4 → a = 4d/√3, V_atom = a³/8.
    a_cubic = 4.0 * d / np.sqrt(3.0)
    v_per_atom_A3 = (a_cubic**3) / 8.0
    # ρ = m/V → m_si / V_atom in g/cm³
    m_si_g = 28.0855 / 6.022_140_76e23
    rho_back = m_si_g / (v_per_atom_A3 * 1e-24)
    assert abs(rho_back - density) < 1e-6, (
        f"density round-trip failed: ρ={density}, d={d}, ρ_back={rho_back}"
    )


@pytest.mark.parametrize("density", [2.10, 2.33, 2.55])
def test_diamond_zero_energy_at_density(density: float) -> None:
    """The Keating energy of an ideal diamond at density ρ must be zero
    when the bond length is set to d(ρ)."""
    d = bond_length_for_density(density)
    net = _scaled_diamond(d)
    e = _keating_energy(net)
    assert np.all(net.degree == MAX_DEGREE)
    assert abs(e) < 1e-6, f"E should be ~0 for ideal diamond at ρ={density}, got {e}"
    _, f = _keating_forces(net)
    assert np.max(np.abs(f)) < 1e-6


@pytest.mark.parametrize("density", [2.10, 2.55])
def test_loop_expansion_at_density(density: float) -> None:
    """Loop expansion should reach 4-regularity at non-default densities."""
    net = random_initial_network(n_atoms=64, density=density, seed=0)
    assert net.n_atoms == 64
    assert np.all(net.degree == MAX_DEGREE), (
        f"degrees not all 4 at ρ={density}: {np.bincount(net.degree)}"
    )
    assert net.bonds.shape == (128, 2)
    # And the network records the density-scaled bond length.
    expected_d = bond_length_for_density(density)
    assert abs(net.d - expected_d) < 1e-9


def test_network_d_is_used_by_default() -> None:
    """Energy uses net.d unless the caller passes an explicit d."""
    net = _scaled_diamond(d=2.50)  # off from the default 2.35
    e_default = _keating_energy(net)
    e_override_correct = _keating_energy(net, d=2.50)
    e_override_wrong = _keating_energy(net, d=2.35)
    assert abs(e_default - e_override_correct) < 1e-12
    # With the wrong d, ideal bonds at 2.50 Å look stretched → E > 0.
    assert e_override_wrong > 1e-3


@pytest.mark.parametrize("density", [2.10, 2.55])
def test_smoke_generate_at_density(density: float) -> None:
    """End-to-end short run: any density should still yield a 4-coord CRN."""
    net, stats = generate_crn(
        n_atoms=64,
        density=density,
        seed=0,
        n_cycles=1,
        n_anneal_per_atom=4,
        quench_attempts_per_atom=3,
        relax_local_steps=4,
        relax_full_max_iter=40,
    )
    assert net.n_atoms == 64
    assert np.all(net.degree == MAX_DEGREE)
    assert abs(net.d - bond_length_for_density(density)) < 1e-9
    assert stats.final_energy <= stats.initial_energy + 1e-6
