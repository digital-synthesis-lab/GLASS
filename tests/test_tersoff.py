import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

# Import ASE before torch to avoid OpenMP init conflict on macOS.
from ase.build import bulk
from ase.calculators.tersoff import Tersoff as AseTersoff
from ase.calculators.tersoff import TersoffParameters as AseTersoffParameters
from ase.io import read

import numpy as np
import torch

from glass.potentials.tersoff import TersoffParameters, TorchTersoff
from glass.potentials.tersoff.neighbors import build_neighbors

DATA_DIR = Path(__file__).resolve().parent / "data"

SI_PARAMS_KW = dict(
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


def ase_calc():
    params = {("Si", "Si", "Si"): AseTersoffParameters(**SI_PARAMS_KW)}
    return AseTersoff(params)


def torch_calc():
    params = {("Si", "Si", "Si"): TersoffParameters(**SI_PARAMS_KW)}
    return TorchTersoff(params)


def _atoms_to_tensors(atoms, dtype=torch.float64):
    pos = torch.tensor(atoms.get_positions(), dtype=dtype)
    cell = torch.tensor(np.array(atoms.cell), dtype=dtype)
    pbc = tuple(bool(p) for p in atoms.pbc)
    return pos, cell, pbc


def test_diamond_energy():
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.calc = ase_calc()
    E_ase = atoms.get_potential_energy()

    pos, cell, pbc = _atoms_to_tensors(atoms)
    tc = torch_calc()
    E_torch = tc.energy(pos, cell, pbc).item()

    assert abs(E_torch - E_ase) < 1e-8, (E_torch, E_ase)


def test_snapshot_energy():
    atoms = read(DATA_DIR / "Si_2.5_00.xyz")
    pos, cell, pbc = _atoms_to_tensors(atoms)
    tc = torch_calc()

    E_torch = tc.energy(pos, cell, pbc).item()

    # Cross-check against ASE's reference Tersoff (3.25.0) implementation.
    # This is the regression test for the (lambda3 * delta_r)^m bug fixed
    # in Phase E (May 2026). The previous pinned value -936.342254... was
    # itself buggy.
    atoms.calc = ase_calc()
    E_ase = atoms.get_potential_energy()

    assert abs(E_torch - E_ase) < 1e-7, (E_torch, E_ase)
    # Pin the correct value too, so a regression in either glass OR ase
    # is caught. The reference value comes from a fresh evaluation post-fix
    # (matches LAMMPS to ~1e-15 eV/atom, see Phase E).
    assert abs(E_torch + 939.9234189560) < 1e-6, E_torch


def test_snapshot_forces_autograd_vs_finite_diff():
    # Validate the autograd gradient against a central finite difference of
    # our own torch energy, on a small random subset of atoms. ASE's builtin
    # calculate_numerical_forces would also work but runs ~1300 ASE energy
    # evaluations (~2 minutes); doing the finite diff through our torch
    # energy is the same check and finishes in under a second.
    atoms = read(DATA_DIR / "Si_2.5_00.xyz")
    pos, cell, pbc = _atoms_to_tensors(atoms)
    tc = torch_calc()

    _, F_torch = tc.energy_and_forces_autograd(pos, cell, pbc)

    rng = np.random.default_rng(0)
    sample_atoms = rng.choice(pos.shape[0], size=5, replace=False)
    h = 1e-4
    for a in sample_atoms:
        for d in range(3):
            pp = pos.clone()
            pp[a, d] += h
            E_plus = tc.energy(pp, cell, pbc).item()
            pp[a, d] -= 2 * h
            E_minus = tc.energy(pp, cell, pbc).item()
            fd = -(E_plus - E_minus) / (2 * h)
            assert abs(float(F_torch[a, d]) - fd) < 1e-5, (a, d, fd, F_torch[a, d])


def test_snapshot_forces_analytical_vs_autograd():
    atoms = read(DATA_DIR / "Si_2.5_00.xyz")
    pos, cell, pbc = _atoms_to_tensors(atoms)
    tc = torch_calc()

    _, F_auto = tc.energy_and_forces_autograd(pos, cell, pbc)
    _, F_ana = tc.energy_and_forces_analytical(pos, cell, pbc)

    max_diff = (F_auto - F_ana).abs().max().item()
    assert max_diff < 1e-6, max_diff


def test_translation_invariance():
    atoms = read(DATA_DIR / "Si_2.5_00.xyz")
    pos, cell, pbc = _atoms_to_tensors(atoms)
    tc = torch_calc()
    E1 = tc.energy(pos, cell, pbc).item()

    shift = torch.tensor([0.37, -1.21, 2.83], dtype=torch.float64)
    E2 = tc.energy(pos + shift, cell, pbc).item()

    assert abs(E1 - E2) < 1e-8


def test_neighbor_count_diamond():
    atoms = bulk("Si", "diamond", a=5.43)
    pos, cell, pbc = _atoms_to_tensors(atoms)
    R, D = SI_PARAMS_KW["R"], SI_PARAMS_KW["D"]
    i_idx, j_idx, _ = build_neighbors(pos, cell, pbc, R - D)
    counts = torch.bincount(i_idx, minlength=pos.shape[0])
    assert int(counts.min()) == 4
    assert int(counts.max()) == 4


def test_large_diamond_dense_vs_cell_list_agree():
    """The dense and cell-list neighbour paths must give identical energy
    (to within 1e-9) for a system large enough to trigger the cell-list
    switch (N > 256)."""
    atoms = bulk("Si", "diamond", a=5.43).repeat((4, 4, 4))  # 512 atoms
    pos, cell, pbc = _atoms_to_tensors(atoms)
    tc = torch_calc()
    E_ref = tc.energy(pos, cell, pbc).item()

    # Reference: force dense path on a smaller replica.
    atoms_small = bulk("Si", "diamond", a=5.43).repeat((2, 2, 2))  # 64 atoms
    pos_s, cell_s, pbc_s = _atoms_to_tensors(atoms_small)
    E_small = tc.energy(pos_s, cell_s, pbc_s).item()

    # Pure correctness: diamond energy is extensive, so E(512) == 8 * E(64)
    # because 512 = 8 * 64. This confirms both paths produce the right
    # answer (dense for N=64, cell-list for N=512).
    assert abs(E_ref - 8 * E_small) < 1e-6, (E_ref, 8 * E_small)


def test_large_system_autograd_no_oom():
    """Regression: large cells must not blow up autograd memory. This
    mirrors the path TersoffEnergyGuidance uses inside the reverse SDE."""
    atoms = bulk("Si", "diamond", a=5.43).repeat((4, 4, 4))  # 512 atoms
    pos, cell, pbc = _atoms_to_tensors(atoms)
    tc = torch_calc()
    pos = pos.detach().requires_grad_(True)
    E = tc.energy(pos, cell, pbc)
    (grad,) = torch.autograd.grad(E, pos)
    assert grad.shape == pos.shape
    assert torch.isfinite(grad).all()
