import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import pytest
import torch
from ase.io import read

from glass.lit.modules.tersoff_guidance import (
    TersoffEnergyGuidance,
    TersoffSchedule,
)
from glass.potentials.tersoff import TersoffParameters, TorchTersoff

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


def _atoms_to_tensors(atoms, dtype=torch.float64):
    pos = torch.tensor(atoms.get_positions(), dtype=dtype)
    cell = torch.tensor(np.array(atoms.cell), dtype=dtype)
    species = torch.zeros(pos.shape[0], dtype=torch.long)
    return pos, cell, species


def test_guidance_forward():
    atoms = read(DATA_DIR / "Si_2.5_00.xyz")
    pos, cell, species = _atoms_to_tensors(atoms)

    guide = TersoffEnergyGuidance(clamp_norm=1e9)
    out = guide(pos, cell, species)

    assert out.shape == pos.shape
    assert torch.isfinite(out).all()


def test_gradient_direction():
    atoms = read(DATA_DIR / "Si_2.5_00.xyz")
    pos, cell, species = _atoms_to_tensors(atoms)

    guide = TersoffEnergyGuidance(clamp_norm=1e9)
    # The guidance vector is -grad E / N, so stepping along it should
    # decrease the energy for a small step size.
    params = {("Si", "Si", "Si"): TersoffParameters(**SI_PARAMS_KW)}
    potential = TorchTersoff(params)

    E0 = potential.energy(pos, cell, pbc=(True, True, True)).item()
    direction = guide(pos, cell, species).to(pos.dtype)

    step = 1e-3
    pos_new = pos + step * direction
    E1 = potential.energy(pos_new, cell, pbc=(True, True, True)).item()

    assert E1 < E0, (E0, E1)


def test_schedule_clamp():
    atoms = read(DATA_DIR / "Si_2.5_00.xyz")
    pos, cell, species = _atoms_to_tensors(atoms)

    # Collapse two atoms to produce a huge gradient.
    pos_bad = pos.clone()
    pos_bad[1] = pos_bad[0] + torch.tensor([1e-4, 0.0, 0.0], dtype=pos.dtype)

    clamp = 2.5
    guide = TersoffEnergyGuidance(clamp_norm=clamp)
    out = guide(pos_bad, cell, species)

    per_atom_norms = out.norm(dim=-1)
    assert torch.isfinite(out).all()
    # All per-atom norms must respect the clamp (small numerical slack).
    assert float(per_atom_norms.max()) <= clamp + 1e-6


def test_schedule_values():
    lam0, tmax, t_gate, k = 0.2, 1.0, 0.3, 50.0

    const = TersoffSchedule(schedule="constant", lambda_0=lam0, tmax=tmax)
    assert const(0.0) == pytest.approx(lam0)
    assert const(tmax / 2) == pytest.approx(lam0)
    assert const(tmax) == pytest.approx(lam0)

    lin = TersoffSchedule(schedule="linear", lambda_0=lam0, tmax=tmax)
    assert lin(0.0) == pytest.approx(lam0)
    assert lin(tmax / 2) == pytest.approx(lam0 * 0.5)
    assert lin(tmax) == pytest.approx(0.0)

    sig = TersoffSchedule(
        schedule="sigmoid", lambda_0=lam0, tmax=tmax, t_gate=t_gate, k=k
    )
    # Below the gate -> near lambda_0; above the gate -> near 0; at gate -> half.
    assert sig(0.0) == pytest.approx(lam0, rel=1e-3)
    assert sig(t_gate) == pytest.approx(lam0 / 2.0, rel=1e-3)
    assert sig(tmax) == pytest.approx(0.0, abs=1e-6)


def test_schedule_accepts_tensor_t():
    sched = TersoffSchedule(schedule="linear", lambda_0=0.1, tmax=1.0)
    lam_tensor = sched(torch.tensor([0.5]))
    lam_scalar = sched(0.5)
    assert lam_tensor == pytest.approx(lam_scalar)


def test_reject_multispecies_input():
    pos = torch.zeros(4, 3, dtype=torch.float64)
    cell = 10.0 * torch.eye(3, dtype=torch.float64)
    # Two distinct species -> must fail.
    species_onehot = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float32
    )
    guide = TersoffEnergyGuidance()
    with pytest.raises(ValueError):
        guide(pos, cell, species_onehot)
