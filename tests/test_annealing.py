import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import torch
from ase.build import bulk

from glass.diffusion.annealing import simulated_anneal, make_anneal_fn
from glass.lit.modules.tersoff_guidance import TersoffEnergyGuidance
from glass.potentials.tersoff import TersoffParameters, TorchTersoff

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


def _tensors(atoms, dtype=torch.float64):
    pos = torch.tensor(atoms.get_positions(), dtype=dtype)
    cell = torch.tensor(np.array(atoms.cell), dtype=dtype)
    species = torch.zeros(pos.shape[0], dtype=torch.long)
    return pos, cell, species


def _energy(pos, cell):
    params = {("Si", "Si", "Si"): TersoffParameters(**SI_PARAMS_KW)}
    return float(
        TorchTersoff(params).energy(pos, cell, pbc=(True, True, True)).item()
    )


def test_sa_decreases_energy_on_perturbed_crystal():
    torch.manual_seed(0)
    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    pos, cell, species = _tensors(atoms)

    # Kick the crystal a bit so there's headroom to relax.
    pos_kick = pos + 0.15 * torch.randn_like(pos)
    E0 = _energy(pos_kick, cell)

    guide = TersoffEnergyGuidance(clamp_norm=1.0)
    pos_new = simulated_anneal(
        pos_kick, cell, species, guide,
        n_steps=200, T0=1e-4, T_end=1e-8, lr=5e-3, lr_clamp=0.05,
    )
    E1 = _energy(pos_new, cell)
    assert E1 < E0, (E0, E1)
    # Expect a meaningful drop for a small kick.
    assert (E0 - E1) / abs(E0) > 0.01, (E0, E1)


def test_sa_no_op_with_zero_temperature_and_zero_lr():
    torch.manual_seed(1)
    atoms = bulk("Si", "diamond", a=5.43)
    pos, cell, species = _tensors(atoms)
    guide = TersoffEnergyGuidance(clamp_norm=1.0)
    pos_new = simulated_anneal(
        pos, cell, species, guide,
        n_steps=10, T0=1e-30, T_end=1e-30, lr=0.0, lr_clamp=0.01, wrap=False,
    )
    assert torch.allclose(pos, pos_new, atol=1e-8)


def test_sa_step_clamp_bounds_motion():
    torch.manual_seed(2)
    atoms = bulk("Si", "diamond", a=5.43)
    pos, cell, species = _tensors(atoms)

    # Use a huge effective lr by setting the guidance clamp extremely loose.
    # The SA-level lr_clamp should still pin motion.
    guide = TersoffEnergyGuidance(clamp_norm=1e6)
    lr_clamp = 0.02
    pos_new = simulated_anneal(
        pos, cell, species, guide,
        n_steps=5, T0=1e-30, T_end=1e-30, lr=1e6, lr_clamp=lr_clamp, wrap=False,
    )
    delta = (pos_new - pos).norm(dim=-1)
    # n_steps steps each bounded by lr_clamp -> total bounded by n_steps*lr_clamp.
    assert float(delta.max()) <= 5 * lr_clamp + 1e-6


def test_sa_zero_steps_is_identity():
    atoms = bulk("Si", "diamond", a=5.43)
    pos, cell, species = _tensors(atoms)
    guide = TersoffEnergyGuidance()
    pos_new = simulated_anneal(pos, cell, species, guide, n_steps=0)
    assert torch.equal(pos, pos_new)


def test_make_anneal_fn_signature():
    atoms = bulk("Si", "diamond", a=5.43)
    pos, cell, species = _tensors(atoms)
    guide = TersoffEnergyGuidance()
    fn = make_anneal_fn(guide, n_steps=3, T0=1e-6, T_end=1e-8, lr=1e-4)
    out = fn(pos, cell, species)
    assert out.shape == pos.shape
    assert torch.isfinite(out).all()
