"""EXAFS/XANES guidance with atom-wise and atom-averaged targets."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch
from ase.io import read
from pathlib import Path

from glass.diffusion import VarianceExplodingDiffuser
from glass.lit.modules import LitScoreNet, LitSpecNet
from glass.lit.modules.likelihood import LikelihoodScore
from glass.utils.atoms import atoms_to_device, compute_target_from_reference

DATA = Path(__file__).parent / "data" / "Si_2.5_00.xyz"
OUT_DIM = 12


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    score = LitScoreNet(num_species=1, num_convs=1, dim=16, ema_decay=0.99, learn_rate=1e-3)
    spec = LitSpecNet(num_species=1, num_convs=1, dim=16, out_dim=OUT_DIM)
    atoms = read(DATA)
    return score.ema_model.eval(), spec.ema_model.eval(), atoms


@pytest.mark.parametrize("average", [False, True])
def test_xas_likelihood_shapes_and_gradient(setup, average):
    score, spec, atoms = setup
    target = compute_target_from_reference(atoms, spec, "exafs", 5.0, "cpu")
    assert target.shape == (len(atoms), OUT_DIM)
    if average:
        target = target.mean(dim=0, keepdim=True)

    lik = LikelihoodScore(score, spec, target, rho=1.0,
                          diffuser=VarianceExplodingDiffuser(k=0.8),
                          guidance_type="exafs", cutoff=5.0, spec_average=average)
    species, pos, cell = atoms_to_device(atoms, "cpu")
    pos = pos + 0.1 * torch.randn_like(pos)
    grad, norm = lik(species, pos, cell, torch.full((len(pos), 1), 0.3), 5.0)

    assert grad.shape == pos.shape and torch.isfinite(grad).all()
    assert norm.shape == ((1, 1) if average else (len(atoms), 1))


def test_averaged_target_accepts_reference_of_another_size(setup):
    """The averaged mode only needs the spectrum, not a matching cell."""
    score, spec, atoms = setup
    small = atoms[:100]
    target = compute_target_from_reference(small, spec, "exafs", 5.0, "cpu").mean(0, keepdim=True)
    lik = LikelihoodScore(score, spec, target, rho=1.0,
                          diffuser=VarianceExplodingDiffuser(k=0.8),
                          guidance_type="exafs", cutoff=5.0, spec_average=True)
    species, pos, cell = atoms_to_device(atoms, "cpu")
    grad, _ = lik(species, pos, cell, torch.full((len(pos), 1), 0.3), 5.0)
    assert torch.isfinite(grad).all()


def test_experimental_xas_spectrum_is_used_as_is(tmp_path):
    import json

    from glass.lit.modules.guidance import load_experimental_data

    path = tmp_path / "exafs.json"
    json.dump({"y": list(range(OUT_DIM))}, open(path, "w"))
    y = load_experimental_data(path, "exafs", cutoff=5.0, bin_size=100, angle_bins=100,
                               qmin=1.0, qmax=20.0, qstep=0.1, device="cpu")
    assert y.shape == (1, OUT_DIM)
    assert torch.equal(y[0], torch.arange(OUT_DIM, dtype=torch.float32))
