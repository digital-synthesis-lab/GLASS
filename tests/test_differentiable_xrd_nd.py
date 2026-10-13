import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch

from glass.lit.modules.differentiable_nd import (
    DifferentiableND,
    compute_b_vals,
    differentiable_compute_nd,
)
from glass.lit.modules.differentiable_xrd import (
    DifferentiableXRD,
    compute_form_factors,
    differentiable_compute_iq,
    xrd_form_factors,
)

Q = torch.linspace(0.5, 12.0, 24, dtype=torch.float64)
R = 3.0  # Å, separation of the two-atom cluster


def _two_atoms():
    """An O atom and a Si atom R apart; one-hot columns in Z order (O, Si)."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [R, 0.0, 0.0]], dtype=torch.float64)
    species = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    return pos, species


def _dw(biso):
    return torch.exp(-(Q**2) * biso / (8 * math.pi**2))


def _sinc():
    return torch.sin(Q * R) / (Q * R)


def test_form_factor_uses_waasmaier_kirfel_argument():
    """f(q) = sum_i a_i exp(-b_i (q / 4 pi)^2) + c, with q in Å^-1."""
    f = compute_form_factors(Q, ["O", "Si"])
    for row, el in enumerate(["O", "Si"]):
        p = xrd_form_factors[el]
        s2 = (Q / (4 * math.pi)) ** 2
        expected = sum(a * torch.exp(-b * s2) for a, b in zip(p["a"], p["b"])) + p["c"]
        assert torch.allclose(f[row], expected)


@pytest.mark.parametrize("biso", [0.0, 1.5])
def test_xrd_debye_sum_matches_closed_form(biso):
    """I(q) = f_O^2 + f_Si^2 + 2 f_O f_Si sinc(qR) DW(q): full self term, and
    the Debye-Waller factor damps the interference term only."""
    pos, species = _two_atoms()
    ff = compute_form_factors(Q, ["O", "Si"])
    iq = differentiable_compute_iq(pos, species, Q, ff, biso=biso)
    f_o, f_si = ff[0], ff[1]
    expected = f_o**2 + f_si**2 + 2 * f_o * f_si * _sinc() * _dw(biso)
    assert torch.allclose(iq, expected, rtol=1e-10)


@pytest.mark.parametrize("biso", [0.0, 1.5])
def test_nd_debye_sum_matches_closed_form(biso):
    """I(q) = b_O^2 + b_Si^2 + 2 b_O b_Si sinc(qR) DW(q)."""
    pos, species = _two_atoms()
    b = compute_b_vals(["O", "Si"], device="cpu", dtype=torch.float64)
    iq = differentiable_compute_nd(pos, species, Q, b, biso=biso)
    expected = b[0] ** 2 + b[1] ** 2 + 2 * b[0] * b[1] * _sinc() * _dw(biso)
    assert torch.allclose(iq, expected, rtol=1e-10)


@pytest.mark.parametrize("guidance_type", ["xrd", "nd"])
def test_guidance_element_tables_follow_one_hot_order(guidance_type):
    """The one-hot species columns are in atomic-number order, so the XRD/ND
    element tables must be too, whatever order the user lists them in."""
    from ase import Atoms

    from glass.lit.modules.guidance import create_guidance_model
    from glass.utils.atoms import initialize_atoms

    z_list, _, _, _ = initialize_atoms(Atoms("SiO2", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
    assert z_list == [8, 14]  # columns: O, Si
    model = create_guidance_model(
        guidance_type, device="cpu", cutoff=5.0, element_names=["Si", "O"]
    )
    assert model.element_names == ["O", "Si"]


@pytest.mark.parametrize("cls", [DifferentiableXRD, DifferentiableND])
def test_intensity_ignores_lattice_translations_when_cell_given(cls):
    """Moving an atom by a lattice vector must not change I(q) once the cell
    is passed, and the gradient must still flow to the positions."""
    torch.manual_seed(0)
    cell = 6.0 * torch.eye(3, dtype=torch.float32)
    pos = 6.0 * torch.rand(8, 3, dtype=torch.float32)
    species = torch.ones(8, 1, dtype=torch.float32)
    model = cls((1.0, 10.0, 0.5), ["Si"])

    shifted = pos.clone()
    shifted[3] += cell[0] - 2 * cell[2]
    ref = model(pos, species, cell)
    assert torch.allclose(model(shifted, species, cell), ref, rtol=1e-4)
    # Without the cell the open-boundary sum sees a different geometry.
    assert not torch.allclose(model(shifted, species), ref)

    shifted.requires_grad_(True)
    model(shifted, species, cell).sum().backward()
    assert torch.isfinite(shifted.grad).all() and shifted.grad.abs().sum() > 0
