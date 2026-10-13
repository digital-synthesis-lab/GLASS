import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from glass.diffusion import VarianceExplodingDiffuser
from glass.diffusion.sampling import (
    _apply_update,
    denoise_by_sde,
    short_contact_atoms,
)


def _zero_score(species, pos, cell, t, cutoff):
    return torch.zeros_like(pos)


def _min_periodic_distance(pos, cell):
    inv = torch.linalg.inv(cell)
    vec = pos[None] - pos[:, None]
    vec = vec - torch.round(vec @ inv) @ cell
    d = vec.norm(dim=-1) + torch.eye(len(pos)) * 1e6
    return float(d.min())


def _grid(n=3, a=2.5):
    g = torch.arange(n, dtype=torch.float32) * a
    pos = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    return pos, n * a * torch.eye(3)


def test_short_contact_atoms_uses_periodic_images():
    cell = 10.0 * torch.eye(3)
    pos = torch.tensor([[0.1, 0.0, 0.0], [9.8, 0.0, 0.0], [5.0, 5.0, 5.0]])
    mask = short_contact_atoms(pos, cell, min_dist=1.0)
    assert mask.tolist() == [True, True, False]


def test_apply_update_reverts_only_offending_atoms():
    pos, cell = _grid()
    disp = torch.zeros_like(pos)
    disp[0] = pos[1] - pos[0] - torch.tensor([0.0, 0.0, 0.3])  # 0.3 Å from atom 1
    disp[5] = torch.tensor([0.1, 0.0, 0.0])  # harmless move
    new = _apply_update(pos, disp, cell, min_dist=1.0, max_iters=3)
    assert torch.equal(new[0], pos[0])
    assert torch.allclose(new[5], pos[5] + disp[5])


def test_apply_update_without_min_dist_is_plain_sum():
    pos, cell = _grid()
    disp = 0.4 * torch.randn_like(pos)
    assert torch.equal(_apply_update(pos, disp, cell, None, 3), pos + disp)


def test_denoise_with_veto_keeps_contacts_above_threshold():
    pos, cell = _grid()
    species = torch.zeros(len(pos), 1)
    diffuser = VarianceExplodingDiffuser(k=0.8)
    ts = torch.linspace(0.6, 1e-3, 64)
    torch.manual_seed(0)
    _, final = denoise_by_sde(
        species, pos.clone(), cell, 5.0, _zero_score, None, ts, diffuser,
        min_dist=1.5,
    )
    assert _min_periodic_distance(final, cell) >= 1.5 - 1e-5
    # Without the veto the same noise produces shorter contacts.
    torch.manual_seed(0)
    _, free = denoise_by_sde(
        species, pos.clone(), cell, 5.0, _zero_score, None, ts, diffuser,
    )
    assert _min_periodic_distance(free, cell) < 1.5
