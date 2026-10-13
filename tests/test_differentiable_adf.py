import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from ase.build import bulk

from glass.lit.modules.differentiable_adf import DifferentiableADF


def _peak_deg(adf_hist, bin_centers):
    return math.degrees(float(bin_centers[adf_hist.sum(dim=0).argmax()]))


def test_angle_is_measured_at_the_centre_atom():
    """A bent 3-atom cluster: atom 0 bonded to atoms 1 and 2 at 100 deg, atoms 1
    and 2 not bonded to each other. The only bond angle is 100 deg at atom 0;
    measuring at a neighbour instead gives (180 - 100) / 2 = 40 deg."""
    theta = math.radians(100.0)
    r = 1.5
    pos = torch.tensor(
        [
            [10.0, 10.0, 10.0],
            [10.0 + r, 10.0, 10.0],
            [10.0 + r * math.cos(theta), 10.0 + r * math.sin(theta), 10.0],
        ]
    )
    cell = 20.0 * torch.eye(3)
    species = torch.ones(3, 1)
    adf = DifferentiableADF(cutoff=1.8, angle_bins=181, sigma=0.02)
    bin_centers, adf_hist, _ = adf(pos, species, cell)
    assert abs(_peak_deg(adf_hist, bin_centers) - 100.0) < 1.5


def test_diamond_silicon_peaks_at_tetrahedral_angle():
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)
    cell = torch.tensor(atoms.get_cell()[:], dtype=torch.float32)
    species = torch.ones(len(atoms), 1)
    adf = DifferentiableADF(cutoff=2.6, angle_bins=181, sigma=0.02)
    bin_centers, adf_hist, _ = adf(pos, species, cell)
    tetrahedral = math.degrees(math.acos(-1.0 / 3.0))  # 109.47
    assert abs(_peak_deg(adf_hist, bin_centers) - tetrahedral) < 1.5
