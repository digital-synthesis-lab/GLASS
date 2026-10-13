import math
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.io import read

from glass.metrics.structural import compute_adf

DATA_DIR = Path(__file__).parent / "data"
TETRAHEDRAL_RAD = math.acos(-1.0 / 3.0)  # 109.47 deg


def test_adf_of_diamond_silicon_peaks_at_tetrahedral_angle():
    atoms = bulk("Si", "diamond", a=5.43) * (3, 3, 3)
    adf = compute_adf(atoms, cutoff=2.8)
    peak = float(adf.angles[np.argmax(adf.adf)])
    assert peak == pytest.approx(TETRAHEDRAL_RAD, abs=math.radians(2.0))


def test_adf_is_invariant_under_atom_relabelling():
    """The ADF is a property of the structure, not of the atom numbering: every
    bond angle around every centre must be counted, including angles whose
    bonds cross the periodic boundary."""
    atoms = read(DATA_DIR / "Si_2.5_00.xyz")
    perm = np.random.default_rng(0).permutation(len(atoms))
    ref = compute_adf(atoms, cutoff=2.8)
    shuffled = compute_adf(atoms[perm], cutoff=2.8)
    assert np.allclose(ref.adf, shuffled.adf, atol=1e-10)


def test_dominant_angle_is_tallest_peak_in_amorphous_si():
    """a-Si has a small 60 degree shoulder before the tetrahedral peak; the
    dominant angle must be the tetrahedral one."""
    atoms = read(Path(__file__).parent / "data" / "Si_2.5_00.xyz")
    adf = compute_adf(atoms)
    assert 95.0 < adf.dominant_angle_degree < 120.0
