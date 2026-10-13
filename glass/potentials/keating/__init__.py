"""Keating bond-stretching + bond-bending potential.

Reference: P. N. Keating, Phys. Rev. 145, 637 (1966).

The Keating potential is a valence-force-field model with explicit bond topology:
    E = (3/16)(α/d²) Σ_{<ij>} (r²−d²)²  +  (3/8)(β/d²) Σ_{<jik>} (r_ij·r_ik + d²/3)²

Unlike neighbor-based potentials (e.g., Tersoff), Keating requires an explicit bond
list and is evaluated only over bonded pairs and angles. Used by Wooten-Winer-Weaire
(1985) and Barkema-Mousseau (2000) for CRN generation.

Default parameters for silicon: α=2.965 eV/Å², β=0.845 eV/Å², d=2.35 Å.
"""

from .ase_calc import KeatingCalculator, silicon_calculator
from .params import KeatingParameters, silicon_parameters
from .potential import TorchKeating

__all__ = [
    "KeatingParameters",
    "silicon_parameters",
    "TorchKeating",
    "KeatingCalculator",
    "silicon_calculator",
]
