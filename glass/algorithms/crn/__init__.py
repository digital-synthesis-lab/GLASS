"""Continuous Random Network (CRN) generation via WWW algorithm.

This module implements the Barkema-Mousseau (BM2000) improved Wooten-Winer-Weaire
(WWW) algorithm for generating 4-coordinated continuous random networks of silicon
atoms under the Keating potential.

The WWW algorithm uses bond transposition (bond-exchange moves) combined with
simulated annealing and periodic T=0 quenching to generate amorphous structures
with low energy and no crystalline memory.

References:
- Wooten, Winer, Weaire, Phys. Rev. Lett. 54, 1392 (1985)
- Barkema, Mousseau, Phys. Rev. B 62, 4985 (2000)

Main entry point:
- `generate_crn()`: Generate a CRN structure using the full BM algorithm

Key classes:
- `Network`: 4-coordinated network with explicit bond topology
- `WWWStats`: Statistics from CRN generation (energy, acceptance rates)

Utilities:
- `random_initial_network()`: Create initial structure via loop expansion
"""

from glass.algorithms.crn.initialize import random_initial_network
from glass.algorithms.crn.network import (
    Network,
    bond_length_for_density,
    cubic_cell_for_si,
)
from glass.algorithms.crn.www import WWWStats, generate_crn

__all__ = [
    "generate_crn",
    "random_initial_network",
    "Network",
    "WWWStats",
    "bond_length_for_density",
    "cubic_cell_for_si",
]
