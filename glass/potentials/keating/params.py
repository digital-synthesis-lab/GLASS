from dataclasses import dataclass


@dataclass
class KeatingParameters:
    """Parameters for the Keating bond-stretching + bond-bending potential.

    Reference: P. N. Keating, Phys. Rev. 145, 637 (1966).
    Used by Wooten-Winer-Weaire (1985) and Barkema-Mousseau (2000).

    The Keating energy is:
        E = (3/16)(α/d²) Σ_{<ij>} (r_ij · r_ij − d²)²
          + (3/8)(β/d²) Σ_{<jik>} (r_ij · r_ik + d²/3)²

    where <ij> denotes bonded pairs and <jik> denotes bonded angle triples.
    """

    alpha: float = 2.965  # eV/Å² — bond-stretch force constant
    beta: float = 0.845145  # eV/Å² — bond-bend force constant (0.285 * alpha)
    d: float = 2.35  # Å — equilibrium bond length (diamond Si)


def silicon_parameters() -> KeatingParameters:
    """Returns Keating parameters for silicon.

    These are the default parameters from Barkema-Mousseau (2000) eq. (2),
    calibrated to diamond silicon.
    """
    return KeatingParameters()
