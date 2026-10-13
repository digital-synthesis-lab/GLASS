from dataclasses import dataclass


@dataclass
class TersoffParameters:
    """Parameters for a Tersoff three-body interaction.

    Field order matches LAMMPS format and ase.calculators.tersoff.TersoffParameters.
    """

    m: float
    gamma: float
    lambda3: float
    c: float
    d: float
    h: float
    n: float
    beta: float
    lambda2: float
    B: float
    R: float
    D: float
    lambda1: float
    A: float
