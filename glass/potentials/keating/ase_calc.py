import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from .params import KeatingParameters, silicon_parameters
from .potential import TorchKeating


class KeatingCalculator(Calculator):
    """ASE Calculator backed by the PyTorch Keating implementation.

    Unlike Tersoff, Keating requires an explicit bond topology.
    The bond list must be provided at construction and is assumed fixed.

    Uses autograd to compute forces from the energy.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        parameters: KeatingParameters,
        bonds: np.ndarray,
        neigh: np.ndarray,
        degree: np.ndarray,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize Keating calculator.

        Args:
            parameters: Keating parameters (alpha, beta, d)
            bonds: (M, 2) bond list (undirected, 0-indexed)
            neigh: (N, 4) neighbor indices (-1 padded)
            degree: (N,) number of neighbors per atom
            dtype: torch dtype for computation
            device: torch device ('cpu' or 'cuda')
            **kwargs: passed to Calculator base class
        """
        super().__init__(**kwargs)
        self._torch_calc = TorchKeating(parameters, dtype=dtype)
        self.dtype = dtype
        self.device = torch.device(device)

        # Store topology (assumed fixed)
        self.bonds = torch.tensor(bonds, dtype=torch.int64, device=self.device)
        self.neigh = torch.tensor(neigh, dtype=torch.int64, device=self.device)
        self.degree = torch.tensor(degree, dtype=torch.int64, device=self.device)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        pos_np = atoms.get_positions()
        cell_np = np.array(atoms.cell)
        pbc = all(atoms.pbc)  # Keating requires orthorhombic, so check uniformity

        pos = torch.tensor(
            pos_np, dtype=self.dtype, device=self.device, requires_grad=True
        )
        cell = torch.tensor(cell_np, dtype=self.dtype, device=self.device)

        E = self._torch_calc.energy(pos, cell, self.bonds, self.neigh, self.degree, pbc)
        (grad,) = torch.autograd.grad(E, pos)
        forces = -grad.detach().cpu().numpy()
        energy = float(E.detach().cpu().item())

        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
        }


def silicon_calculator(
    bonds: np.ndarray, neigh: np.ndarray, degree: np.ndarray, **kwargs
) -> KeatingCalculator:
    """Convenience: pre-parameterized Si Keating calculator.

    Args:
        bonds: (M, 2) bond list
        neigh: (N, 4) neighbor indices
        degree: (N,) coordination numbers
        **kwargs: passed to KeatingCalculator

    Returns:
        KeatingCalculator with default Si parameters
    """
    params = silicon_parameters()
    return KeatingCalculator(params, bonds, neigh, degree, **kwargs)
