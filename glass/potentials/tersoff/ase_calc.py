import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from .params import TersoffParameters
from .potential import TorchTersoff


class TorchTersoffCalculator(Calculator):
    """ASE Calculator backed by the PyTorch Tersoff implementation.

    Uses autograd to compute forces from the energy.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        parameters: dict,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._torch_calc = TorchTersoff(parameters, dtype=dtype)
        self.dtype = dtype
        self.device = torch.device(device)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        pos_np = atoms.get_positions()
        cell_np = np.array(atoms.cell)
        pbc = tuple(bool(p) for p in atoms.pbc)

        pos = torch.tensor(
            pos_np, dtype=self.dtype, device=self.device, requires_grad=True
        )
        cell = torch.tensor(cell_np, dtype=self.dtype, device=self.device)

        E = self._torch_calc.energy(pos, cell, pbc)
        (grad,) = torch.autograd.grad(E, pos)
        forces = -grad.detach().cpu().numpy()
        energy = float(E.detach().cpu().item())

        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
        }


def silicon_calculator(**kwargs) -> TorchTersoffCalculator:
    """Convenience: pre-parameterized Si Tersoff calculator (diamond tutorial params)."""
    si_params = {
        ("Si", "Si", "Si"): TersoffParameters(
            A=3264.7,
            B=95.373,
            lambda1=3.2394,
            lambda2=1.3258,
            lambda3=1.3258,
            beta=0.33675,
            gamma=1.00,
            m=3.00,
            n=22.956,
            c=4.8381,
            d=2.0417,
            h=0.0000,
            R=3.00,
            D=0.20,
        )
    }
    return TorchTersoffCalculator(si_params, **kwargs)
