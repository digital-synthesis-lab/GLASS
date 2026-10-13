"""PyTorch implementation of the Keating potential.

Reference: P. N. Keating, Phys. Rev. 145, 637 (1966).
Used by Wooten-Winer-Weaire (1985) and Barkema-Mousseau (2000).

E = (3/16)(α/d²) Σ_{<ij>}      (r_ij · r_ij − d²)²
  + (3/8) (β/d²) Σ_{<jik>}     (r_ij · r_ik + d²/3)²

Unlike Tersoff (neighbor-based), Keating requires an explicit bond list.
"""

from __future__ import annotations

import torch

from .params import KeatingParameters


class TorchKeating:
    """PyTorch implementation of Keating bond-stretching + bond-bending potential.

    Single-species Si only. Requires explicit bond topology (not computed from cutoff).
    PBC is orthorhombic via minimum-image displacement.
    """

    def __init__(
        self, parameters: KeatingParameters, dtype: torch.dtype = torch.float64
    ):
        self.params = parameters
        self.dtype = dtype

    @staticmethod
    def _mic_disp_torch(
        rj: torch.Tensor, ri: torch.Tensor, cell: torch.Tensor
    ) -> torch.Tensor:
        """Minimum-image displacement r_j − r_i in an orthorhombic cell.

        Args:
            rj: (3,) or (..., 3) position of atom j
            ri: (3,) or (..., 3) position of atom i
            cell: (3, 3) cell matrix (must be orthorhombic/diagonal)

        Returns:
            (..., 3) minimum-image displacement
        """
        # Extract diagonal elements for orthorhombic cell
        box_diag = torch.diagonal(cell)  # (3,)
        dr = rj - ri
        # Wrap to minimum image
        dr = dr - box_diag * torch.round(dr / box_diag)
        return dr

    def energy(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        bonds: torch.Tensor,
        neigh: torch.Tensor,
        degree: torch.Tensor,
        pbc: bool = True,
    ) -> torch.Tensor:
        """Compute Keating energy.

        Args:
            positions: (N, 3) atom positions
            cell: (3, 3) cell matrix (orthorhombic only)
            bonds: (M, 2) bond list (undirected, each pair once)
            neigh: (N, 4) neighbor indices (-1 padded)
            degree: (N,) number of neighbors per atom
            pbc: whether to apply periodic boundary conditions

        Returns:
            Scalar energy tensor
        """
        p = self.params
        device = positions.device
        dtype = self.dtype

        # Convert to torch tensors if needed
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=dtype, device=device)
        if not isinstance(cell, torch.Tensor):
            cell = torch.tensor(cell, dtype=dtype, device=device)
        if not isinstance(bonds, torch.Tensor):
            bonds = torch.tensor(bonds, dtype=torch.int64, device=device)
        if not isinstance(neigh, torch.Tensor):
            neigh = torch.tensor(neigh, dtype=torch.int64, device=device)
        if not isinstance(degree, torch.Tensor):
            degree = torch.tensor(degree, dtype=torch.int64, device=device)

        # Verify orthorhombic cell
        off_diag = cell - torch.diag(torch.diagonal(cell))
        if not torch.allclose(off_diag, torch.zeros_like(off_diag), atol=1e-10):
            raise ValueError("Only orthorhombic cells supported")

        inv_d2 = 1.0 / (p.d * p.d)
        cs = (3.0 / 16.0) * p.alpha * inv_d2  # bond term coefficient
        cb = (3.0 / 8.0) * p.beta * inv_d2  # angle term coefficient
        third_d2 = (p.d * p.d) / 3.0

        # Bond energy term
        e_bond = torch.tensor(0.0, dtype=dtype, device=device)
        if bonds.shape[0] > 0:
            i_idx = bonds[:, 0]
            j_idx = bonds[:, 1]
            ri = positions[i_idx]  # (M, 3)
            rj = positions[j_idx]  # (M, 3)

            if pbc:
                dr = self._mic_disp_torch(rj, ri, cell)  # (M, 3)
            else:
                dr = rj - ri

            r2 = torch.sum(dr * dr, dim=1)  # (M,)
            diff = r2 - p.d * p.d
            e_bond = cs * torch.sum(diff * diff)

        # Angle energy term
        e_ang = torch.tensor(0.0, dtype=dtype, device=device)
        n_atoms = positions.shape[0]

        for i in range(n_atoms):
            di = int(degree[i].item())
            if di < 2:
                continue

            # Get neighbors of atom i
            neighs_i = neigh[i, :di]  # (di,)

            # Loop over all pairs of neighbors
            for a in range(di):
                j = int(neighs_i[a].item())
                for b in range(a + 1, di):
                    k = int(neighs_i[b].item())

                    # Compute r_ij and r_ik
                    if pbc:
                        r_ij = self._mic_disp_torch(
                            positions[j], positions[i], cell
                        )
                        r_ik = self._mic_disp_torch(
                            positions[k], positions[i], cell
                        )
                    else:
                        r_ij = positions[j] - positions[i]
                        r_ik = positions[k] - positions[i]

                    dot = torch.dot(r_ij, r_ik)
                    s = dot + third_d2
                    e_ang = e_ang + cb * s * s

        return e_bond + e_ang

    def energy_and_forces_autograd(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        bonds: torch.Tensor,
        neigh: torch.Tensor,
        degree: torch.Tensor,
        pbc: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energy and forces via autograd.

        Args:
            positions: (N, 3) atom positions
            cell: (3, 3) cell matrix
            bonds: (M, 2) bond list
            neigh: (N, 4) neighbor indices
            degree: (N,) number of neighbors
            pbc: whether to apply PBC

        Returns:
            (energy, forces) where forces is (N, 3)
        """
        pos = positions.clone().detach().requires_grad_(True)
        energy = self.energy(pos, cell, bonds, neigh, degree, pbc)
        (forces,) = torch.autograd.grad(energy, pos, create_graph=False)
        return energy.detach(), -forces.detach()
