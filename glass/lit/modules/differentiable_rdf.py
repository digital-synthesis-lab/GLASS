import torch
import numpy as np
import itertools
import math


class DifferentiableRDF(torch.nn.Module):
    def __init__(self, cutoff=[0.0, 8.0], bin_size=100, sigma=0.15):
        super().__init__()
        self.cutoff = cutoff
        self.bin_size = bin_size
        self.sigma = sigma
        if type(self.cutoff) is not list:
            bin_edges = torch.linspace(
                0, self.cutoff, self.bin_size + 1
            )  # , device=device)
        else:
            bin_edges = torch.linspace(
                self.cutoff[0], self.cutoff[1], self.bin_size + 1
            )  # , device=device)
        self.bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def forward(self, positions, types, cell):
        if isinstance(positions, np.ndarray):
            positions = torch.tensor(positions, dtype=torch.float32)
        if isinstance(types, np.ndarray):
            types = torch.tensor(types, dtype=torch.int64)
        if isinstance(cell, np.ndarray):
            cell = torch.tensor(cell, dtype=torch.float32)

        positions = positions.requires_grad_(True).to(cell.device)
        types = types.argmax(dim=-1)
        types = types.to(cell.device)
        cell = cell.to(cell.device)

        unique_types = torch.unique(types)
        type_pairs = list(
            itertools.combinations_with_replacement(unique_types.tolist(), 2)
        )

        rdfs = []
        for type_i, type_j in type_pairs:
            rdf_ij = self._partial_rdf_pbc(positions, types, type_i, type_j, cell)
            if type_i != type_j:
                rdf_ji = self._partial_rdf_pbc(positions, types, type_j, type_i, cell)
                rdf_avg = (rdf_ij + rdf_ji) / 2
                rdfs.append(rdf_avg)
            else:
                rdfs.append(rdf_ij)

        # added a wrapped cleaning up bad values 05/08
        # rdfs = torch.nan_to_num(torch.stack(rdfs), nan=0.0, posinf=0.0, neginf=0.0)
        # return self.bin_centers, rdfs, type_pairs
        return self.bin_centers, torch.stack(rdfs), type_pairs

    def _partial_rdf_pbc(self, positions, types, type_i, type_j, cell):
        device = positions.device
        idx_i = (types == type_i).nonzero(as_tuple=True)[0]
        idx_j = (types == type_j).nonzero(as_tuple=True)[0]
        if idx_i.numel() == 0 or idx_j.numel() == 0:
            return torch.zeros(self.bin_size, device=device)

        pos_i = positions[idx_i]
        pos_j = positions[idx_j]

        rij = pos_i[:, None, :] - pos_j[None, :, :]  # (Ni, Nj, 3)

        # Apply minimum image convention
        cell_inv = torch.inverse(cell)
        frac = rij @ cell_inv.T
        frac = frac - frac.round()
        rij_pbc = frac @ cell.T

        dists = torch.norm(rij_pbc, dim=-1)
        if type(self.cutoff) is not list:
            valid_mask = (dists > 1e-5) & (dists <= self.cutoff)
        else:
            valid_mask = (dists > 1e-5) & (dists <= self.cutoff[-1])
        dists = dists[valid_mask]

        if dists.numel() == 0:
            return torch.zeros(self.bin_size, device=device)

        if type(self.cutoff) is not list:
            bin_edges = torch.linspace(
                0, self.cutoff, self.bin_size + 1
            )  # , device=device)
        else:
            bin_edges = torch.linspace(
                self.cutoff[0], self.cutoff[1], self.bin_size + 1
            )  # , device=device)

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        delta_r = bin_edges[1] - bin_edges[0]

        # Smooth Gaussian kernel binning
        dists_exp = dists.unsqueeze(1)
        bins_exp = bin_centers.unsqueeze(0)
        weights = torch.exp(-0.5 * ((dists_exp - bins_exp) / self.sigma) ** 2)
        weights = weights.sum(dim=0)  # (bin_size,)
        # weights *= (delta_r / (math.sqrt(2*math.pi) * self.sigma)) # 10/23 ADDED THIS PART TO GET PROPER G(r)

        # Normalize
        Ni = len(idx_i)
        Nj = len(idx_j)
        vol = torch.abs(torch.det(cell))
        rho_j = Nj / vol  # number density of type j

        shell_volumes = 4 * torch.pi * bin_centers**2 * delta_r
        norm = Ni * rho_j * shell_volumes  # expected ideal gas counts

        g_r = weights / (norm + 1e-8)  # avoid division by zero
        return g_r
