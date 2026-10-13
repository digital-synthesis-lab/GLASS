import torch
import math
from collections import defaultdict


class DifferentiableADF(torch.nn.Module):
    def __init__(
        self,
        cutoff=8.0,
        angle_bins=50,
        angle_range=(0, math.pi),
        sigma=0.1,
        normalize=False,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.angle_bins = angle_bins
        self.angle_range = angle_range
        self.sigma = sigma
        self.normalize = normalize
        bin_centers = torch.linspace(angle_range[0], angle_range[1], angle_bins)
        self.register_buffer("bin_centers", bin_centers)

    def forward(self, pos, species, cell):
        device = pos.device
        types = species.argmax(dim=-1)
        N = pos.shape[0]

        inv_cell = torch.inverse(cell)
        frac = pos @ inv_cell.T
        diff = frac[:, None, :] - frac[None, :, :]
        diff -= torch.round(diff)
        vecs = diff @ cell
        dists = vecs.norm(dim=-1)
        mask = (dists < self.cutoff) & (~torch.eye(N, dtype=bool, device=device))

        # triplet_mask[c, a, b] is True when both a and b are bonded to the
        # centre c (and a != b), so each enumerated triplet is a real
        # bonded-neighbour pair around c.
        all_atoms = torch.arange(N, device=device)
        triplet_mask = mask[all_atoms[:, None], :] & mask[all_atoms[:, None], :].transpose(1, 2)
        triplet_mask &= ~torch.eye(N, dtype=bool, device=device)[None, :, :]

        c_idx, j_idx, k_idx = torch.nonzero(triplet_mask, as_tuple=True)

        # Both bond vectors must emanate from the centre c; using
        # vecs[j, k] here (neighbour-to-neighbour) measures the angle at the
        # wrong vertex and over a possibly non-bonded pair.
        vec_cj = vecs[j_idx, c_idx]
        vec_ck = vecs[k_idx, c_idx]

        angles = self._compute_angle_batch(vec_cj, vec_ck)

        t_c = types[c_idx]
        t_j = types[j_idx]
        t_k = types[k_idx]
        min_t = torch.minimum(t_j, t_k)
        max_t = torch.maximum(t_j, t_k)
        triplet_types = torch.stack([min_t, t_c, max_t], dim=1)

        triplet_keys, inverse = torch.unique(triplet_types, dim=0, return_inverse=True)

        diff = angles[:, None] - self.bin_centers[None, :]
        gauss = torch.exp(-0.5 * (diff / self.sigma) ** 2)

        adf_hist = torch.zeros((triplet_keys.shape[0], self.angle_bins), device=device)
        adf_hist.index_add_(0, inverse, gauss)

        if self.normalize:
            adf_hist = adf_hist / (adf_hist.sum() + 1e-12)

        return self.bin_centers, adf_hist, triplet_keys.tolist()

    def _compute_angle_batch(self, v1, v2):
        dot = (v1 * v2).sum(dim=-1)
        norm = v1.norm(dim=-1) * v2.norm(dim=-1)
        cos = dot / (norm + 1e-12)
        # d(acos)/dx diverges at +-1; stay slightly inside so collinear or
        # overlapping triplets cannot produce infinite gradients.
        return torch.acos(torch.clamp(cos, -1.0 + 1e-6, 1.0 - 1e-6))
