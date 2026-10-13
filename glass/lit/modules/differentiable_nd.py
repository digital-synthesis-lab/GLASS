import math

import torch

from .differentiable_xrd import wrap_positions


neutron_scattering_lengths = {
    "H": -3.739,
    "D": 6.671,
    "C": 6.646,
    "N": 9.36,
    "O": 5.803,
    "Si": 4.1491,
    "Ge": 8.185,
    "Sb": 5.57,
    "Te": 5.80,
    "Zr": 7.16,
    "Ni": 10.3,
    "Cu": 7.718,
    "Al": 3.449,
}


def compute_b_vals(element_names, device, dtype):
    return torch.tensor(
        [neutron_scattering_lengths[el] for el in element_names],
        device=device,
        dtype=dtype,
    )  # (S,)


def differentiable_compute_nd(
    pos, species, q_vals, b_vals, occupancy=None, biso=0.0, rthres=0.0, epsilon=1e-8
):
    device = pos.device
    N, M = pos.shape[0], q_vals.shape[0]
    b_all = species @ b_vals  # (N,)

    if occupancy is None:
        occupancy = torch.ones(N, device=device, dtype=pos.dtype)

    # Pair indices
    i, j = torch.triu_indices(N, N, offset=1, device=device)
    rij = pos[i] - pos[j]
    dist = rij.norm(dim=-1).clamp(min=epsilon)

    if rthres > 0:
        mask = dist >= rthres
        i, j, dist = i[mask], j[mask], dist[mask]

    qr = dist[:, None] * q_vals[None, :]  # (P, M)
    sinc = torch.where(qr < 1e-4, torch.ones_like(qr), torch.sin(qr) / qr)

    occ_i = occupancy[i].unsqueeze(1)  # (P, 1)
    occ_j = occupancy[j].unsqueeze(1)  # (P, 1)
    b_i = b_all[i].unsqueeze(1)
    b_j = b_all[j].unsqueeze(1)

    pair_contrib = 2 * occ_i * occ_j * b_i * b_j * sinc
    I_q = pair_contrib.sum(dim=0)

    # Debye-Waller damps interference terms only (self-scattering undamped).
    if biso != 0.0:
        dw = torch.exp(-(q_vals**2) * biso / (8 * math.pi**2))
        I_q = I_q * dw

    # Self-scattering: sum_i (occ_i * b_i)^2
    I_q = I_q + ((occupancy * b_all) ** 2).sum()

    return I_q


class DifferentiableND(torch.nn.Module):
    def __init__(self, q_vals, element_names, biso=1.5):
        super().__init__()
        qmin, qmax, qstep = q_vals
        self.register_buffer("q_tensor", torch.arange(qmin, qmax, qstep))
        self.element_names = element_names
        self.biso = biso

    def forward(self, pos, species, cell=None):
        if cell is not None:
            pos = wrap_positions(pos, cell.to(pos.device, pos.dtype))
        q_vals = self.q_tensor.to(pos.device)
        b_vals = compute_b_vals(self.element_names, device=pos.device, dtype=pos.dtype)
        return differentiable_compute_nd(pos, species, q_vals, b_vals, biso=self.biso)
