import math

import torch


xrd_form_factors = {
    "C": {
        "a": [2.657506, 1.078079, 1.490909, -4.24107, 0.713791],
        "b": [14.780758, 0.776775, 42.086842, -0.000294, 0.239535],
        "c": 4.297983,
    },
    "O": {
        "a": [2.960427, 2.508818, 0.637853, 0.722838, 1.142756],
        "b": [14.182259, 5.936858, 0.112726, 34.958481, 0.39024],
        "c": 0.027014,
    },
    "Si": {
        "a": [5.275329, 3.191038, 1.511514, 1.356849, 2.519114],
        "b": [2.631338, 33.730728, 0.081119, 86.288643, 1.170087],
        "c": 0.145073,
    },
    "Ge": {
        "a": [16.540613, 1.5679, 3.727829, 3.345098, 6.785079],
        "b": [2.866618, 0.012198, 13.432163, 58.866047, 0.210974],
        "c": 0.018726,
    },
    "Sb": {
        "a": [5.394956, 6.54957, 19.650681, 1.82782, 17.867832],
        "b": [33.326523, 0.030974, 5.564929, 87.130966, 0.523992],
        "c": -0.290506,
    },
    "Te": {
        "a": [6.660302, 6.940756, 19.847015, 1.557175, 17.802427],
        "b": [33.031654, 0.02575, 5.065547, 84.101616, 0.48766],
        "c": -0.806668,
    },
    "Zr": {
        "a": [17.859772, 10.911038, 5.821115, 3.512513, 0.746965],
        "b": [1.310692, 12.319285, 0.104353, 91.777542, 0.104353],
        "c": 1.124859,
    },
    "Ni": {
        "a": [13.521865, 6.947285, 3.866028, 2.1359, 4.284731],
        "b": [4.077277, 0.286763, 14.622634, 71.96608, 0.004437],
        "c": -2.762697,
    },
    "Cu": {
        "a": [14.014192, 4.784577, 5.056806, 1.457971, 6.932996],
        "b": [3.73828, 0.003744, 13.034982, 72.554794, 0.265666],
        "c": -3.254477,
    },
    "Al": {
        "a": [4.730796, 2.313951, 1.54198, 1.117564, 3.154754],
        "b": [3.628931, 43.051167, 0.09596, 108.932388, 1.555918],
        "c": 0.139509,
    },
}


def wrap_positions(pos, cell):
    """Wrap Cartesian positions into the cell.

    The Debye sum is evaluated on the atoms inside the cell without periodic
    images, so the result depends on where atoms sit relative to the cell
    boundaries. Wrapping makes it independent of how far atoms drifted during
    sampling. The floor has zero gradient, so d(wrapped)/d(pos) is the identity.
    """
    frac = pos @ torch.linalg.inv(cell)
    frac = frac - torch.floor(frac.detach())
    return frac @ cell


def compute_form_factors(q_vals, element_names):
    # Waasmaier & Kirfel (1995): f(q) = sum_i a_i exp(-b_i s^2) + c with
    # s = sin(theta)/lambda = q / (4 pi), q in Å^-1.
    s2 = (q_vals / (4.0 * math.pi)).pow(2).view(1, -1)
    f_all = []
    for el in element_names:
        p = xrd_form_factors[el]
        a = torch.tensor(p["a"], dtype=q_vals.dtype, device=q_vals.device).view(-1, 1)
        b = torch.tensor(p["b"], dtype=q_vals.dtype, device=q_vals.device).view(-1, 1)
        c = torch.tensor(p["c"], dtype=q_vals.dtype, device=q_vals.device)
        f = (a * torch.exp(-b * s2)).sum(dim=0) + c
        f_all.append(f)
    return torch.stack(f_all, dim=0)  # (S, M)


def differentiable_compute_iq(
    pos, species, q_vals, form_factors, biso=1.5, rthres=0.0, epsilon=1e-8
):
    device = pos.device
    N, M = pos.shape[0], q_vals.shape[0]

    # f_q_all: (N, M) ← weighted sum over species
    f_q_all = species @ form_factors  # (N, M)

    # Pair indices (P, 2)
    i, j = torch.triu_indices(N, N, offset=1, device=device)
    rij = pos[i] - pos[j]
    dist = rij.norm(dim=-1).clamp(min=epsilon)

    if rthres > 0:
        mask = dist >= rthres
        i, j, dist = i[mask], j[mask], dist[mask]

    # Compute q * r and sinc
    qr = dist[:, None] * q_vals[None, :]  # (P, M)
    sinc = torch.where(qr < 1e-4, torch.ones_like(qr), torch.sin(qr) / qr)  # (P, M)

    # Pairwise contributions: (P, M)
    f_i = f_q_all[i]  # (P, M)
    f_j = f_q_all[j]  # (P, M)
    pair_contrib = 2 * f_i * f_j * sinc  # (P, M)
    I_q = pair_contrib.sum(dim=0)  # (M,)

    # Debye-Waller damps the interference (pair) terms only; the i == j
    # self-scattering term sinc(0) = 1 is not damped.
    if biso != 0.0:
        dw = torch.exp(-(q_vals**2) * biso / (8 * math.pi**2))  # (M,)
        I_q = I_q * dw

    # Self-terms: sum_i f_i^2
    I_q = I_q + (f_q_all**2).sum(dim=0)

    return I_q


class DifferentiableXRD(torch.nn.Module):
    def __init__(self, q_vals, element_names, biso=1.5):
        super().__init__()
        qmin, qmax, qstep = q_vals
        # q_tensor = torch.linspace(qmin, qmax, int((qmax - qmin) / qstep))
        q_tensor = torch.arange(qmin, qmax, qstep)  # .unsqueeze(-1)
        self.register_buffer("q_tensor", q_tensor)
        self.element_names = element_names
        self.biso = biso

    def forward(self, pos, species, cell=None):
        if cell is not None:
            pos = wrap_positions(pos, cell.to(pos.device, pos.dtype))
        q_vals = self.q_tensor.to(pos.device)
        form_factors = compute_form_factors(q_vals, self.element_names)
        assert pos.device == species.device == form_factors.device

        return differentiable_compute_iq(
            pos, species, q_vals, form_factors, biso=self.biso
        )
