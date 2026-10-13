import math

import torch

from .neighbors import build_neighbors
from .params import TersoffParameters

# Matches the guards in ase.calculators.tersoff
_MAX_EXP_ARG = 69.0776
_MIN_EXP_ARG = -69.0776


class TorchTersoff:
    """PyTorch reimplementation of the LAMMPS-style Tersoff potential.

    Single-species only. API mirrors ase.calculators.tersoff.Tersoff:
    parameters keyed by (elem, elem, elem) -> TersoffParameters.
    """

    def __init__(self, parameters: dict, dtype: torch.dtype = torch.float64):
        if len(parameters) != 1:
            raise NotImplementedError("Only single-species parameter sets supported.")
        self.key = next(iter(parameters.keys()))
        if not (self.key[0] == self.key[1] == self.key[2]):
            raise NotImplementedError("Only homogeneous (A,A,A) keys supported.")
        self.params: TersoffParameters = parameters[self.key]
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------
    @staticmethod
    def _fc(r: torch.Tensor, R: float, D: float) -> torch.Tensor:
        out = torch.where(
            r < R - D,
            torch.ones_like(r),
            0.5 * (1.0 - torch.sin(math.pi * (r - R) / (2.0 * D))),
        )
        out = torch.where(r > R + D, torch.zeros_like(r), out)
        return out

    @staticmethod
    def _fc_d(r: torch.Tensor, R: float, D: float) -> torch.Tensor:
        inside = (r > R - D) & (r < R + D)
        val = -0.25 * math.pi / D * torch.cos(math.pi * (r - R) / (2.0 * D))
        return torch.where(inside, val, torch.zeros_like(r))

    @staticmethod
    def _gijk(costheta: torch.Tensor, p: TersoffParameters) -> torch.Tensor:
        c2 = p.c * p.c
        d2 = p.d * p.d
        hcth = p.h - costheta
        return p.gamma * (1.0 + c2 / d2 - c2 / (d2 + hcth * hcth))

    @staticmethod
    def _gijk_d(costheta: torch.Tensor, p: TersoffParameters) -> torch.Tensor:
        c2 = p.c * p.c
        d2 = p.d * p.d
        hcth = p.h - costheta
        return (-2.0 * p.gamma * c2 * hcth) / (d2 + hcth * hcth) ** 2

    @staticmethod
    def _safe_exp(arg: torch.Tensor) -> torch.Tensor:
        clamped = torch.clamp(arg, _MIN_EXP_ARG, _MAX_EXP_ARG)
        out = torch.exp(clamped)
        out = torch.where(arg > _MAX_EXP_ARG, torch.full_like(out, 1.0e30), out)
        out = torch.where(arg < _MIN_EXP_ARG, torch.zeros_like(out), out)
        return out

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------
    def _energy_from_pairs(
        self,
        positions: torch.Tensor,
        i_idx: torch.Tensor,
        j_idx: torch.Tensor,
        shift_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Total Tersoff energy from an enumerated directed neighbour list.

        Implements the segmented triple-sum over shared-source pairs
        without ever allocating a (P, P) mask. For each source atom i,
        zeta_{ij} = sum_{k ∈ nbrs(i), k != j} f_c(r_ik) · g(θ_{ijk}) ·
        exp(λ₃·(r_ij − r_ik)^m). We pack per-atom neighbour slices into a
        (N_src, n_max, 3) padded tensor and sum along a masked k axis, so
        peak memory is O(N · n_max²) instead of O(P²).
        """
        p = self.params
        device = positions.device
        dtype = positions.dtype
        P = int(i_idx.shape[0])

        if P == 0:
            # Keep the autograd connection to `positions` so that callers
            # taking ∇E with respect to positions don't trip
            # "element 0 of tensors does not require grad" when the
            # neighbour list happens to be empty (e.g. low-density
            # configurations at high noise).
            return positions.sum() * 0.0

        # Sort pairs by source atom i so pairs with the same i are
        # contiguous — this lets us build the padded (N_src, n_max)
        # layout with a single gather.
        sort_idx = torch.argsort(i_idx, stable=True)
        i_sorted = i_idx[sort_idx]
        j_sorted = j_idx[sort_idx]
        shift_sorted = shift_vec[sort_idx]

        r_ij_vec = positions[j_sorted] + shift_sorted - positions[i_sorted]
        r_ij = torch.linalg.norm(r_ij_vec, dim=-1)  # (P,)

        fc_ij = self._fc(r_ij, p.R, p.D)

        # Per-source counts and offsets.
        # unique_i gives the source-atom id for each active group; inv
        # maps each pair to its group index in [0, N_src).
        unique_i, inv, counts = torch.unique_consecutive(
            i_sorted, return_inverse=True, return_counts=True
        )
        N_src = int(unique_i.shape[0])
        n_max = int(counts.max().item())

        # Within-group position (0, 1, ..., n_i-1) for each pair.
        # starts[g] is the first pair index of group g in the sorted
        # order; pos_in_group = arange(P) - starts[inv].
        starts = torch.cumsum(counts, dim=0) - counts  # (N_src,)
        pos_in_group = (
            torch.arange(P, device=device) - starts[inv]
        )  # (P,)

        # Gather the P pairs into a padded (N_src, n_max, *) layout.
        # We need a "slot" index for each pair = (group, pos_in_group).
        # A sentinel slot collects nothing: out-of-range entries in the
        # padded tensors stay at zero.
        flat_slot = inv * n_max + pos_in_group  # (P,)

        def _scatter_into_padded(values: torch.Tensor) -> torch.Tensor:
            """values: (P, ...) -> (N_src, n_max, ...)."""
            trailing = values.shape[1:]
            padded = values.new_zeros((N_src * n_max, *trailing))
            padded.index_copy_(0, flat_slot, values)
            return padded.view(N_src, n_max, *trailing)

        j_pad = torch.full(
            (N_src * n_max,), -1, dtype=torch.long, device=device
        )
        j_pad.index_copy_(0, flat_slot, j_sorted)
        j_pad = j_pad.view(N_src, n_max)

        r_ij_vec_pad = _scatter_into_padded(r_ij_vec)    # (N, n_max, 3)
        r_ij_pad = _scatter_into_padded(r_ij)            # (N, n_max)

        # Valid mask: which slots actually hold a pair.
        slot_mask = torch.zeros(
            N_src * n_max, dtype=torch.bool, device=device
        )
        slot_mask[flat_slot] = True
        slot_mask = slot_mask.view(N_src, n_max)         # (N, n_max)

        # Now compute zeta for every (i, j) = (g, a) against every
        # candidate k = (g, b) using a (N, n_max, n_max) local
        # broadcast. Peak memory: N_src · n_max² · (a few floats). For
        # Si at typical densities n_max is ~6, so even N_src = 10_000
        # this is 10 000 · 36 · 8 · 4 B ≈ 12 MB.
        rij_a = r_ij_vec_pad.unsqueeze(2)                 # (N, a, 1, 3)
        rij_b = r_ij_vec_pad.unsqueeze(1)                 # (N, 1, b, 3)
        r_a = r_ij_pad.unsqueeze(2)                       # (N, a, 1)
        r_b = r_ij_pad.unsqueeze(1)                       # (N, 1, b)

        dot = (rij_a * rij_b).sum(dim=-1)                 # (N, a, b)
        # Avoid 0/0 at padding slots; we mask later.
        denom = r_a * r_b
        denom = torch.where(
            denom > 0, denom, torch.ones_like(denom)
        )
        costheta = dot / denom

        fc_ik = self._fc(r_b, p.R, p.D)                   # (N, 1, b)
        g_theta = self._gijk(costheta, p)                 # (N, a, b)

        delta_r = r_a - r_b                               # (N, a, b)
        # Tersoff 1988 / ASE 3.25 / LAMMPS form: arg = (lambda3 * delta_r)^m.
        # Earlier versions of this file wrote `lambda3 * delta_r**p.m`, which
        # is `lambda3 · delta_r^m` and is missing a factor of lambda3^(m-1).
        # The bug is invisible in diamond Si (delta_r=0 for every triple),
        # but it grows with density in amorphous Si because more neighbour
        # pairs have nonzero delta_r in the cutoff fringe.
        arg = (p.lambda3 * delta_r) ** p.m
        ex_delr = self._safe_exp(arg)                     # (N, a, b)

        triple_term = fc_ik * g_theta * ex_delr           # (N, a, b)

        # Mask out (i) invalid a slots, (ii) invalid b slots, and
        # (iii) the a == b diagonal (k != j).
        mask_a = slot_mask.unsqueeze(2)                   # (N, a, 1)
        mask_b = slot_mask.unsqueeze(1)                   # (N, 1, b)
        eye_ab = torch.eye(
            n_max, dtype=torch.bool, device=device
        ).unsqueeze(0)                                    # (1, a, b)
        valid_ab = mask_a & mask_b & (~eye_ab)
        triple_term = torch.where(
            valid_ab, triple_term, torch.zeros_like(triple_term)
        )

        zeta_pad = triple_term.sum(dim=2)                 # (N, a)

        # Scatter zeta back to pair order.
        zeta = zeta_pad.reshape(-1).gather(0, flat_slot)  # (P,) sorted order

        tmp = p.beta * zeta
        b_ij = (1.0 + tmp ** p.n) ** (-1.0 / (2.0 * p.n))

        repulsive = p.A * torch.exp(-p.lambda1 * r_ij)
        attractive = -p.B * torch.exp(-p.lambda2 * r_ij)

        V_ij = fc_ij * (repulsive + b_ij * attractive)    # (P,) sorted
        return 0.5 * V_ij.sum()

    def energy(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc=(True, True, True),
    ) -> torch.Tensor:
        p = self.params
        cutoff = p.R + p.D
        i_idx, j_idx, shift_vec = build_neighbors(positions, cell, pbc, cutoff)
        return self._energy_from_pairs(positions, i_idx, j_idx, shift_vec)

    def energy_and_forces_autograd(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc=(True, True, True),
    ):
        pos = positions.detach().clone().to(self.dtype).requires_grad_(True)
        cell_t = cell.to(self.dtype)
        E = self.energy(pos, cell_t, pbc)
        (grad,) = torch.autograd.grad(E, pos)
        return E.detach(), -grad.detach()

    # ------------------------------------------------------------------
    # Analytical forces (ported from ase.calculators.tersoff)
    # ------------------------------------------------------------------
    def energy_and_forces_analytical(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc=(True, True, True),
    ):
        p = self.params
        cutoff = p.R + p.D
        pos = positions.to(self.dtype)
        cell_t = cell.to(self.dtype)

        i_all, j_all, s_all = build_neighbors(pos, cell_t, pbc, cutoff)
        N = pos.shape[0]
        forces = torch.zeros((N, 3), dtype=self.dtype, device=pos.device)
        E_total = torch.zeros((), dtype=self.dtype, device=pos.device)

        # Group pairs by i.
        for i in range(N):
            mask_i = i_all == i
            j_idx = j_all[mask_i]
            shifts = s_all[mask_i]
            if j_idx.numel() == 0:
                continue
            vectors = pos[j_idx] + shifts - pos[i]  # (ni, 3)
            distances = torch.linalg.norm(vectors, dim=-1)  # (ni,)

            ni = j_idx.shape[0]
            for jj in range(ni):
                idx_j = int(j_idx[jj].item())
                r_ij_vec = vectors[jj]
                abs_rij = distances[jj]
                rij_hat = r_ij_vec / abs_rij

                fc_ij = self._fc(abs_rij, p.R, p.D)
                if float(fc_ij) == 0.0:
                    continue

                # zeta_ij and its derivatives
                zeta, dzeta_dri, dzeta_drj_all, dzeta_drk_all = self._zeta_and_deriv(
                    jj, vectors, distances, p
                )

                b_ij = (1.0 + (p.beta * zeta) ** p.n) ** (-1.0 / (2.0 * p.n))
                # derivative of b_ij wrt zeta (only if zeta>0)
                if float(zeta) > 0.0:
                    tmp = p.beta * zeta
                    bij_d = (
                        -0.5
                        * (1.0 + tmp**p.n) ** (-1.0 - (1.0 / (2.0 * p.n)))
                        * (p.beta * tmp ** (p.n - 1.0))
                    )
                else:
                    bij_d = torch.zeros((), dtype=self.dtype, device=pos.device)

                repulsive = p.A * torch.exp(-p.lambda1 * abs_rij)
                attractive = -p.B * torch.exp(-p.lambda2 * abs_rij)

                # Distribute pair energy 0.25 + 0.25 to atoms i, j (matches ASE).
                E_total = E_total + 0.5 * fc_ij * (repulsive + b_ij * attractive)

                dfc_ij = self._fc_d(abs_rij, p.R, p.D)
                rep_deriv = -p.lambda1 * repulsive
                att_deriv = -p.lambda2 * attractive

                tmp_scalar = dfc_ij * (repulsive + b_ij * attractive)
                tmp_scalar = tmp_scalar + fc_ij * (rep_deriv + b_ij * att_deriv)

                # grad wrt position of atom j (radial part)
                grad_j_radial = 0.5 * tmp_scalar * rij_hat
                forces[i] = forces[i] + grad_j_radial
                forces[idx_j] = forces[idx_j] - grad_j_radial

                # three-body force contributions
                for kk in range(ni):
                    if kk == jj:
                        continue
                    idx_k = int(j_idx[kk].item())
                    # dzeta contributions (already computed):
                    dztdri = dzeta_dri[kk]
                    dztdrj = dzeta_drj_all[kk]
                    dztdrk = dzeta_drk_all[kk]

                    gi = 0.5 * fc_ij * bij_d * dztdri * attractive
                    gj = 0.5 * fc_ij * bij_d * dztdrj * attractive
                    gk = 0.5 * fc_ij * bij_d * dztdrk * attractive

                    forces[i] = forces[i] - gi
                    forces[idx_j] = forces[idx_j] - gj
                    forces[idx_k] = forces[idx_k] - gk

        return E_total, forces

    def _zeta_and_deriv(self, j_slot, vectors, distances, p: TersoffParameters):
        """Compute zeta_ij and per-k derivatives.

        Returns
        -------
        zeta : scalar tensor
        dzeta_dri, dzeta_drj_all, dzeta_drk_all : (ni, 3) tensors
            Each row k gives the k-th term's contribution to d(zeta_ij)/dr_{i|j|k}.
            Rows where k == j_slot are zero.
        """
        ni = vectors.shape[0]
        device = vectors.device
        dtype = vectors.dtype

        r_ij_vec = vectors[j_slot]
        abs_rij = distances[j_slot]
        rij_hat = r_ij_vec / abs_rij

        zeta = torch.zeros((), dtype=dtype, device=device)
        dri = torch.zeros((ni, 3), dtype=dtype, device=device)
        drj = torch.zeros((ni, 3), dtype=dtype, device=device)
        drk = torch.zeros((ni, 3), dtype=dtype, device=device)

        for k in range(ni):
            if k == j_slot:
                continue
            r_ik_vec = vectors[k]
            abs_rik = distances[k]
            if abs_rik > p.R + p.D:
                continue
            rik_hat = r_ik_vec / abs_rik

            fcik = self._fc(abs_rik, p.R, p.D)
            dfcik = self._fc_d(abs_rik, p.R, p.D)

            # Tersoff 1988 / ASE 3.25 / LAMMPS form: arg = (lambda3 * delta_r)^m.
            # See _energy_from_pairs above for the bug history.
            arg = (p.lambda3 * (abs_rij - abs_rik)) ** p.m
            ex_delr = self._safe_exp(arg)
            # d(arg)/d(r_ij) = m · lambda3^m · delta_r^(m-1)
            ex_delr_d = (
                p.m
                * p.lambda3 ** p.m
                * (abs_rij - abs_rik) ** (p.m - 1)
                * ex_delr
            )

            costheta = rij_hat @ rik_hat
            gijk = self._gijk(costheta, p)
            gijk_d = self._gijk_d(costheta, p)

            zeta = zeta + fcik * gijk * ex_delr

            # d cos(theta) / d r_i, r_j, r_k
            drj_cos = (rik_hat / abs_rik - costheta * rij_hat / abs_rij) / 1.0
            drj_cos = (r_ik_vec / abs_rik - costheta * r_ij_vec / abs_rij) / abs_rij
            drk_cos = (r_ij_vec / abs_rij - costheta * r_ik_vec / abs_rik) / abs_rik
            dri_cos = -(drj_cos + drk_cos)

            dri_k = -dfcik * gijk * ex_delr * rik_hat
            dri_k = dri_k + fcik * gijk_d * ex_delr * dri_cos
            dri_k = dri_k + fcik * gijk * ex_delr_d * rik_hat
            dri_k = dri_k - fcik * gijk * ex_delr_d * rij_hat

            drj_k = fcik * gijk_d * ex_delr * drj_cos
            drj_k = drj_k + fcik * gijk * ex_delr_d * rij_hat

            drk_k = dfcik * gijk * ex_delr * rik_hat
            drk_k = drk_k + fcik * gijk_d * ex_delr * drk_cos
            drk_k = drk_k - fcik * gijk * ex_delr_d * rik_hat

            dri[k] = dri_k
            drj[k] = drj_k
            drk[k] = drk_k

        return zeta, dri, drj, drk


# Backward-compat alias
Tersoff = TorchTersoff
