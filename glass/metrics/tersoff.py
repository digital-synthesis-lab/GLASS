"""Tersoff potential energy and force metrics for atomic structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from ase import Atoms
from scipy.stats import wasserstein_distance


# Force-magnitude histogram parameters used throughout.
_FORCE_BINS = 50
_FORCE_MAX_DEFAULT = 20.0  # eV/Å; clipped for histogram, not for scalar stats


@dataclass
class TersoffMetrics:
    """Tersoff potential energy and force statistics for a single structure.

    All force quantities refer to the per-atom Cartesian force magnitude
    |F_i| = sqrt(Fx²+Fy²+Fz²) in eV/Å.

    Attributes:
        energy_per_atom: Total Tersoff energy divided by N_atoms [eV/atom].
        forces_rms: RMS of per-atom force magnitudes [eV/Å].
        forces_max: Maximum per-atom force magnitude [eV/Å].
        forces_mean: Mean per-atom force magnitude [eV/Å].
        forces_std: Standard deviation of per-atom force magnitudes [eV/Å].
        forces_histogram: Normalised histogram counts of per-atom |F|.
            Length = _FORCE_BINS. Sum ≈ 1.0 (probability mass).
        forces_histogram_bins: Bin *edges* (length = _FORCE_BINS + 1) [eV/Å].
            Covers [0, forces_max_histogram]; atoms beyond the upper edge are
            clipped into the last bin.
        forces_max_histogram: Upper edge of the histogram range [eV/Å].
    """

    energy_per_atom: float
    forces_rms: float
    forces_max: float
    forces_mean: float
    forces_std: float
    forces_histogram: List[float]
    forces_histogram_bins: List[float]
    forces_max_histogram: float

    def to_dict(self) -> Dict:
        return {
            "energy_per_atom": self.energy_per_atom,
            "forces_rms": self.forces_rms,
            "forces_max": self.forces_max,
            "forces_mean": self.forces_mean,
            "forces_std": self.forces_std,
            "forces_histogram": self.forces_histogram,
            "forces_histogram_bins": self.forces_histogram_bins,
            "forces_max_histogram": self.forces_max_histogram,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TersoffMetrics":
        return cls(
            energy_per_atom=d["energy_per_atom"],
            forces_rms=d["forces_rms"],
            forces_max=d["forces_max"],
            forces_mean=d["forces_mean"],
            forces_std=d["forces_std"],
            forces_histogram=d["forces_histogram"],
            forces_histogram_bins=d["forces_histogram_bins"],
            forces_max_histogram=d["forces_max_histogram"],
        )


def compute_tersoff_metrics(
    atoms: Atoms,
    device: str = "cpu",
    force_max_histogram: Optional[float] = None,
    n_bins: int = _FORCE_BINS,
) -> TersoffMetrics:
    """Compute Tersoff energy and force statistics for an ASE Atoms object.

    Uses the glass PyTorch Tersoff Si parameterization.  The atoms object
    must contain only Si; if it contains other species the calculator will
    still run but parameters are Si-Si-Si only.

    Args:
        atoms: ASE Atoms object.
        device: Torch device string ('cpu' or 'cuda').
        force_max_histogram: Upper edge for the force histogram [eV/Å].
            If None, uses max(|F|) rounded up to the nearest integer, capped
            at _FORCE_MAX_DEFAULT.
        n_bins: Number of histogram bins.

    Returns:
        TersoffMetrics with energy and force distribution statistics.
    """
    from glass.potentials.tersoff.ase_calc import silicon_calculator

    calc = silicon_calculator(device=device)
    atoms = atoms.copy()
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()          # (N, 3) eV/Å
    n_atoms = len(atoms)

    force_mags = np.linalg.norm(forces, axis=1)  # (N,) eV/Å

    fmax = float(force_mags.max())
    fmean = float(force_mags.mean())
    fstd = float(force_mags.std())
    frms = float(np.sqrt((force_mags ** 2).mean()))

    if force_max_histogram is None:
        force_max_histogram = min(float(np.ceil(fmax)) + 1.0, _FORCE_MAX_DEFAULT)
        force_max_histogram = max(force_max_histogram, 1.0)

    bins = np.linspace(0.0, force_max_histogram, n_bins + 1)
    counts, _ = np.histogram(force_mags, bins=bins)
    hist_norm = counts / max(counts.sum(), 1)

    return TersoffMetrics(
        energy_per_atom=float(energy) / n_atoms,
        forces_rms=frms,
        forces_max=fmax,
        forces_mean=fmean,
        forces_std=fstd,
        forces_histogram=hist_norm.tolist(),
        forces_histogram_bins=bins.tolist(),
        forces_max_histogram=force_max_histogram,
    )


# ---------------------------------------------------------------------------
# Error / comparison functions
# ---------------------------------------------------------------------------

def _aligned_histograms(
    a: TersoffMetrics, b: TersoffMetrics
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return two histograms resampled onto a common bin grid.

    Uses the finer of the two bin grids (more bins, lower upper bound).
    Returns (hist_a, hist_b, bin_centers).
    """
    # Use the same number of bins, common range [0, max of both upper edges]
    n = len(a.forces_histogram)
    upper = max(a.forces_max_histogram, b.forces_max_histogram)
    bins = np.linspace(0.0, upper, n + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    def _resample(m: TersoffMetrics) -> np.ndarray:
        orig_bins = np.array(m.forces_histogram_bins)
        orig_centers = 0.5 * (orig_bins[:-1] + orig_bins[1:])
        hist = np.array(m.forces_histogram)
        return np.interp(centers, orig_centers, hist, left=0.0, right=0.0)

    ha = _resample(a)
    hb = _resample(b)
    # Renormalise
    ha = ha / max(ha.sum(), 1e-12)
    hb = hb / max(hb.sum(), 1e-12)
    return ha, hb, centers


def tersoff_energy_error(
    ref: TersoffMetrics, target: TersoffMetrics
) -> float:
    """Absolute difference in energy per atom [eV/atom]."""
    return abs(target.energy_per_atom - ref.energy_per_atom)


def tersoff_forces_rms_error(
    ref: TersoffMetrics, target: TersoffMetrics
) -> float:
    """Absolute difference in force RMS [eV/Å]."""
    return abs(target.forces_rms - ref.forces_rms)


def tersoff_forces_max_error(
    ref: TersoffMetrics, target: TersoffMetrics
) -> float:
    """Absolute difference in max force magnitude [eV/Å]."""
    return abs(target.forces_max - ref.forces_max)


def tersoff_forces_histogram_rmse(
    ref: TersoffMetrics, target: TersoffMetrics
) -> float:
    """RMSE between the normalised force-magnitude histograms."""
    ha, hb, _ = _aligned_histograms(ref, target)
    return float(np.sqrt(np.mean((ha - hb) ** 2)))


def tersoff_forces_emd(
    ref: TersoffMetrics, target: TersoffMetrics
) -> float:
    """Earth Mover's Distance between force-magnitude distributions [eV/Å]."""
    ha, hb, centers = _aligned_histograms(ref, target)
    return float(wasserstein_distance(centers, centers, ha, hb))
