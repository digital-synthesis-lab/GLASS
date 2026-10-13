"""Error metrics for comparing structural metrics between structures.

This module provides functions to compute dissimilarity/error metrics between
reference and target structures based on their computed metrics.
"""

import numpy as np
from typing import Optional, Dict, Union, List, Tuple
from scipy.stats import wasserstein_distance
from scipy.interpolate import interp1d

from glass.metrics.core import (
    PDFMetrics,
    ADFMetrics,
    CoordinationMetrics,
    RingMetrics,
    StructuralMetrics,
)


def _interpolate_to_common_grid(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    num_points: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate two curves to a common grid.
    
    Args:
        x1, y1: First curve
        x2, y2: Second curve
        num_points: Number of points for interpolation
    
    Returns:
        Tuple of (common_x, y1_interp, y2_interp)
    """
    # Determine common range
    x_min = max(x1.min(), x2.min())
    x_max = min(x1.max(), x2.max())
    
    # Create common grid
    x_common = np.linspace(x_min, x_max, num_points)
    
    # Interpolate both curves
    f1 = interp1d(x1, y1, kind='linear', bounds_error=False, fill_value=0)
    f2 = interp1d(x2, y2, kind='linear', bounds_error=False, fill_value=0)
    
    y1_interp = f1(x_common)
    y2_interp = f2(x_common)
    
    return x_common, y1_interp, y2_interp


# ============================================================================
# PDF Error Metrics
# ============================================================================

def pdf_rmse(ref: PDFMetrics, target: PDFMetrics) -> float:
    """Compute RMSE between two PDFs.
    
    Interpolates both PDFs to a common grid and computes root mean square error.
    
    Args:
        ref: Reference PDF metrics
        target: Target PDF metrics
    
    Returns:
        RMSE value (absolute)
    """
    _, g_ref, g_target = _interpolate_to_common_grid(
        ref.r, ref.g_r, target.r, target.g_r
    )
    
    # Compute RMSE
    mse = np.mean((g_ref - g_target) ** 2)
    return float(np.sqrt(mse))


def pdf_mae(ref: PDFMetrics, target: PDFMetrics) -> float:
    """Compute mean absolute error between two PDFs.
    
    Args:
        ref: Reference PDF metrics
        target: Target PDF metrics
    
    Returns:
        MAE value (absolute)
    """
    _, g_ref, g_target = _interpolate_to_common_grid(
        ref.r, ref.g_r, target.r, target.g_r
    )
    
    return float(np.mean(np.abs(g_ref - g_target)))


def pdf_area_between(ref: PDFMetrics, target: PDFMetrics) -> float:
    """Compute area between two PDF curves.
    
    This is the integral of the absolute difference between curves.
    
    Args:
        ref: Reference PDF metrics
        target: Target PDF metrics
    
    Returns:
        Area between curves (integral of |g_ref - g_target|)
    """
    x_common, g_ref, g_target = _interpolate_to_common_grid(
        ref.r, ref.g_r, target.r, target.g_r
    )
    
    # Integrate absolute difference using trapezoidal rule
    dx = x_common[1] - x_common[0]
    area = np.trapezoid(np.abs(g_ref - g_target), x_common)
    
    return float(area)


def pdf_cosine_similarity(ref: PDFMetrics, target: PDFMetrics) -> float:
    """Compute cosine similarity between two PDFs.
    
    Returns value in [-1, 1] where 1 means identical, 0 means orthogonal.
    
    Args:
        ref: Reference PDF metrics
        target: Target PDF metrics
    
    Returns:
        Cosine similarity (not error, higher is better)
    """
    _, g_ref, g_target = _interpolate_to_common_grid(
        ref.r, ref.g_r, target.r, target.g_r
    )
    
    # Compute cosine similarity
    dot_product = np.sum(g_ref * g_target)
    norm_ref = np.sqrt(np.sum(g_ref ** 2))
    norm_target = np.sqrt(np.sum(g_target ** 2))
    
    if norm_ref == 0 or norm_target == 0:
        return 0.0
    
    return float(dot_product / (norm_ref * norm_target))


def pdf_r_chi2(ref: PDFMetrics, target: PDFMetrics, sigma: float = 0.1) -> float:
    """Compute reduced chi-squared between two PDFs.
    
    chi^2 = sum((ref - target)^2 / sigma^2) / N
    
    Args:
        ref: Reference PDF metrics
        target: Target PDF metrics
        sigma: Estimated uncertainty (default 0.1)
    
    Returns:
        Reduced chi-squared value
    """
    _, g_ref, g_target = _interpolate_to_common_grid(
        ref.r, ref.g_r, target.r, target.g_r
    )
    
    # Compute chi-squared
    chi2 = np.sum((g_ref - g_target) ** 2 / sigma ** 2)
    n_points = len(g_ref)
    
    return float(chi2 / n_points)


def pdf_peak_position_error(ref: PDFMetrics, target: PDFMetrics) -> Optional[float]:
    """Compute error in first peak position.
    
    Args:
        ref: Reference PDF metrics
        target: Target PDF metrics
    
    Returns:
        Absolute error in peak position (Å), or None if peak not found
    """
    if ref.first_peak_position is None or target.first_peak_position is None:
        return None
    
    return float(abs(ref.first_peak_position - target.first_peak_position))


def pdf_peak_height_error(ref: PDFMetrics, target: PDFMetrics) -> Optional[float]:
    """Compute relative error in first peak height.
    
    Args:
        ref: Reference PDF metrics
        target: Target PDF metrics
    
    Returns:
        Relative error in peak height (%), or None if peak not found
    """
    if ref.first_peak_height is None or target.first_peak_height is None:
        return None
    
    if ref.first_peak_height == 0:
        return float(abs(target.first_peak_height))
    
    rel_error = abs(ref.first_peak_height - target.first_peak_height) / ref.first_peak_height
    return float(rel_error * 100)  # Return as percentage


# ============================================================================
# ADF Error Metrics
# ============================================================================

def adf_rmse(ref: ADFMetrics, target: ADFMetrics) -> float:
    """Compute RMSE between two ADFs.
    
    Args:
        ref: Reference ADF metrics
        target: Target ADF metrics
    
    Returns:
        RMSE value
    """
    _, adf_ref, adf_target = _interpolate_to_common_grid(
        ref.angles, ref.adf, target.angles, target.adf
    )
    
    mse = np.mean((adf_ref - adf_target) ** 2)
    return float(np.sqrt(mse))


def adf_cosine_similarity(ref: ADFMetrics, target: ADFMetrics) -> float:
    """Compute cosine similarity between two ADFs.
    
    Args:
        ref: Reference ADF metrics
        target: Target ADF metrics
    
    Returns:
        Cosine similarity in [-1, 1]
    """
    _, adf_ref, adf_target = _interpolate_to_common_grid(
        ref.angles, ref.adf, target.angles, target.adf
    )
    
    dot_product = np.sum(adf_ref * adf_target)
    norm_ref = np.sqrt(np.sum(adf_ref ** 2))
    norm_target = np.sqrt(np.sum(adf_target ** 2))
    
    if norm_ref == 0 or norm_target == 0:
        return 0.0
    
    return float(dot_product / (norm_ref * norm_target))


# ============================================================================
# Coordination Number Error Metrics
# ============================================================================

def coordination_emd(ref: CoordinationMetrics, target: CoordinationMetrics) -> float:
    """Compute Earth Mover's Distance (Wasserstein) between coordination distributions.
    
    This is the optimal transport distance between two histograms.
    
    Args:
        ref: Reference coordination metrics
        target: Target coordination metrics
    
    Returns:
        EMD/Wasserstein distance
    """
    # Normalize histograms to probability distributions
    hist_ref = ref.coordination_histogram.astype(float)
    hist_ref = hist_ref / hist_ref.sum() if hist_ref.sum() > 0 else hist_ref
    
    hist_target = target.coordination_histogram.astype(float)
    hist_target = hist_target / hist_target.sum() if hist_target.sum() > 0 else hist_target
    
    # Determine common support
    max_coord = max(len(hist_ref), len(hist_target))
    
    # Pad shorter histogram
    if len(hist_ref) < max_coord:
        hist_ref = np.pad(hist_ref, (0, max_coord - len(hist_ref)))
    if len(hist_target) < max_coord:
        hist_target = np.pad(hist_target, (0, max_coord - len(hist_target)))
    
    # Compute Wasserstein distance
    # The positions are the coordination numbers themselves [0, 1, 2, ...]
    positions = np.arange(max_coord).astype(float)
    
    return float(wasserstein_distance(positions, positions, hist_ref, hist_target))


def coordination_histogram_rmse(ref: CoordinationMetrics, target: CoordinationMetrics) -> float:
    """Compute RMSE between coordination histograms.
    
    Args:
        ref: Reference coordination metrics
        target: Target coordination metrics
    
    Returns:
        RMSE value
    """
    hist_ref = ref.coordination_histogram.astype(float)
    hist_target = target.coordination_histogram.astype(float)
    
    # Pad to common length
    max_len = max(len(hist_ref), len(hist_target))
    if len(hist_ref) < max_len:
        hist_ref = np.pad(hist_ref, (0, max_len - len(hist_ref)))
    if len(hist_target) < max_len:
        hist_target = np.pad(hist_target, (0, max_len - len(hist_target)))
    
    mse = np.mean((hist_ref - hist_target) ** 2)
    return float(np.sqrt(mse))


def coordination_mean_error(ref: CoordinationMetrics, target: CoordinationMetrics) -> float:
    """Compute absolute error in mean coordination number.
    
    Args:
        ref: Reference coordination metrics
        target: Target coordination metrics
    
    Returns:
        Absolute error in mean coordination
    """
    return float(abs(ref.mean_coordination - target.mean_coordination))


def coordination_std_error(ref: CoordinationMetrics, target: CoordinationMetrics) -> float:
    """Compute absolute error in coordination std.
    
    Args:
        ref: Reference coordination metrics
        target: Target coordination metrics
    
    Returns:
        Absolute error in std
    """
    return float(abs(ref.std_coordination - target.std_coordination))


# ============================================================================
# Ring Statistics Error Metrics
# ============================================================================

def rings_rmse(ref: RingMetrics, target: RingMetrics) -> float:
    """Compute RMSE between ring count distributions.
    
    Args:
        ref: Reference ring metrics
        target: Target ring metrics
    
    Returns:
        RMSE value
    """
    # Use ring_counts (not fractions) for RMSE
    counts_ref = ref.ring_counts.astype(float)
    counts_target = target.ring_counts.astype(float)
    
    # Pad to common length
    max_len = max(len(counts_ref), len(counts_target))
    if len(counts_ref) < max_len:
        counts_ref = np.pad(counts_ref, (0, max_len - len(counts_ref)))
    if len(counts_target) < max_len:
        counts_target = np.pad(counts_target, (0, max_len - len(counts_target)))
    
    mse = np.mean((counts_ref - counts_target) ** 2)
    return float(np.sqrt(mse))


def rings_mae(ref: RingMetrics, target: RingMetrics) -> float:
    """Compute mean absolute error between ring count distributions.
    
    Args:
        ref: Reference ring metrics
        target: Target ring metrics
    
    Returns:
        MAE value
    """
    counts_ref = ref.ring_counts.astype(float)
    counts_target = target.ring_counts.astype(float)
    
    # Pad to common length
    max_len = max(len(counts_ref), len(counts_target))
    if len(counts_ref) < max_len:
        counts_ref = np.pad(counts_ref, (0, max_len - len(counts_ref)))
    if len(counts_target) < max_len:
        counts_target = np.pad(counts_target, (0, max_len - len(counts_target)))
    
    return float(np.mean(np.abs(counts_ref - counts_target)))


def rings_cosine_similarity(ref: RingMetrics, target: RingMetrics) -> float:
    """Compute cosine similarity between ring distributions.
    
    Args:
        ref: Reference ring metrics
        target: Target ring metrics
    
    Returns:
        Cosine similarity in [-1, 1]
    """
    # Use ring_counts for cosine similarity
    counts_ref = ref.ring_counts.astype(float)
    counts_target = target.ring_counts.astype(float)
    
    # Pad to common length
    max_len = max(len(counts_ref), len(counts_target))
    if len(counts_ref) < max_len:
        counts_ref = np.pad(counts_ref, (0, max_len - len(counts_ref)))
    if len(counts_target) < max_len:
        counts_target = np.pad(counts_target, (0, max_len - len(counts_target)))
    
    dot_product = np.sum(counts_ref * counts_target)
    norm_ref = np.sqrt(np.sum(counts_ref ** 2))
    norm_target = np.sqrt(np.sum(counts_target ** 2))
    
    if norm_ref == 0 or norm_target == 0:
        return 0.0
    
    return float(dot_product / (norm_ref * norm_target))


def rings_emd(ref: RingMetrics, target: RingMetrics) -> float:
    """Compute Earth Mover's Distance (Wasserstein) between ring distributions.
    
    Args:
        ref: Reference ring metrics
        target: Target ring metrics
    
    Returns:
        EMD/Wasserstein distance
    """
    # Normalize to probability distributions
    hist_ref = ref.ring_counts.astype(float)
    hist_ref = hist_ref / hist_ref.sum() if hist_ref.sum() > 0 else hist_ref
    
    hist_target = target.ring_counts.astype(float)
    hist_target = hist_target / hist_target.sum() if hist_target.sum() > 0 else hist_target
    
    # Pad to common length
    max_len = max(len(hist_ref), len(hist_target))
    if len(hist_ref) < max_len:
        hist_ref = np.pad(hist_ref, (0, max_len - len(hist_ref)))
    if len(hist_target) < max_len:
        hist_target = np.pad(hist_target, (0, max_len - len(hist_target)))
    
    # The positions are the ring lengths themselves [0, 1, 2, ...]
    positions = np.arange(max_len).astype(float)
    
    return float(wasserstein_distance(positions, positions, hist_ref, hist_target))


def rings_total_error(ref: RingMetrics, target: RingMetrics) -> float:
    """Compute relative error in total number of rings.
    
    Args:
        ref: Reference ring metrics
        target: Target ring metrics
    
    Returns:
        Relative error in total ring count
    """
    if ref.total_rings == 0:
        if target.total_rings == 0:
            return 0.0
        return float(abs(target.total_rings))
    
    rel_error = abs(ref.total_rings - target.total_rings) / ref.total_rings
    return float(rel_error)


# ============================================================================
# Combined Error Metrics
# ============================================================================

def compute_all_errors(
    ref: StructuralMetrics,
    target: StructuralMetrics,
) -> Dict[str, Union[float, None]]:
    """Compute all error metrics between reference and target structures.
    
    Args:
        ref: Reference structural metrics
        target: Target structural metrics
    
    Returns:
        Dictionary with all error metrics
    """
    errors = {}
    
    # PDF errors
    errors['pdf_rmse'] = pdf_rmse(ref.pdf, target.pdf)
    errors['pdf_mae'] = pdf_mae(ref.pdf, target.pdf)
    errors['pdf_area'] = pdf_area_between(ref.pdf, target.pdf)
    errors['pdf_cosine'] = pdf_cosine_similarity(ref.pdf, target.pdf)
    errors['pdf_r_chi2'] = pdf_r_chi2(ref.pdf, target.pdf)
    errors['pdf_peak_position_error'] = pdf_peak_position_error(ref.pdf, target.pdf)
    errors['pdf_peak_height_error'] = pdf_peak_height_error(ref.pdf, target.pdf)
    
    # ADF errors
    errors['adf_rmse'] = adf_rmse(ref.adf, target.adf)
    errors['adf_cosine'] = adf_cosine_similarity(ref.adf, target.adf)
    
    # Coordination errors
    errors['coordination_emd'] = coordination_emd(ref.coordination, target.coordination)
    errors['coordination_rmse'] = coordination_histogram_rmse(ref.coordination, target.coordination)
    errors['coordination_mean_error'] = coordination_mean_error(ref.coordination, target.coordination)
    errors['coordination_std_error'] = coordination_std_error(ref.coordination, target.coordination)
    
    # Ring statistics errors (if available)
    if ref.rings is not None and target.rings is not None:
        errors['rings_rmse'] = rings_rmse(ref.rings, target.rings)
        errors['rings_mae'] = rings_mae(ref.rings, target.rings)
        errors['rings_cosine'] = rings_cosine_similarity(ref.rings, target.rings)
        errors['rings_emd'] = rings_emd(ref.rings, target.rings)
        errors['rings_total_error'] = rings_total_error(ref.rings, target.rings)

    # Tersoff energy/force errors (if available)
    if ref.tersoff is not None and target.tersoff is not None:
        from glass.metrics.tersoff import (
            tersoff_energy_error,
            tersoff_forces_rms_error,
            tersoff_forces_max_error,
            tersoff_forces_histogram_rmse,
            tersoff_forces_emd,
        )
        errors['tersoff_energy_error'] = tersoff_energy_error(ref.tersoff, target.tersoff)
        errors['tersoff_forces_rms_error'] = tersoff_forces_rms_error(ref.tersoff, target.tersoff)
        errors['tersoff_forces_max_error'] = tersoff_forces_max_error(ref.tersoff, target.tersoff)
        errors['tersoff_forces_histogram_rmse'] = tersoff_forces_histogram_rmse(ref.tersoff, target.tersoff)
        errors['tersoff_forces_emd'] = tersoff_forces_emd(ref.tersoff, target.tersoff)

    return errors


def compute_weighted_error(
    errors: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute weighted sum of errors.
    
    Args:
        errors: Dictionary of error metrics
        weights: Dictionary of weights for each metric (default: equal weights)
    
    Returns:
        Weighted error sum
    """
    if weights is None:
        # Default weights emphasizing structural features
        weights = {
            'pdf_rmse': 1.0,
            'coordination_emd': 1.0,
            'adf_rmse': 0.5,
        }
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for key, weight in weights.items():
        if key in errors and errors[key] is not None:
            weighted_sum += weight * errors[key]
            total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    return float(weighted_sum / total_weight)
