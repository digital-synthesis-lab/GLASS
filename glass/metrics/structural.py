"""Structural metrics: PDF and ADF computations."""

import numpy as np
from typing import Optional
from ase import Atoms
from ase.neighborlist import neighbor_list
from scipy.signal import find_peaks

from glass.metrics.core import PDFMetrics, ADFMetrics


def compute_pdf(
    atoms: Atoms,
    cutoff: float = 8.0,
    bin_size: int = 200,
    sigma: Optional[float] = 0.15,
    find_peaks_prominence: float = 0.1,
) -> PDFMetrics:
    """Compute Pair Distribution Function (PDF/RDF) for ASE Atoms.
    
    Memory-efficient implementation using ASE neighbor_list.
    
    Args:
        atoms: ASE Atoms object
        cutoff: Maximum distance for PDF computation
        bin_size: Number of bins
        sigma: Gaussian smoothing sigma. Default 0.15 Å.
        find_peaks_prominence: Minimum prominence for peak detection
    
    Returns:
        PDFMetrics object with PDF and extracted features
    """
    # Get neighbor pairs
    i, j, d = neighbor_list('ijd', atoms, cutoff)
    
    # Remove self-interactions
    mask = i != j
    d = d[mask]
    
    if len(d) == 0:
        r = np.linspace(0, cutoff, bin_size)
        g_r = np.zeros(bin_size)
        return PDFMetrics(
            r=r, g_r=g_r,
            first_peak_position=None, first_peak_height=None,
            first_minima_position=None, first_minima_height=None,
            coord_cutoff=None,
        )
    
    # Create histogram
    bin_edges = np.linspace(0, cutoff, bin_size + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    delta_r = bin_edges[1] - bin_edges[0]
    
    hist, _ = np.histogram(d, bins=bin_edges)
    
    # Normalization: g(r) = (V / N^2) * hist / (4πr²Δr)
    n_atoms = len(atoms)
    volume = atoms.get_volume()
    prefactor = volume / (n_atoms ** 2)
    
    shell_volumes = np.zeros_like(bin_centers)
    mask = bin_centers > 0
    shell_volumes[mask] = 4 * np.pi * bin_centers[mask]**2 * delta_r
    
    g_r = prefactor * hist / (shell_volumes + 1e-10)
    
    # Apply smoothing
    if sigma is not None and sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        g_r = gaussian_filter1d(g_r, sigma=sigma/delta_r, mode='nearest')
    
    # Find peaks and minimum
    first_peak_pos = None
    first_peak_height = None
    first_min_pos = None
    first_min_height = None
    coord_cutoff = None
    
    peaks, _ = find_peaks(g_r, prominence=find_peaks_prominence, distance=5)
    
    if len(peaks) > 0:
        first_peak_idx = peaks[0]
        first_peak_pos = float(bin_centers[first_peak_idx])
        first_peak_height = float(g_r[first_peak_idx])
        
        # Find minimum after first peak
        search_start = first_peak_idx + 3
        search_end = len(g_r)
        if len(peaks) > 1:
            search_end = min(search_end, peaks[1] - 3)
        
        if search_start < search_end:
            min_idx = search_start + np.argmin(g_r[search_start:search_end])
            first_min_pos = float(bin_centers[min_idx])
            first_min_height = float(g_r[min_idx])
            
            if first_min_pos < 5.0 and first_min_pos > first_peak_pos + 0.1:
                coord_cutoff = first_min_pos
            else:
                coord_cutoff = min(first_peak_pos * 1.3, 3.2)
        
        if coord_cutoff is None:
            coord_cutoff = min(first_peak_pos * 1.35, 3.2)
    
    return PDFMetrics(
        r=bin_centers,
        g_r=g_r,
        first_peak_position=first_peak_pos,
        first_peak_height=first_peak_height,
        first_minima_position=first_min_pos,
        first_minima_height=first_min_height,
        coord_cutoff=coord_cutoff,
    )


def compute_adf(
    atoms: Atoms,
    cutoff: Optional[float] = None,
    bin_size: int = 100,
    sigma: float = 0.05,
    auto_cutoff: bool = True,
) -> ADFMetrics:
    """Compute Angular Distribution Function (ADF) for ASE Atoms.
    
    Args:
        atoms: ASE Atoms object
        cutoff: Cutoff for triplet search
        bin_size: Number of angle bins
        sigma: Gaussian smoothing sigma
        auto_cutoff: If True and cutoff is None, use PDF first minimum
    
    Returns:
        ADFMetrics object
    """
    from collections import defaultdict

    if cutoff is None and auto_cutoff:
        pdf_metrics = compute_pdf(atoms, cutoff=8.0, bin_size=100)
        cutoff = pdf_metrics.coord_cutoff or 3.5
    elif cutoff is None:
        cutoff = 3.5

    # Same neighbour definition as compute_coordination / compute_pdf (the
    # ``neighbor_list`` function, no skin), so ADF and coordination counts agree.
    # ``D`` is the minimum-image bond vector centre -> neighbour, so bonds that
    # cross the periodic boundary need no manual offset bookkeeping.
    i_idx, _j_idx, D = neighbor_list("ijD", atoms, cutoff)

    nbr_vecs: dict = defaultdict(list)
    for centre, vec in zip(i_idx, D):
        nbr_vecs[int(centre)].append(vec)

    # For each centre, enumerate every unordered pair of its bonded neighbours
    # (a < b), so each neighbour-centre-neighbour angle is recorded exactly once
    # regardless of how the atoms are numbered.
    angles = []
    for vecs in nbr_vecs.values():
        if len(vecs) < 2:
            continue
        V = np.asarray(vecs)
        norms = np.linalg.norm(V, axis=1)
        m = len(V)
        for a in range(m):
            if norms[a] < 1e-10:
                continue
            for b in range(a + 1, m):
                if norms[b] < 1e-10:
                    continue
                cos_angle = float(np.dot(V[a], V[b]) / (norms[a] * norms[b]))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angles.append(np.arccos(cos_angle))
    
    angles = np.array(angles)
    
    angle_bins = np.linspace(0, np.pi, bin_size + 1)
    bin_centers = 0.5 * (angle_bins[:-1] + angle_bins[1:])
    
    if len(angles) == 0:
        adf = np.zeros(bin_size)
    else:
        adf = np.zeros(bin_size)
        for angle in angles:
            adf += np.exp(-0.5 * ((bin_centers - angle) / sigma) ** 2)
        adf = adf / (len(angles) * sigma * np.sqrt(2 * np.pi))
    
    dominant_angle = None
    dominant_angle_deg = None
    
    peaks, _ = find_peaks(adf, prominence=0.01)
    if len(peaks) > 0:
        # The tallest peak, not the first: in amorphous Si a small 60 degree
        # shoulder (three-membered rings) precedes the tetrahedral peak.
        dominant_idx = peaks[np.argmax(adf[peaks])]
        dominant_angle = float(bin_centers[dominant_idx])
        dominant_angle_deg = float(np.degrees(dominant_angle))
    
    return ADFMetrics(
        angles=bin_centers,
        adf=adf,
        dominant_angle=dominant_angle,
        dominant_angle_degree=dominant_angle_deg,
    )
