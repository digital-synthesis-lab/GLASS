"""Core dataclasses for structural metrics.

This module defines the data structures used to store computed metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PDFMetrics:
    """Container for PDF (Pair Distribution Function) metrics.
    
    Attributes:
        r: Distance values (bin centers)
        g_r: PDF values g(r)
        first_peak_position: Position of first peak (typically Si-Si bond length)
        first_peak_height: Height of first peak
        first_minima_position: Position of first minimum (for coordination cutoff)
        first_minima_height: Height of first minimum
        coord_cutoff: Recommended coordination cutoff (first minimum position)
    """
    r: np.ndarray
    g_r: np.ndarray
    first_peak_position: Optional[float] = None
    first_peak_height: Optional[float] = None
    first_minima_position: Optional[float] = None
    first_minima_height: Optional[float] = None
    coord_cutoff: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "r": self.r.tolist(),
            "g_r": self.g_r.tolist(),
            "first_peak_position": self.first_peak_position,
            "first_peak_height": self.first_peak_height,
            "first_minima_position": self.first_minima_position,
            "first_minima_height": self.first_minima_height,
            "coord_cutoff": self.coord_cutoff,
        }


@dataclass
class ADFMetrics:
    """Container for ADF (Angular Distribution Function) metrics.
    
    Attributes:
        angles: Angle values in radians
        adf: ADF values
        dominant_angle: Most probable angle (peak position)
        dominant_angle_degree: Most probable angle in degrees
    """
    angles: np.ndarray
    adf: np.ndarray
    dominant_angle: Optional[float] = None
    dominant_angle_degree: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "angles": self.angles.tolist(),
            "adf": self.adf.tolist(),
            "dominant_angle": self.dominant_angle,
            "dominant_angle_degree": self.dominant_angle_degree,
        }


@dataclass
class CoordinationMetrics:
    """Container for coordination number metrics.
    
    Attributes:
        coordination_numbers: Array of coordination number for each atom
        mean_coordination: Mean coordination number
        std_coordination: Standard deviation of coordination
        coordination_histogram: Histogram of coordination numbers using bincount.
                              hist[n] = number of atoms with coordination n
        histogram_bins: Integer coordination numbers [0, 1, 2, ..., max_coord]
    """
    coordination_numbers: np.ndarray
    mean_coordination: float
    std_coordination: float
    coordination_histogram: np.ndarray = field(repr=False)
    histogram_bins: np.ndarray = field(repr=False)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "coordination_numbers": self.coordination_numbers.tolist(),
            "mean_coordination": self.mean_coordination,
            "std_coordination": self.std_coordination,
            "coordination_histogram": self.coordination_histogram.tolist(),
            "histogram_bins": self.histogram_bins.tolist(),
        }


@dataclass
class DihedralMetrics:
    """Container for dihedral angle metrics.
    
    Attributes:
        dihedral_angles: Array of dihedral angles in radians
        dihedral_histogram: Histogram of dihedral angles
        histogram_bins: Bin edges for histogram
        mean_dihedral: Mean dihedral angle
        std_dihedral: Standard deviation
    """
    dihedral_angles: np.ndarray
    dihedral_histogram: np.ndarray = field(repr=False)
    histogram_bins: np.ndarray = field(repr=False)
    mean_dihedral: float
    std_dihedral: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "dihedral_angles": self.dihedral_angles.tolist(),
            "dihedral_histogram": self.dihedral_histogram.tolist(),
            "histogram_bins": self.histogram_bins.tolist(),
            "mean_dihedral": self.mean_dihedral,
            "std_dihedral": self.std_dihedral,
        }


@dataclass
class RingMetrics:
    """Container for ring statistics metrics.
    
    Attributes:
        ring_lengths: Array of ring sizes (0 to maxlength)
        ring_counts: Count of rings for each size (float; fractional values
            arise only when averaging across frames)
        ring_fractions: Fractional distribution (percentage) for each ring size
        total_rings: Total number of rings found
        cutoff: Cutoff used for neighbor identification
        maxlength: Maximum ring size considered
    """
    ring_lengths: np.ndarray
    ring_counts: np.ndarray
    ring_fractions: np.ndarray
    total_rings: float
    cutoff: float
    maxlength: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "ring_lengths": self.ring_lengths.tolist(),
            "ring_counts": self.ring_counts.tolist(),
            "ring_fractions": self.ring_fractions.tolist(),
            "total_rings": self.total_rings,
            "cutoff": self.cutoff,
            "maxlength": self.maxlength,
        }


@dataclass
class StructuralMetrics:
    """Complete structural metrics for a single structure.
    
    Attributes:
        n_atoms: Number of atoms
        composition: Chemical formula
        cell: Unit cell parameters [a, b, c, alpha, beta, gamma]
        density: Number density (atoms/Å³)
        pdf: PDF metrics
        adf: ADF metrics
        coordination: Coordination metrics
        dihedrals: Dihedral metrics (optional)
        rings: Ring statistics metrics (optional)
    """
    n_atoms: int
    composition: str
    cell: List[float]
    density: float
    pdf: 'PDFMetrics'
    adf: 'ADFMetrics'
    coordination: 'CoordinationMetrics'
    dihedrals: Optional['DihedralMetrics'] = None
    rings: Optional['RingMetrics'] = None
    tersoff: Optional['TersoffMetrics'] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        from pathlib import Path
        import json
        
        result = {
            "n_atoms": self.n_atoms,
            "composition": self.composition,
            "cell": self.cell,
            "density": self.density,
            "pdf": self.pdf.to_dict(),
            "adf": self.adf.to_dict(),
            "coordination": self.coordination.to_dict(),
        }
        if self.dihedrals is not None:
            result["dihedrals"] = self.dihedrals.to_dict()
        if self.rings is not None:
            result["rings"] = self.rings.to_dict()
        if self.tersoff is not None:
            result["tersoff"] = self.tersoff.to_dict()
        return result
    
    def to_json(self, filepath: Union[str, Path], indent: int = 2) -> None:
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)
