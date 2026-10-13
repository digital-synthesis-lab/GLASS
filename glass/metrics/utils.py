"""Utility functions for structural metrics."""

import json
from pathlib import Path
from typing import Union, Dict, Any

from glass.metrics.core import StructuralMetrics


def load_metrics_from_json(filepath: Union[str, Path], structure_name: str = None) -> StructuralMetrics:
    """Load StructuralMetrics from a JSON file.
    
    Args:
        filepath: Path to JSON file
        structure_name: If the JSON contains multiple structures (CLI output format),
                       specify which structure to load. If None and only one structure
                       exists, that structure is used.
    
    Returns:
        StructuralMetrics object
    """
    import numpy as np
    from glass.metrics.core import (
        PDFMetrics, ADFMetrics, CoordinationMetrics,
        DihedralMetrics, RingMetrics,
    )
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Handle CLI output format: {"metadata": {...}, "structures": {"name": {...}}}
    if 'structures' in data:
        structures = data['structures']
        if len(structures) == 0:
            raise ValueError("No structures found in JSON file")
        
        if structure_name is not None:
            if structure_name not in structures:
                available = list(structures.keys())
                raise ValueError(f"Structure '{structure_name}' not found. Available: {available}")
            data = structures[structure_name]
        elif len(structures) == 1:
            data = list(structures.values())[0]
        else:
            # Multiple structures, use the first one
            data = list(structures.values())[0]
    
    # Handle error structures
    if 'error' in data:
        raise ValueError(f"Cannot load metrics with error: {data['error']}")
    
    # Build PDF metrics
    pdf_data = data['pdf']
    pdf = PDFMetrics(
        r=np.array(pdf_data['r']),
        g_r=np.array(pdf_data['g_r']),
        first_peak_position=pdf_data.get('first_peak_position'),
        first_peak_height=pdf_data.get('first_peak_height'),
        first_minima_position=pdf_data.get('first_minima_position'),
        first_minima_height=pdf_data.get('first_minima_height'),
        coord_cutoff=pdf_data.get('coord_cutoff'),
    )
    
    # Build ADF metrics
    adf_data = data['adf']
    adf = ADFMetrics(
        angles=np.array(adf_data['angles']),
        adf=np.array(adf_data['adf']),
        dominant_angle=adf_data.get('dominant_angle'),
        dominant_angle_degree=adf_data.get('dominant_angle_degree'),
    )
    
    # Build Coordination metrics
    coord_data = data['coordination']
    coordination = CoordinationMetrics(
        coordination_numbers=np.array(coord_data['coordination_numbers']),
        mean_coordination=coord_data['mean_coordination'],
        std_coordination=coord_data['std_coordination'],
        coordination_histogram=np.array(coord_data['coordination_histogram']),
        histogram_bins=np.array(coord_data['histogram_bins']),
    )
    
    # Build optional metrics
    dihedrals = None
    if 'dihedrals' in data and data['dihedrals']:
        dih_data = data['dihedrals']
        dihedrals = DihedralMetrics(
            dihedral_angles=np.array(dih_data['dihedral_angles']),
            dihedral_histogram=np.array(dih_data['dihedral_histogram']),
            histogram_bins=np.array(dih_data['histogram_bins']),
            mean_dihedral=dih_data['mean_dihedral'],
            std_dihedral=dih_data['std_dihedral'],
        )
    
    rings = None
    if 'rings' in data and data['rings']:
        ring_data = data['rings']
        rings = RingMetrics(
            ring_lengths=np.array(ring_data['ring_lengths']),
            ring_counts=np.array(ring_data['ring_counts']),
            ring_fractions=np.array(ring_data['ring_fractions']),
            total_rings=ring_data['total_rings'],
            cutoff=ring_data['cutoff'],
            maxlength=ring_data['maxlength'],
        )

    tersoff = None
    if 'tersoff' in data and data['tersoff']:
        from glass.metrics.tersoff import TersoffMetrics

        tersoff = TersoffMetrics.from_dict(data['tersoff'])

    return StructuralMetrics(
        n_atoms=data['n_atoms'],
        composition=data['composition'],
        cell=data['cell'],
        density=data['density'],
        pdf=pdf,
        adf=adf,
        coordination=coordination,
        dihedrals=dihedrals,
        rings=rings,
        tersoff=tersoff,
    )
