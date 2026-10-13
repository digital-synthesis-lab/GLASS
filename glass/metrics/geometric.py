"""Geometric metrics: coordination numbers and dihedral angles."""

import numpy as np
from typing import Optional
from ase import Atoms
from ase.neighborlist import neighbor_list, NeighborList

from glass.metrics.core import CoordinationMetrics, DihedralMetrics


def compute_coordination(
    atoms: Atoms,
    cutoff: Optional[float] = None,
    auto_cutoff: bool = True,
) -> CoordinationMetrics:
    """Compute coordination number distribution for ASE Atoms.
    
    Args:
        atoms: ASE Atoms object
        cutoff: Cutoff for neighbor counting. If None and auto_cutoff=True,
               will be determined from PDF first minimum.
        auto_cutoff: If True and cutoff is None, use PDF first minimum
    
    Returns:
        CoordinationMetrics object with coordination statistics
    """
    # Determine cutoff
    if cutoff is None and auto_cutoff:
        from glass.metrics.structural import compute_pdf
        pdf_metrics = compute_pdf(atoms, cutoff=8.0, bin_size=100)
        cutoff = pdf_metrics.coord_cutoff
        if cutoff is None:
            cutoff = 3.0  # Default for Si
    elif cutoff is None:
        cutoff = 3.0
    
    # Compute coordination using ASE neighbor_list
    i, j = neighbor_list('ij', atoms, cutoff)
    
    # Count neighbors for each atom
    coordination = np.bincount(i, minlength=len(atoms))
    
    # Create histogram using bincount for integer bins
    # hist[n] = number of atoms with coordination number n
    hist = np.bincount(coordination)
    
    # Create bins: [0, 1, 2, ..., max_coord]
    # hist[0] = atoms with 0 neighbors, hist[1] = atoms with 1 neighbor, etc.
    nbrs = np.arange(len(hist))
    
    return CoordinationMetrics(
        coordination_numbers=coordination,
        mean_coordination=float(np.mean(coordination)),
        std_coordination=float(np.std(coordination)),
        coordination_histogram=hist,
        histogram_bins=nbrs,
    )


def compute_dihedrals(
    atoms: Atoms,
    bond_cutoff: float = 2.8,
    dihedral_bins: int = 180,
) -> Optional[DihedralMetrics]:
    """Compute dihedral angle distribution for ASE Atoms.
    
    Finds all torsion angles (dihedrals) defined by bonded quadruplets
    and returns their distribution.
    
    Args:
        atoms: ASE Atoms object
        bond_cutoff: Maximum distance for bond detection
        dihedral_bins: Number of bins for histogram
    
    Returns:
        DihedralMetrics object, or None if no dihedrals found
    """
    # Build neighbor list
    nl = NeighborList(
        [bond_cutoff / 2.0] * len(atoms),
        self_interaction=False,
        bothways=True,
    )
    nl.update(atoms)
    
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    
    dihedrals = []
    
    # Find dihedrals by looking for quadruplets i-j-k-l
    # where bonds exist: i-j, j-k, k-l
    for j in range(len(atoms)):
        neighbors_j, offsets_j = nl.get_neighbors(j)
        
        for idx_i, i in enumerate(neighbors_j):
            if i >= j:  # Avoid duplicates
                continue
                
            for idx_k, k in enumerate(neighbors_j):
                if k <= j:  # Avoid duplicates
                    continue
                if k == i:
                    continue
                
                # Get neighbors of k
                neighbors_k, offsets_k = nl.get_neighbors(k)
                
                for l in neighbors_k:
                    if l == j or l == i:
                        continue
                    if l < k:  # Avoid duplicates
                        continue
                    
                    # We have quadruplet i-j-k-l
                    # Calculate dihedral angle
                    p_i = positions[i] + offsets_j[idx_i] @ cell
                    p_j = positions[j]
                    p_k = positions[k] + offsets_k[np.where(neighbors_k == l)[0][0]] @ cell
                    p_l = positions[l]
                    
                    # Vectors
                    b1 = p_j - p_i
                    b2 = p_k - p_j
                    b3 = p_l - p_k
                    
                    # Normalize b2
                    b2_norm = np.linalg.norm(b2)
                    if b2_norm < 1e-10:
                        continue
                    b2 = b2 / b2_norm
                    
                    # Compute normal vectors
                    n1 = np.cross(b1, b2)
                    n2 = np.cross(b2, b3)
                    
                    # Normalize
                    n1_norm = np.linalg.norm(n1)
                    n2_norm = np.linalg.norm(n2)
                    if n1_norm < 1e-10 or n2_norm < 1e-10:
                        continue
                    
                    n1 = n1 / n1_norm
                    n2 = n2 / n2_norm
                    
                    # Compute angle
                    m1 = np.cross(n1, b2)
                    x = np.dot(n1, n2)
                    y = np.dot(m1, n2)
                    
                    angle = np.arctan2(y, x)
                    dihedrals.append(angle)
    
    if len(dihedrals) == 0:
        return None
    
    dihedrals = np.array(dihedrals)
    
    # Create histogram
    hist, bin_edges = np.histogram(
        dihedrals, bins=dihedral_bins, range=(-np.pi, np.pi)
    )
    
    return DihedralMetrics(
        dihedral_angles=dihedrals,
        dihedral_histogram=hist,
        histogram_bins=bin_edges,
        mean_dihedral=float(np.mean(dihedrals)),
        std_dihedral=float(np.std(dihedrals)),
    )
