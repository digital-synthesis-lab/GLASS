"""Tests for the glass.metrics module."""

import os
import pytest
import numpy as np
from pathlib import Path
from ase.build import bulk
from ase.io import read

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from glass.metrics import (
    compute_pdf,
    compute_adf,
    compute_coordination,
    compute_dihedrals,
    compute_all_metrics,
    StructuralMetrics,
)


DATA_DIR = Path(__file__).resolve().parent / "data"


class TestPDFMetrics:
    """Test PDF computation."""
    
    def test_compute_pdf_diamond_si(self):
        """Test PDF computation for diamond Si."""
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)  # Make supercell
        
        pdf = compute_pdf(atoms, cutoff=8.0, bin_size=100)
        
        assert pdf.r is not None
        assert pdf.g_r is not None
        assert len(pdf.r) == 100
        assert len(pdf.g_r) == 100
        
        # Should find first peak around Si-Si bond length
        assert pdf.first_peak_position is not None
        assert 2.0 < pdf.first_peak_position < 3.0
    
    def test_pdf_coord_cutoff_auto(self):
        """Test automatic cutoff detection from PDF."""
        # Use disordered structure for better PDF
        atoms = read(DATA_DIR / "Si_2.5_00.xyz")
        
        pdf = compute_pdf(atoms, cutoff=8.0)
        
        # Should detect coordination cutoff
        assert pdf.coord_cutoff is not None
        assert pdf.coord_cutoff > 0
        assert pdf.coord_cutoff > pdf.first_peak_position  # Cutoff should be after peak
    
    def test_pdf_approaches_unity_at_large_r(self):
        """Test that PDF approaches 1 as r → ∞."""
        atoms = read(DATA_DIR / "Si_2.5_00.xyz")
        
        pdf = compute_pdf(atoms, cutoff=8.0, bin_size=200)
        
        # Check tail region (r > 6 Å, well beyond coordination shells)
        mask_tail = (pdf.r > 6.0) & (pdf.r < 8.0)
        g_r_tail = pdf.g_r[mask_tail]
        
        # PDF should be close to 1 in bulk region
        assert len(g_r_tail) > 10  # Should have enough bins
        mean_tail = np.mean(g_r_tail)
        std_tail = np.std(g_r_tail)
        
        # Mean should be close to 1 (within 10%)
        assert 0.8 < mean_tail < 1.2, f"PDF tail mean {mean_tail:.3f} not close to 1"
        
        # Standard deviation should be moderate (not too noisy)
        assert std_tail < 0.5, f"PDF tail std {std_tail:.3f} too noisy"
    
    def test_pdf_no_edge_drop(self):
        """Test that PDF doesn't artificially drop near cutoff."""
        atoms = read(DATA_DIR / "Si_2.5_00.xyz")
        
        # Compute with different cutoffs
        pdf_6 = compute_pdf(atoms, cutoff=6.0, bin_size=120)
        pdf_8 = compute_pdf(atoms, cutoff=8.0, bin_size=160)
        
        # In overlapping region (r < 6), values should be similar
        # Find common r range
        r_common = pdf_6.r[pdf_6.r < 5.5]
        
        for r_val in r_common[::10]:  # Sample every 10th point
            idx_6 = np.argmin(np.abs(pdf_6.r - r_val))
            idx_8 = np.argmin(np.abs(pdf_8.r - r_val))
            
            g_6 = pdf_6.g_r[idx_6]
            g_8 = pdf_8.g_r[idx_8]
            
            # Values should be similar (within 20% relative difference)
            if g_6 > 0.1:  # Avoid comparing near-zero values
                relative_diff = abs(g_6 - g_8) / g_6
                assert relative_diff < 0.3, f"PDF varies with cutoff at r={r_val:.2f}: {g_6:.3f} vs {g_8:.3f}"
    
    def test_pdf_output_finite(self):
        """Test PDF values are finite."""
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)
        
        pdf = compute_pdf(atoms)
        
        assert np.isfinite(pdf.g_r).all()
        assert np.isfinite(pdf.r).all()


class TestADFMetrics:
    """Test ADF computation."""
    
    def test_compute_adf_diamond_si(self):
        """Test ADF for diamond Si."""
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)
        
        adf = compute_adf(atoms, cutoff=3.5)
        
        assert adf.angles is not None
        assert adf.adf is not None
        assert len(adf.angles) == 100
    
    def test_adf_auto_cutoff(self):
        """Test ADF with automatic cutoff from PDF."""
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)
        
        adf = compute_adf(atoms, cutoff=None, auto_cutoff=True)
        
        assert adf.adf is not None
        assert np.isfinite(adf.adf).all()


class TestCoordinationMetrics:
    """Test coordination number computation."""
    
    def test_coordination_diamond_si(self):
        """Test coordination for diamond Si (should be 4)."""
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)
        
        coord = compute_coordination(atoms, cutoff=2.5)
        
        # Diamond Si has coordination 4
        assert coord.mean_coordination > 3.5
        assert coord.mean_coordination < 4.5
        assert coord.std_coordination < 1.0  # Should be uniform
    
    def test_coordination_auto_cutoff(self):
        """Test coordination with automatic cutoff."""
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)
        
        coord = compute_coordination(atoms, cutoff=None, auto_cutoff=True)
        
        assert coord.coordination_numbers is not None
        assert len(coord.coordination_numbers) == len(atoms)


class TestAllMetrics:
    """Test comprehensive metrics computation."""
    
    def test_compute_all_metrics(self):
        """Test computing all metrics at once."""
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)
        
        metrics = compute_all_metrics(
            atoms,
            pdf_cutoff=8.0,
            auto_cutoff=True,
            include_dihedrals=True,
        )
        
        assert isinstance(metrics, StructuralMetrics)
        assert metrics.n_atoms == len(atoms)
        assert metrics.composition == "Si16"
        assert metrics.density > 0
        
        # Check all metrics computed
        assert metrics.pdf is not None
        assert metrics.adf is not None
        assert metrics.coordination is not None
    
    def test_metrics_to_dict(self):
        """Test conversion to dictionary."""
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)
        
        metrics = compute_all_metrics(
            atoms,
            include_dihedrals=False,
        )
        
        data = metrics.to_dict()
        
        assert "n_atoms" in data
        assert "composition" in data
        assert "pdf" in data
        assert "adf" in data
        assert "coordination" in data
    
    def test_metrics_to_json(self, tmp_path):
        """Test saving to JSON file."""
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)
        
        metrics = compute_all_metrics(
            atoms,
            include_dihedrals=False,
        )
        
        output_file = tmp_path / "metrics.json"
        metrics.to_json(output_file)
        
        assert output_file.exists()
        
        # Verify can load back
        import json
        with open(output_file) as f:
            data = json.load(f)
        
        assert data["n_atoms"] == len(atoms)

    def test_metrics_json_roundtrip_keeps_rings_and_tersoff(self, tmp_path):
        """load_metrics_from_json must restore the optional ring and Tersoff
        blocks, otherwise `glass compare --from-json` silently drops them."""
        from glass.metrics import load_metrics_from_json

        atoms = bulk("Si", "diamond", a=5.43) * (2, 2, 2)
        metrics = compute_all_metrics(
            atoms, include_dihedrals=False, include_rings=True, include_tersoff=True,
        )
        path = tmp_path / "metrics.json"
        metrics.to_json(path)
        loaded = load_metrics_from_json(path)

        np.testing.assert_allclose(loaded.rings.ring_counts, metrics.rings.ring_counts)
        assert loaded.tersoff.energy_per_atom == pytest.approx(
            metrics.tersoff.energy_per_atom
        )


class TestDihedralMetrics:
    """Test dihedral angle computation."""
    
    def test_dihedrals_diamond_si(self):
        """Test dihedral computation for diamond Si."""
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)
        
        dihedrals = compute_dihedrals(atoms, bond_cutoff=2.8)
        
        # Diamond Si has many dihedral angles
        assert dihedrals is not None
        assert len(dihedrals.dihedral_angles) > 0
        assert np.isfinite(dihedrals.dihedral_angles).all()
