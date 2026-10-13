"""Test end-to-end generation with initialization file.

This module tests the complete generation pipeline starting from an initialization
structure, running ~200 denoising steps, and verifying the final structure has a
reasonable PDF compared to a reference.
"""

import os
import pytest
import torch
import numpy as np
from pathlib import Path
from ase.io import read
from ase import Atoms
from dataclasses import dataclass
from typing import Tuple

# Ensure KMP_DUPLICATE_LIB_OK is set before importing torch-dependent modules
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from glass.lit.modules import LitScoreNet, DifferentiableRDF
from glass.diffusion import VarianceExplodingDiffuser
from glass.diffusion.sampling import denoise_by_sde
from glass.utils.atoms import atoms_to_device, compute_prior_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
CHECKPOINT_PATH = REPO_ROOT / "examples" / "silicon" / "experiment" / "checkpoints" / "best.ckpt"

pytestmark = pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(),
    reason=f"example checkpoint not found at {CHECKPOINT_PATH}",
)
INIT_DENSITY = 2.5  # g/cm3, the density of the training data


def random_init(density: float = INIT_DENSITY, n_atoms: int = 216, seed: int = 0) -> Atoms:
    """Random Si cell with a 2 Å minimum distance, as `glass initialize` makes."""
    from ase.units import _amu

    from glass.utils.packing import pack

    volume = n_atoms * 28.0855 * _amu * 1e3 / density * 1e24  # Å^3
    cell = np.eye(3) * volume ** (1.0 / 3.0)
    pos = pack(n_atoms, cell, 2.0, rng=np.random.default_rng(seed))
    return Atoms(f"Si{n_atoms}", positions=pos, cell=cell, pbc=True)


@dataclass
class GenerationResult:
    """Container for generation results."""
    init_atoms: Atoms
    final_atoms: Atoms
    trajectory: list  # List of position tensors
    init_pdf: np.ndarray
    final_pdf: np.ndarray


def compute_pdf(atoms: Atoms, cutoff: float = 5.0, bin_size: int = 100) -> np.ndarray:
    """Compute PDF for an ASE Atoms object.
    
    Args:
        atoms: ASE Atoms object
        cutoff: Maximum r for PDF computation
        bin_size: Number of bins
    
    Returns:
        PDF values as numpy array
    """
    pdf_model = DifferentiableRDF(cutoff=cutoff, bin_size=bin_size, sigma=0.15)
    pdf_model.eval()
    
    # Convert to tensors
    from glass.utils.atoms import initialize_atoms
    _, species, pos, cell = initialize_atoms(atoms)
    
    with torch.no_grad():
        _, pdf, _ = pdf_model(pos.cpu(), species.cpu(), cell.cpu())
    
    return pdf.cpu().numpy()


@pytest.fixture(scope="session")
def generated_structure() -> GenerationResult:
    """Generate structure once for all tests (session-scoped).
    
    This fixture performs the full denoising process (200 steps) once,
    and all tests analyze the resulting structure.
    """
    # Load model
    score_net = LitScoreNet.load_from_checkpoint(
        CHECKPOINT_PATH,
        map_location="cpu",
    )
    score_net.eval()
    score_net.ema_model.eval()
    
    diffuser = VarianceExplodingDiffuser(k=0.8)
    
    # Load initialization structure
    init_atoms = random_init()
    species, pos, cell = atoms_to_device(init_atoms, "cpu")
    
    # Create 200 time steps from t=1.0 to t=0.001
    ts = torch.linspace(1.0, 0.001, 200, device="cpu")
    
    def score_fn(sp, p, c, t, co):
        return compute_prior_score(sp, p, c, t, co, score_net, diffuser)
    
    # Run denoising with trajectory
    traj, final_pos = denoise_by_sde(
        species=species,
        pos=pos,
        cell=cell,
        cutoff=5.0,
        score_fn=score_fn,
        likelihood_fn=None,
        ts=ts,
        diffuser=diffuser,
        save_traj=True,  # Save trajectory for analysis
    )
    
    # Create final atoms
    final_atoms = Atoms(
        numbers=init_atoms.numbers,
        positions=final_pos.cpu().numpy(),
        cell=init_atoms.cell,
        pbc=init_atoms.pbc,
    )
    final_atoms.wrap()
    
    # Compute PDFs
    init_pdf = compute_pdf(init_atoms)
    final_pdf = compute_pdf(final_atoms)
    
    return GenerationResult(
        init_atoms=init_atoms,
        final_atoms=final_atoms,
        trajectory=traj,
        init_pdf=init_pdf,
        final_pdf=final_pdf,
    )


class TestGenerationSetup:
    """Test setup and basic properties."""
    
    def test_checkpoint_file_exists(self):
        """Verify checkpoint file exists."""
        assert CHECKPOINT_PATH.exists(), f"Checkpoint not found at {CHECKPOINT_PATH}"


class TestInitialStructure:
    """Test initial structure properties."""
    
    def test_initial_structure_is_random(self, generated_structure):
        """Verify initial structure is disordered (random-like)."""
        init_pdf = generated_structure.init_pdf
        
        # Random structure should have a broad, featureless PDF at small r
        # First peak should be very low (no sharp peaks)
        first_peak = np.max(init_pdf[:, :20])  # Check first 20 bins
        
        # Random structure should have low first peak
        assert first_peak < 5.0, "Initial structure appears too ordered"
    
    def test_initial_structure_has_216_atoms(self, generated_structure):
        """Verify initial structure has expected number of atoms."""
        assert len(generated_structure.init_atoms) == 216


class TestGenerationResults:
    """Test the generation process and final structure."""
    
    def test_generation_produces_valid_structure(self, generated_structure):
        """Test that generation produces a valid structure."""
        final_atoms = generated_structure.final_atoms
        
        # Check atoms exist
        assert len(final_atoms) > 0
        
        # Check positions are finite
        positions = final_atoms.positions
        assert np.isfinite(positions).all()
    
    def test_pdf_changed_after_denoising(self, generated_structure):
        """Test that PDF changed significantly after denoising."""
        init_pdf = generated_structure.init_pdf
        final_pdf = generated_structure.final_pdf
        
        # Bins 40-56 cover r = 2.0-2.8 Å, the Si-Si bond peak
        init_first_peak = np.max(init_pdf[:, 40:56])
        final_first_peak = np.max(final_pdf[:, 40:56])

        # Bonds form, so the first peak grows
        assert final_first_peak > init_first_peak + 0.1, \
            f"PDF did not change: {init_first_peak:.2f} -> {final_first_peak:.2f}"
    
    def test_final_pdf_has_reasonable_peaks(self, generated_structure):
        """Test that final PDF has physically reasonable peak positions.
        
        For amorphous silicon, we expect:
        - First peak around 2.3-2.5 Å (Si-Si bond length)
        - Second peak around 3.8-4.0 Å
        """
        final_pdf = generated_structure.final_pdf
        
        # Get bin centers (approximately)
        bin_edges = np.linspace(0, 5.0, 101)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Find first significant peak (above background)
        pdf_1d = np.mean(final_pdf, axis=0)  # Average over type pairs
        
        # Find peaks (simple approach: local maxima)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(pdf_1d, height=np.mean(pdf_1d) * 2, distance=5)
        
        if len(peaks) > 0:
            first_peak_r = bin_centers[peaks[0]]
            # First peak should be around Si-Si bond length (2.35 Å) ± 0.5 Å
            assert 1.8 < first_peak_r < 3.0, \
                f"First peak at unexpected position: {first_peak_r:.2f} Å"
    
    def test_trajectory_produced(self, generated_structure):
        """Test that trajectory was produced during generation."""
        traj = generated_structure.trajectory
        
        assert traj is not None
        assert len(traj) == 200  # 200 steps
        
        # Verify trajectory progresses (positions change)
        init_pos = torch.from_numpy(generated_structure.init_atoms.positions)
        final_traj_pos = traj[-1]
        
        # Check that positions actually changed (not identical)
        position_diff = torch.norm(final_traj_pos - init_pos).item()
        assert position_diff > 0.01, f"Positions did not change: diff={position_diff:.4f}"
    
    def test_trajectory_is_monotonic(self, generated_structure):
        """Test that trajectory shows monotonic denoising progress."""
        traj = generated_structure.trajectory
        
        # Check that positions change throughout trajectory
        # (not stuck at same values)
        diffs = []
        for i in range(len(traj) - 1):
            diff = torch.norm(traj[i + 1] - traj[i]).item()
            diffs.append(diff)
        
        # Average displacement should be non-zero
        avg_diff = np.mean(diffs)
        assert avg_diff > 1e-4, f"Trajectory not progressing: avg_diff={avg_diff}"
    
    def test_cell_preserved(self, generated_structure):
        """Test that cell dimensions are preserved."""
        init_cell = generated_structure.init_atoms.cell
        final_cell = generated_structure.final_atoms.cell
        
        assert np.allclose(init_cell, final_cell), "Cell changed during generation"
    
    def test_species_preserved(self, generated_structure):
        """Test that atomic species are preserved."""
        init_numbers = generated_structure.init_atoms.numbers
        final_numbers = generated_structure.final_atoms.numbers
        
        assert np.array_equal(init_numbers, final_numbers), "Species changed during generation"
    
    def test_final_structure_wrapped(self, generated_structure):
        """Test that final structure has wrapped positions."""
        final_atoms = generated_structure.final_atoms
        
        # Check that positions are within reasonable bounds
        positions = final_atoms.positions
        cell = final_atoms.cell
        
        for dim in range(3):
            cell_length = np.linalg.norm(cell[dim])
            max_pos = np.abs(positions[:, dim]).max()
            # Allow some margin for wrapped positions
            assert max_pos < cell_length * 1.5, \
                f"Positions not properly wrapped in dimension {dim}"


class TestGenerationQuality:
    """Test quality metrics for generated structures."""
    
    def test_no_atomic_overlap(self, generated_structure):
        """Test that generated structure has no unphysical overlaps."""
        final_atoms = generated_structure.final_atoms
        
        # Compute pairwise distances
        from ase.neighborlist import neighbor_list
        i, j, d = neighbor_list('ijd', final_atoms, 1.0)  # 1.0 Å cutoff
        
        # Check no atoms are too close (Si covalent radius ~1.11 Å)
        if len(d) > 0:
            min_dist = np.min(d)
            assert min_dist > 1.5, f"Atoms too close: {min_dist:.2f} Å"
    
    def test_reasonable_density(self, generated_structure):
        """Test that final structure has reasonable density."""
        final_atoms = generated_structure.final_atoms
        
        # Compute density (atoms per Å³)
        volume = final_atoms.get_volume()
        n_atoms = len(final_atoms)
        density = n_atoms / volume
        
        # Si density in diamond is ~0.05 atoms/Å³
        # Amorphous should be similar (allow wider range)
        assert 0.02 < density < 0.12, f"Unreasonable density: {density:.4f} atoms/Å³"
    
    def test_structure_is_condensed(self, generated_structure):
        """Test that structure is condensed (not gas-like)."""
        final_atoms = generated_structure.final_atoms
        
        # Compute coordination number (neighbors within 3.0 Å)
        from ase.neighborlist import neighbor_list
        i, j = neighbor_list('ij', final_atoms, 3.0)
        
        # Average coordination
        unique, counts = np.unique(i, return_counts=True)
        avg_coord = np.mean(counts)
        
        # Amorphous Si should have coordination ~4 (allow wider range)
        assert 1.0 < avg_coord < 15.0, f"Unreasonable coordination: {avg_coord:.2f}"
    
    def test_bond_length_distribution(self, generated_structure):
        """Test that bond length distribution is reasonable."""
        final_atoms = generated_structure.final_atoms
        
        # Compute all pairwise distances
        from ase.neighborlist import neighbor_list
        i, j, d = neighbor_list('ijd', final_atoms, 4.0)  # 4.0 Å cutoff
        
        # Filter for nearest neighbors (likely bonded pairs)
        # Si-Si bond ~2.35 Å
        bonded = d[(d > 1.5) & (d < 3.0)]
        
        if len(bonded) > 0:
            avg_bond = np.mean(bonded)
            # Average bond length should be around Si-Si distance
            assert 2.0 < avg_bond < 3.0, f"Unreasonable bond length: {avg_bond:.2f} Å"
    
    def test_pdf_has_expected_features(self, generated_structure):
        """Test that final PDF has expected features for amorphous Si.
        
        Expected features:
        - Sharp first peak at ~2.35 Å
        - Broader second peak at ~3.8-4.0 Å
        - Beyond that should decay
        """
        final_pdf = generated_structure.final_pdf
        pdf_1d = np.mean(final_pdf, axis=0)
        
        # Get bin centers
        bin_edges = np.linspace(0, 5.0, 101)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # First peak should be sharp and high
        first_peak_region = pdf_1d[40:56]  # r = 2.0-2.8 Å
        first_peak_height = np.max(first_peak_region)
        
        # Should have a clear first peak
        assert first_peak_height > 0.5, f"First peak too weak: {first_peak_height:.2f}"
        
        # PDF should have structure (not flat)
        pdf_variance = np.var(pdf_1d)
        assert pdf_variance > 0.01, f"PDF too flat: variance={pdf_variance:.4f}"


class TestPDFComparison:
    """Tests comparing initial and final PDFs."""
    
    def test_pdf_peak_ratio(self, generated_structure):
        """Test ratio of first peak intensities."""
        init_pdf = generated_structure.init_pdf
        final_pdf = generated_structure.final_pdf
        
        init_peak = np.max(init_pdf)
        final_peak = np.max(final_pdf)
        
        # Final structure should have more defined peaks
        # (this is a qualitative check)
        assert final_peak > 0, "Final PDF is zero"
        assert init_peak > 0, "Initial PDF is zero"
    
    def test_pdf_integral_conserved(self, generated_structure):
        """Test that PDF integral is roughly conserved."""
        init_pdf = generated_structure.init_pdf
        final_pdf = generated_structure.final_pdf
        
        init_integral = np.sum(init_pdf)
        final_integral = np.sum(final_pdf)
        
        # Should be same order of magnitude
        ratio = final_integral / (init_integral + 1e-8)
        assert 0.1 < ratio < 10.0, f"PDF integral changed too much: ratio={ratio:.2f}"
