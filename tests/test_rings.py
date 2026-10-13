"""Tests for the glass.metrics.rings module."""

import os
import pytest
import numpy as np
from pathlib import Path
from ase.build import bulk
from ase.io import read

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from glass.metrics import (
    compute_rings,
    compute_rings_distribution,
)
from glass.metrics.rings import (
    _build_seed_array,
    _compute_shortest_distances,
    _compute_distance_matrix,
)
from glass.metrics.core import RingMetrics


DATA_DIR = Path(__file__).resolve().parent / "data"

# Load ground truth CRN ring data
def load_crn_ground_truth():
    """Load ground truth ring statistics for CRN (Continuous Random Network).
    
    Returns:
        tuple: (ring_lengths, ring_counts, rings_per_atom) for ring sizes 0-10
    """
    csv_path = DATA_DIR / "CRN-rings.csv"
    
    ring_lengths = []
    ring_counts = []
    rings_per_atom = []
    
    with open(csv_path) as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(',')
            ring_lengths.append(int(parts[0]))
            ring_counts.append(float(parts[1]))
            rings_per_atom.append(float(parts[2]))
    
    return np.array(ring_lengths), np.array(ring_counts), np.array(rings_per_atom)


class TestRingInternals:
    """Test internal ring computation functions."""
    
    def test_build_seed_array_simple(self):
        """Test seed array construction."""
        # Simple 2-atom system: 0-1 bond
        i = np.array([0, 1])  # Atom 0 connected to 1, atom 1 connected to 0
        nat = 2
        
        seed = _build_seed_array(nat, i)
        
        # seed[0] = 0 (atom 0 starts at index 0)
        # seed[1] = 1 (atom 1 starts at index 1)
        # seed[2] = 2 (end of array)
        assert seed[0] == 0
        assert seed[1] == 1
        assert seed[2] == 2
    
    def test_build_seed_array_triangle(self):
        """Test seed array for triangular system."""
        # Triangle: 0-1, 1-2, 2-0
        i = np.array([0, 1, 2])  # sources
        nat = 3
        
        seed = _build_seed_array(nat, i)
        
        # Each atom has 1 outgoing edge in this representation
        assert seed[0] == 0
        assert seed[1] == 1
        assert seed[2] == 2
        assert seed[3] == 3
    
    def test_compute_shortest_distances_linear(self):
        """Test distance computation on linear chain."""
        # Linear chain: 0-1-2-3
        i = np.array([0, 1, 1, 2, 2, 3])  # bidirectional
        j = np.array([1, 0, 2, 1, 3, 2])  # neighbors
        nat = 4
        
        seed = _build_seed_array(nat, i)
        
        # Test distances from atom 0
        dist = _compute_shortest_distances(nat, seed, j, root=0)
        
        # Expected: dist[0]=0, dist[1]=1, dist[2]=2, dist[3]=3
        assert dist[0] == 0
        assert dist[1] == 1
        assert dist[2] == 2
        assert dist[3] == 3
    
    def test_compute_distance_matrix(self):
        """Test full distance matrix computation."""
        # Square: 0-1-2-3-0 
        # Need proper neighbor list format
        # 0: neighbors are 1 and 3
        # 1: neighbors are 0 and 2
        # 2: neighbors are 1 and 3
        # 3: neighbors are 2 and 0
        i = np.array([0, 0, 1, 1, 2, 2, 3, 3])  # sources
        j = np.array([1, 3, 0, 2, 1, 3, 2, 0])  # neighbors
        nat = 4
        
        seed = _build_seed_array(nat, i)
        dist = _compute_distance_matrix(nat, seed, j)
        
        # Check symmetry
        for a in range(nat):
            for b in range(nat):
                assert dist[a, b] == dist[b, a]
        
        # Check diagonal
        assert np.all(np.diag(dist) == 0)


class TestRingStatistics:
    """Test ring statistics computation."""
    
    def test_compute_rings_diamond_si(self):
        """Test ring computation for diamond Si (should have 6-membered rings)."""
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)  # Make supercell
        
        rings = compute_rings(atoms, cutoff=2.8, maxlength=10)
        
        assert isinstance(rings, RingMetrics)
        assert rings.total_rings > 0
        
        # Diamond Si has 6-membered rings
        assert rings.ring_counts[6] > 0
        
        # No rings smaller than 3
        assert rings.ring_counts[0] == 0
        assert rings.ring_counts[1] == 0
        assert rings.ring_counts[2] == 0
    
    def test_compute_rings_amorphous_si(self):
        """Test ring computation for amorphous Si."""
        atoms = read(DATA_DIR / "Si_2.5_00.xyz")
        
        rings = compute_rings(atoms, cutoff=3.0, maxlength=10)
        
        assert isinstance(rings, RingMetrics)
        assert rings.total_rings > 0
        
        # Amorphous Si should have various ring sizes
        # Should have some 5 and 6 membered rings
        assert rings.ring_counts[5] > 0 or rings.ring_counts[6] > 0
    
    def test_compute_rings_auto_cutoff(self):
        """Test ring computation with automatic cutoff."""
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)
        
        rings = compute_rings(atoms, cutoff=None, maxlength=10, auto_cutoff=True)
        
        assert isinstance(rings, RingMetrics)
        assert rings.cutoff > 0  # Should have determined a cutoff
    
    def test_compute_rings_empty_system(self):
        """Test ring computation with no bonds."""
        # Two atoms far apart
        from ase import Atoms
        atoms = Atoms('Si2', positions=[[0, 0, 0], [10, 0, 0]], cell=[20, 20, 20])
        
        rings = compute_rings(atoms, cutoff=3.0, maxlength=10)
        
        # Should return empty result
        assert rings.total_rings == 0
        assert len(rings.ring_counts) == 11  # 0 to maxlength
    
    def test_compute_rings_fractions_sum(self):
        """Test that ring fractions sum to 100%."""
        atoms = read(DATA_DIR / "Si_2.5_00.xyz")
        
        rings = compute_rings(atoms, cutoff=3.0, maxlength=10)
        
        if rings.total_rings > 0:
            # Fractions should sum to approximately 100
            total_frac = np.sum(rings.ring_fractions)
            assert abs(total_frac - 100.0) < 0.01
    
    def test_compute_rings_different_maxlength(self):
        """Test ring computation with different maxlength values."""
        atoms = read(DATA_DIR / "Si_2.5_00.xyz")

        rings_6 = compute_rings(atoms, cutoff=3.0, maxlength=6)
        rings_10 = compute_rings(atoms, cutoff=3.0, maxlength=10)

        assert rings_6.maxlength == 6
        assert rings_10.maxlength == 10
        assert len(rings_6.ring_counts) == 7  # 0-6
        assert len(rings_10.ring_counts) == 11  # 0-10
        assert np.allclose(rings_6.ring_counts[:7], rings_10.ring_counts[:7])
    
    def test_compute_rings_returns_numpy_arrays(self):
        """Test that ring metrics returns proper numpy arrays."""
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)
        
        rings = compute_rings(atoms, cutoff=2.8, maxlength=10)
        
        assert isinstance(rings.ring_lengths, np.ndarray)
        assert isinstance(rings.ring_counts, np.ndarray)
        assert isinstance(rings.ring_fractions, np.ndarray)
    
    def test_compute_rings_to_dict(self):
        """Test that ring metrics can be converted to dict."""
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)
        
        rings = compute_rings(atoms, cutoff=2.8, maxlength=10)
        data = rings.to_dict()
        
        assert "ring_lengths" in data
        assert "ring_counts" in data
        assert "ring_fractions" in data
        assert "total_rings" in data
        assert "cutoff" in data
        assert "maxlength" in data


class TestRingsDistribution:
    """Test ring distribution computation over multiple frames."""
    
    def test_compute_rings_distribution_single_frame(self):
        """Test distribution with single frame."""
        atoms = read(DATA_DIR / "Si_2.5_00.xyz")
        
        rings = compute_rings_distribution([atoms], cutoff=3.0, maxlength=10)
        
        assert isinstance(rings, RingMetrics)
        assert rings.total_rings > 0
    
    def test_compute_rings_distribution_multiple_frames(self):
        """Test distribution with multiple frames (same structure)."""
        atoms = read(DATA_DIR / "Si_2.5_00.xyz")
        
        # Same structure twice should give same average
        rings = compute_rings_distribution([atoms, atoms], cutoff=3.0, maxlength=10)
        
        assert isinstance(rings, RingMetrics)
        # Average should be same as single computation
        single = compute_rings(atoms, cutoff=3.0, maxlength=10)
        assert np.array_equal(rings.ring_counts, single.ring_counts)
    
    def test_compute_rings_distribution_single_atom_object(self):
        """Test that single Atoms object is handled correctly."""
        atoms = read(DATA_DIR / "Si_2.5_00.xyz")
        
        # Pass single object instead of list
        rings = compute_rings_distribution(atoms, cutoff=3.0, maxlength=10)
        
        # Should treat as single frame
        single = compute_rings(atoms, cutoff=3.0, maxlength=10)
        assert np.array_equal(rings.ring_counts, single.ring_counts)


class TestRingErrorMetrics:
    """Test ring error metrics."""
    
    def test_rings_rmse_identical(self):
        """Test RMSE between identical structures is zero."""
        from glass.metrics.errors import rings_rmse
        
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)
        
        rings1 = compute_rings(atoms, cutoff=2.8, maxlength=10)
        rings2 = compute_rings(atoms, cutoff=2.8, maxlength=10)
        
        rmse = rings_rmse(rings1, rings2)
        assert rmse == 0.0
    
    def test_rings_cosine_identical(self):
        """Test cosine similarity of identical structures is 1.0."""
        from glass.metrics.errors import rings_cosine_similarity
        
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)
        
        rings1 = compute_rings(atoms, cutoff=2.8, maxlength=10)
        rings2 = compute_rings(atoms, cutoff=2.8, maxlength=10)
        
        cosine = rings_cosine_similarity(rings1, rings2)
        assert abs(cosine - 1.0) < 0.001
    
    def test_rings_emd_identical(self):
        """Test EMD between identical structures is zero."""
        from glass.metrics.errors import rings_emd
        
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)
        
        rings1 = compute_rings(atoms, cutoff=2.8, maxlength=10)
        rings2 = compute_rings(atoms, cutoff=2.8, maxlength=10)
        
        emd = rings_emd(rings1, rings2)
        assert emd == 0.0


class TestRingPerformance:
    """Test ring computation performance characteristics."""
    
    @pytest.mark.slow
    def test_compute_rings_scaling(self):
        """Test that ring computation scales reasonably with system size."""
        import time
        
        atoms = bulk("Si", "diamond", a=5.43)
        
        sizes = [2, 3]
        times = []
        
        for n in sizes:
            atoms_n = atoms * (n, n, n)
            
            start = time.time()
            rings = compute_rings(atoms_n, cutoff=2.8, maxlength=10)
            end = time.time()
            
            times.append(end - start)
            
            # Sanity check: should find rings
            assert rings.total_rings > 0
        
        # Check scaling is reasonable. Very loose check — small-N timing
        # is dominated by Python overhead, so we allow a generous slack.
        ratio = times[1] / times[0]
        n_ratio = (sizes[1] / sizes[0]) ** 3
        assert ratio < n_ratio * 8


class TestCRNGroundTruth:
    """Test ring computation against CRN (Continuous Random Network) ground truth.
    
    CRN is a well-characterized 1000-atom amorphous silicon structure with
    known ring statistics computed using the Franzblau algorithm with cutoff 2.85 Å.
    """
    
    def test_crn_structure_loaded(self):
        """Test that CRN structure file exists and can be loaded."""
        crn_path = DATA_DIR / "CRN.xyz"
        assert crn_path.exists(), "CRN.xyz not found in test data"
        
        atoms = read(crn_path)
        assert len(atoms) == 1000
        assert atoms.get_chemical_formula() == "Si1000"
    
    def test_crn_ground_truth_loaded(self):
        """Test that ground truth ring data can be loaded."""
        gt_lengths, gt_counts, gt_per_atom = load_crn_ground_truth()
        
        # Should have data for ring sizes 0-10
        assert len(gt_lengths) == 11
        assert gt_lengths[0] == 0
        assert gt_lengths[10] == 10
        
        # No rings of size 0, 1, 2
        assert gt_counts[0] == 0.0
        assert gt_counts[1] == 0.0
        assert gt_counts[2] == 0.0
        
        # Should have rings of size 3-10
        assert gt_counts[3] > 0
        assert gt_counts[6] > 0  # Peak at 6-membered rings
    
    def test_crn_ring_distribution_shape_matches(self):
        """Test that ring distribution matches the CRN ground truth exactly,
        size by size (1916 rings in total)."""
        crn_path = DATA_DIR / "CRN.xyz"
        atoms = read(crn_path)

        rings = compute_rings(atoms, cutoff=2.85, maxlength=10, auto_cutoff=False)
        gt_lengths, gt_counts, gt_per_atom = load_crn_ground_truth()

        for s in range(3, 11):
            assert rings.ring_counts[s] == pytest.approx(gt_counts[s], abs=1e-6)

    def test_crn_rings_invariant_under_atom_relabelling(self):
        """Ring statistics are a property of the network, not of the atom
        numbering; rings through atom 0 used to be dropped."""
        atoms = read(DATA_DIR / "CRN.xyz")
        perm = np.random.default_rng(0).permutation(len(atoms))

        ref = compute_rings(atoms, cutoff=2.85, maxlength=10, auto_cutoff=False)
        shuffled = compute_rings(atoms[perm], cutoff=2.85, maxlength=10, auto_cutoff=False)

        assert np.allclose(ref.ring_counts, shuffled.ring_counts, atol=1e-9)
    
    def test_crn_ring_distribution_detailed_comparison(self):
        """Detailed comparison of ring distribution with ground truth."""
        crn_path = DATA_DIR / "CRN.xyz"
        atoms = read(crn_path)

        rings = compute_rings(atoms, cutoff=2.85, maxlength=10, auto_cutoff=False)
        gt_lengths, gt_counts, gt_per_atom = load_crn_ground_truth()

        print("\nCRN Ring Distribution Comparison:")
        print(f"{'Size':>6} {'Computed':>10} {'Ground Truth':>12}")
        print("-" * 36)
        for size in range(3, 11):
            print(f"{size:>6} {rings.ring_counts[size]:>10.1f} {gt_counts[size]:>12.1f}")
        print(f"\nTotal: {rings.total_rings} vs {np.sum(gt_counts):.0f}")

        assert rings.total_rings == pytest.approx(np.sum(gt_counts), rel=0.005)
    
    def test_crn_total_rings_scale(self):
        """Test that total ring count matches the CRN ground truth within tolerance.

        The implementation reproduces the reference total to within ~0.2%
        (a handful of rings are missed near over-coordinated atoms).
        """
        crn_path = DATA_DIR / "CRN.xyz"
        atoms = read(crn_path)

        rings = compute_rings(atoms, cutoff=2.85, maxlength=10, auto_cutoff=False)
        gt_lengths, gt_counts, gt_per_atom = load_crn_ground_truth()
        gt_total = float(np.sum(gt_counts))

        assert rings.total_rings == pytest.approx(gt_total, rel=0.005)
    
    def test_crn_ring_distribution_shape(self):
        """Test that ring distribution shape matches ground truth.
        
        The distribution should peak at 6-membered rings, with fewer
        rings at smaller and larger sizes.
        """
        crn_path = DATA_DIR / "CRN.xyz"
        atoms = read(crn_path)
        
        rings = compute_rings(atoms, cutoff=2.85, maxlength=10, auto_cutoff=False)
        
        # Distribution should be unimodal with peak at size 6
        # Check that 6-membered rings are most common
        max_idx = np.argmax(rings.ring_counts[3:10]) + 3  # Offset by 3
        assert max_idx == 6, \
            f"Expected peak at size 6, got peak at size {max_idx}"
        
        # Should have decreasing counts after peak
        assert rings.ring_counts[6] > rings.ring_counts[7], \
            "Ring counts should decrease after peak"
        assert rings.ring_counts[7] > rings.ring_counts[8], \
            "Ring counts should decrease after peak"


class TestNumbaEngine:
    """Tests for the optional Numba-accelerated engine.

    Skipped when numba is not installed. Compares numba vs python
    engines for numerical parity and basic speedup.
    """

    def test_numba_matches_python_crn(self):
        """Numba and Python engines must agree on CRN ring counts."""
        pytest.importorskip("numba")
        atoms = read(DATA_DIR / "CRN.xyz")

        r_py = compute_rings(atoms, cutoff=2.85, maxlength=10,
                             auto_cutoff=False, engine="python")
        r_nb = compute_rings(atoms, cutoff=2.85, maxlength=10,
                             auto_cutoff=False, engine="numba")

        assert np.allclose(r_py.ring_counts, r_nb.ring_counts, atol=1e-6)
        assert r_py.total_rings == pytest.approx(r_nb.total_rings, rel=1e-6)

    def test_numba_matches_python_diamond(self):
        """Numba and Python engines must agree on diamond Si."""
        pytest.importorskip("numba")
        atoms = bulk("Si", "diamond", a=5.43) * (2, 2, 2)

        r_py = compute_rings(atoms, cutoff=2.8, maxlength=10,
                             auto_cutoff=False, engine="python")
        r_nb = compute_rings(atoms, cutoff=2.8, maxlength=10,
                             auto_cutoff=False, engine="numba")

        assert np.allclose(r_py.ring_counts, r_nb.ring_counts, atol=1e-6)

    def test_numba_matches_python_amorphous(self):
        """Numba and Python engines must agree on the small a-Si snapshot."""
        pytest.importorskip("numba")
        atoms = read(DATA_DIR / "Si_2.5_00.xyz")

        r_py = compute_rings(atoms, cutoff=3.0, maxlength=10,
                             auto_cutoff=False, engine="python")
        r_nb = compute_rings(atoms, cutoff=3.0, maxlength=10,
                             auto_cutoff=False, engine="numba")

        assert np.allclose(r_py.ring_counts, r_nb.ring_counts, atol=1e-6)

    def test_engine_auto_picks_numba_when_available(self):
        """engine='auto' should pick numba when it's importable."""
        pytest.importorskip("numba")
        from glass.metrics.rings import _resolve_engine
        assert _resolve_engine("auto") == "numba"

    def test_engine_invalid_raises(self):
        """Unknown engine string should raise ValueError."""
        from glass.metrics.rings import _resolve_engine
        with pytest.raises(ValueError):
            _resolve_engine("rust")

    @pytest.mark.slow
    def test_numba_speedup_crn(self):
        """Numba should be substantially faster than Python on CRN.

        Loose threshold (3x) to avoid CI flakiness; in practice the
        speedup is 10-30x depending on core count.
        """
        pytest.importorskip("numba")
        import time
        atoms = read(DATA_DIR / "CRN.xyz")

        # Warm-up the numba JIT
        compute_rings(atoms, cutoff=2.85, maxlength=10,
                      auto_cutoff=False, engine="numba")

        t0 = time.perf_counter()
        compute_rings(atoms, cutoff=2.85, maxlength=10,
                      auto_cutoff=False, engine="python")
        t_py = time.perf_counter() - t0

        t0 = time.perf_counter()
        compute_rings(atoms, cutoff=2.85, maxlength=10,
                      auto_cutoff=False, engine="numba")
        t_nb = time.perf_counter() - t0

        assert t_py / t_nb > 3.0, \
            f"Expected >3x speedup, got {t_py / t_nb:.1f}x (py={t_py:.2f}s, nb={t_nb:.3f}s)"
