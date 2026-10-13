import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import warnings

import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from click.testing import CliRunner

from glass.cli.initialize import initialize
from glass.utils.packing import (
    min_pairwise_distance,
    mc_soft_pack,
    pack,
    poisson_disk_pack,
)


# ---------------------------------------------------------------------------
# Packing unit tests
# ---------------------------------------------------------------------------


def test_poisson_disk_easy_regime():
    """N=216 Si in 15.91 Å cube, min_dist=2.0 Å: must succeed, <1 s,
    honour the min-distance."""
    rng = np.random.default_rng(0)
    cell = np.eye(3) * 15.91
    min_dist = 2.0

    t0 = time.perf_counter()
    pos = poisson_disk_pack(216, cell, min_dist, rng=rng, max_passes=200)
    dt = time.perf_counter() - t0

    assert pos is not None
    assert pos.shape == (216, 3)
    dmin = min_pairwise_distance(pos, cell)
    assert dmin >= min_dist - 1e-9, dmin
    assert dt < 1.0, f"slow: {dt:.3f}s"


def test_poisson_disk_reproducibility():
    cell = np.eye(3) * 15.91
    pos_a = poisson_disk_pack(216, cell, 2.0, rng=np.random.default_rng(7))
    pos_b = poisson_disk_pack(216, cell, 2.0, rng=np.random.default_rng(7))
    assert pos_a is not None and pos_b is not None
    assert np.array_equal(pos_a, pos_b)


def test_poisson_disk_pbc_respected():
    """Atoms placed near opposite faces must satisfy the minimum-image
    distance constraint."""
    rng = np.random.default_rng(1)
    # Tight cell where wrap-around matters.
    cell = np.eye(3) * 6.0
    pos = poisson_disk_pack(12, cell, 2.0, rng=rng, max_passes=50)
    assert pos is not None
    # Verify via ASE (independent reference).
    atoms = Atoms("Si" * pos.shape[0], positions=pos, cell=cell, pbc=True)
    d = atoms.get_all_distances(mic=True)
    d = d[d > 0]
    assert d.min() >= 2.0 - 1e-9, d.min()


def test_mc_fallback_triggers_and_converges_close():
    """Dense regime (min_dist=2.5 Å, ~0.43 packing fraction): Poisson-disk
    exhausts; MC fallback — seeded with the partial Poisson set plus
    uniform fill from the same RNG stream (matching the dispatcher) —
    reaches ≥ 0.95 * min_dist.
    """
    cell = np.eye(3) * 15.91
    min_dist = 2.5
    rng = np.random.default_rng(0)

    # Single RNG threaded through both calls: Poisson consumes some of
    # the stream, the fill continues from where Poisson left off, so
    # fill positions can never coincide with Poisson positions.
    partial = poisson_disk_pack(
        216, cell, min_dist, rng=rng, max_passes=40, return_partial=True,
    )
    assert isinstance(partial, tuple)
    partial_pos, n_placed = partial
    assert n_placed < 216

    n_remaining = 216 - n_placed
    fill = rng.random((n_remaining, 3)) @ cell
    seed_positions = np.vstack([partial_pos, fill])
    pos, dmin = mc_soft_pack(
        seed_positions, cell, min_dist, rng=rng, n_sweeps=2500,
    )
    assert pos.shape == (216, 3)
    assert dmin >= 0.95 * min_dist, dmin


def test_pack_dispatch_falls_back_automatically():
    """pack(...) never raises on a dense case; it warns if it couldn't
    meet min_distance."""
    cell = np.eye(3) * 15.91
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pos = pack(216, cell, 2.5, rng=np.random.default_rng(0))
    assert pos.shape == (216, 3)
    dmin = min_pairwise_distance(pos, cell)
    # Must be within the 0.95 tolerance (no "below tolerance" warning).
    assert dmin >= 0.95 * 2.5, dmin
    # Should have emitted at least one warning (either slightly-below or
    # no warning at all if it barely made it; the former is the typical
    # path for this seed).
    messages = [str(w.message) for w in caught]
    # Not strictly required that a warning fires on every seed, so just
    # check that if one did, it was a UserWarning and didn't mention
    # "below tolerance".
    for m in messages:
        assert "below tolerance" not in m


def test_min_pairwise_distance_matches_ase():
    rng = np.random.default_rng(42)
    cell = np.eye(3) * 10.0
    positions = rng.random((32, 3)) @ cell
    ours = min_pairwise_distance(positions, cell)
    atoms = Atoms("Si" * 32, positions=positions, cell=cell, pbc=True)
    d = atoms.get_all_distances(mic=True)
    ref = d[d > 0].min()
    assert abs(ours - ref) < 1e-6, (ours, ref)


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


def _runner():
    return CliRunner()


def test_cli_backward_compatible(tmp_path):
    out = tmp_path / "init.xyz"
    result = _runner().invoke(
        initialize,
        [
            "--output", str(out),
            "--density", "1.0",
            "--species", "Si",
            "--num-atoms", "216",
            "--min-distance", "2.0",
            "--seed", "0",
        ],
    )
    assert result.exit_code == 0, result.output
    atoms = read(str(out))
    assert len(atoms) == 216
    d = atoms.get_all_distances(mic=True)
    assert d[d > 0].min() >= 2.0 - 1e-9


def test_cli_cell_flag(tmp_path):
    """--cell-a without --density: density computed implicitly."""
    out = tmp_path / "init.xyz"
    result = _runner().invoke(
        initialize,
        [
            "--output", str(out),
            "--cell-a", "15.91",
            "--species", "Si",
            "--num-atoms", "216",
            "--min-distance", "2.0",
            "--seed", "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Density:" in result.output
    atoms = read(str(out))
    assert len(atoms) == 216
    assert abs(atoms.cell[0, 0] - 15.91) < 1e-6


def test_cli_conflicting_cell_density_count(tmp_path):
    """cell + density + num-atoms that disagree raises ClickException."""
    out = tmp_path / "init.xyz"
    # a=15.91, n=216 Si -> density ~2.5. Request density=1.0 -> conflict.
    result = _runner().invoke(
        initialize,
        [
            "--output", str(out),
            "--cell-a", "15.91",
            "--density", "1.0",
            "--species", "Si",
            "--num-atoms", "216",
            "--min-distance", "2.0",
        ],
    )
    assert result.exit_code != 0
    # Click prints the ClickException message prefixed with "Error:".
    assert "Inconsistent" in result.output or "inconsistent" in result.output


def test_cli_cell_and_density_derive_counts(tmp_path):
    """--cell-a + --density (no --num-atoms) derives num-atoms for single
    species."""
    out = tmp_path / "init.xyz"
    result = _runner().invoke(
        initialize,
        [
            "--output", str(out),
            "--cell-a", "15.91",
            "--density", "2.5",
            "--species", "Si",
            "--min-distance", "2.0",
            "--seed", "0",
        ],
    )
    assert result.exit_code == 0, result.output
    atoms = read(str(out))
    # 2.5 g/cm^3 Si in 15.91^3 Å^3 -> 216.0 atoms.
    assert len(atoms) == 216


def test_cli_seed_reproducibility(tmp_path):
    """Same --seed produces byte-identical output."""
    out_a = tmp_path / "a.xyz"
    out_b = tmp_path / "b.xyz"
    args = [
        "--density", "2.5",
        "--species", "Si",
        "--num-atoms", "216",
        "--min-distance", "2.0",
        "--seed", "12345",
    ]
    r1 = _runner().invoke(initialize, ["--output", str(out_a), *args])
    r2 = _runner().invoke(initialize, ["--output", str(out_b), *args])
    assert r1.exit_code == 0 and r2.exit_code == 0, (r1.output, r2.output)
    assert out_a.read_bytes() == out_b.read_bytes()
