"""Tests for TersoffMetrics and integration with compute_all_metrics / errors."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.io import read

from glass.metrics.tersoff import (
    TersoffMetrics,
    compute_tersoff_metrics,
    tersoff_energy_error,
    tersoff_forces_emd,
    tersoff_forces_histogram_rmse,
    tersoff_forces_max_error,
    tersoff_forces_rms_error,
)

DATA_DIR = Path(__file__).parent / "data"
SI_SNAPSHOT = DATA_DIR / "Si_2.5_00.xyz"


# ---------------------------------------------------------------------------
# Unit tests for compute_tersoff_metrics
# ---------------------------------------------------------------------------


def _make_si_bulk(n: int = 8) -> "ase.Atoms":
    from ase.build import bulk

    a = bulk("Si", "diamond", a=5.43)
    a = a.repeat([2, 1, 1])
    return a


def test_compute_tersoff_metrics_returns_correct_type():
    atoms = _make_si_bulk()
    m = compute_tersoff_metrics(atoms)
    assert isinstance(m, TersoffMetrics)


def test_compute_tersoff_metrics_energy_finite():
    atoms = _make_si_bulk()
    m = compute_tersoff_metrics(atoms)
    assert np.isfinite(m.energy_per_atom)
    assert m.energy_per_atom < 0  # Si diamond is bound


def test_compute_tersoff_metrics_forces_near_zero_for_equilibrium():
    """Diamond Si at its equilibrium lattice constant has near-zero forces."""
    atoms = bulk("Si", "diamond", a=5.43).repeat([2, 2, 2])
    m = compute_tersoff_metrics(atoms)
    assert m.forces_rms < 0.1, f"Forces too large for diamond: {m.forces_rms}"
    assert m.forces_max < 0.5, f"Max force too large for diamond: {m.forces_max}"


def test_compute_tersoff_metrics_histogram_normalised():
    atoms = _make_si_bulk()
    m = compute_tersoff_metrics(atoms)
    hist = np.array(m.forces_histogram)
    assert abs(hist.sum() - 1.0) < 1e-6 or hist.sum() <= 1.0 + 1e-6
    assert len(m.forces_histogram_bins) == len(m.forces_histogram) + 1


def test_compute_tersoff_metrics_roundtrip_dict():
    atoms = _make_si_bulk()
    m = compute_tersoff_metrics(atoms)
    d = m.to_dict()
    m2 = TersoffMetrics.from_dict(d)
    assert abs(m.energy_per_atom - m2.energy_per_atom) < 1e-12
    assert abs(m.forces_rms - m2.forces_rms) < 1e-12
    assert m.forces_histogram == m2.forces_histogram


@pytest.mark.skipif(not SI_SNAPSHOT.exists(), reason="Si snapshot not available")
def test_compute_tersoff_metrics_on_amorphous_snapshot():
    """Amorphous Si snapshot should have non-trivial forces."""
    atoms = read(str(SI_SNAPSHOT))
    m = compute_tersoff_metrics(atoms)
    assert m.forces_rms > 0.1, "Amorphous Si should have finite forces"
    assert m.forces_max > m.forces_rms


# ---------------------------------------------------------------------------
# Error metric tests
# ---------------------------------------------------------------------------


def _two_metrics(scale: float = 2.0):
    """Return two TersoffMetrics with a known energy/force offset."""
    atoms = bulk("Si", "diamond", a=5.43).repeat([2, 2, 2])
    m_ref = compute_tersoff_metrics(atoms)
    # Manually shift energy and forces for deterministic error tests
    d = m_ref.to_dict()
    d["energy_per_atom"] += scale * 0.1
    d["forces_rms"] += scale * 0.05
    d["forces_max"] += scale * 0.1
    d["forces_mean"] += scale * 0.03
    d["forces_std"] += scale * 0.02
    m_target = TersoffMetrics.from_dict(d)
    return m_ref, m_target


def test_tersoff_energy_error_sign_independent():
    m_ref, m_tgt = _two_metrics()
    err = tersoff_energy_error(m_ref, m_tgt)
    assert err >= 0.0
    err_rev = tersoff_energy_error(m_tgt, m_ref)
    assert abs(err - err_rev) < 1e-12  # symmetric (both take abs)


def test_tersoff_forces_rms_error_positive():
    m_ref, m_tgt = _two_metrics()
    assert tersoff_forces_rms_error(m_ref, m_tgt) > 0.0


def test_tersoff_forces_max_error_positive():
    m_ref, m_tgt = _two_metrics()
    assert tersoff_forces_max_error(m_ref, m_tgt) > 0.0


def test_tersoff_forces_histogram_rmse_zero_for_identical():
    atoms = bulk("Si", "diamond", a=5.43).repeat([2, 2, 2])
    m = compute_tersoff_metrics(atoms)
    assert tersoff_forces_histogram_rmse(m, m) < 1e-10


def test_tersoff_forces_emd_zero_for_identical():
    atoms = bulk("Si", "diamond", a=5.43).repeat([2, 2, 2])
    m = compute_tersoff_metrics(atoms)
    assert tersoff_forces_emd(m, m) < 1e-10


# ---------------------------------------------------------------------------
# Integration with compute_all_metrics and compute_all_errors
# ---------------------------------------------------------------------------


def test_compute_all_metrics_without_tersoff_leaves_field_none():
    from glass.metrics import compute_all_metrics

    atoms = _make_si_bulk()
    m = compute_all_metrics(atoms, include_tersoff=False)
    assert m.tersoff is None


def test_compute_all_metrics_with_tersoff_populates_field():
    from glass.metrics import compute_all_metrics

    atoms = _make_si_bulk()
    m = compute_all_metrics(atoms, include_tersoff=True)
    assert m.tersoff is not None
    assert isinstance(m.tersoff, TersoffMetrics)


def test_compute_all_metrics_tersoff_in_to_dict():
    from glass.metrics import compute_all_metrics

    atoms = _make_si_bulk()
    m = compute_all_metrics(atoms, include_tersoff=True)
    d = m.to_dict()
    assert "tersoff" in d
    assert "energy_per_atom" in d["tersoff"]


def test_compute_all_errors_includes_tersoff_keys_when_both_have_tersoff():
    from glass.metrics import compute_all_metrics
    from glass.metrics.errors import compute_all_errors

    atoms = _make_si_bulk()
    m_ref = compute_all_metrics(atoms, include_tersoff=True)
    m_tgt = compute_all_metrics(atoms, include_tersoff=True)
    errors = compute_all_errors(m_ref, m_tgt)
    assert "tersoff_energy_error" in errors
    assert "tersoff_forces_rms_error" in errors
    assert "tersoff_forces_histogram_rmse" in errors
    assert "tersoff_forces_emd" in errors


def test_compute_all_errors_no_tersoff_keys_when_tersoff_absent():
    from glass.metrics import compute_all_metrics
    from glass.metrics.errors import compute_all_errors

    atoms = _make_si_bulk()
    m_ref = compute_all_metrics(atoms, include_tersoff=False)
    m_tgt = compute_all_metrics(atoms, include_tersoff=False)
    errors = compute_all_errors(m_ref, m_tgt)
    assert "tersoff_energy_error" not in errors


def test_compute_all_errors_no_tersoff_keys_when_only_one_has_tersoff():
    """If only one of ref/target has Tersoff, the block is skipped."""
    from glass.metrics import compute_all_metrics
    from glass.metrics.errors import compute_all_errors

    atoms = _make_si_bulk()
    m_ref = compute_all_metrics(atoms, include_tersoff=True)
    m_tgt = compute_all_metrics(atoms, include_tersoff=False)
    errors = compute_all_errors(m_ref, m_tgt)
    assert "tersoff_energy_error" not in errors
