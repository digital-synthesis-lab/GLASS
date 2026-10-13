import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch

from glass.diffusion.schedules import power_law_ts, linear_ts


def test_rho_one_matches_linspace():
    tmin, tmax, n = 1e-3, 1.0, 256
    ts_pl = power_law_ts(tmin, tmax, n, rho=1.0)
    ts_ls = torch.linspace(tmax, tmin, n)
    assert ts_pl.shape == (n,)
    assert torch.allclose(ts_pl, ts_ls, atol=1e-6)


def test_linear_ts_alias_matches_linspace():
    tmin, tmax, n = 1e-3, 1.0, 128
    ts = linear_ts(tmin, tmax, n)
    ref = torch.linspace(tmax, tmin, n)
    assert torch.allclose(ts, ref, atol=1e-6)


def test_bounds():
    for rho in (0.5, 1.0, 2.0, 3.0):
        ts = power_law_ts(1e-3, 1.0, 64, rho=rho)
        assert float(ts[0]) == pytest.approx(1.0, abs=1e-6)
        assert float(ts[-1]) == pytest.approx(1e-3, abs=1e-6)


def test_monotonic_decreasing():
    for rho in (0.5, 1.0, 2.0, 3.0):
        ts = power_law_ts(1e-3, 1.0, 64, rho=rho)
        diffs = ts[1:] - ts[:-1]
        assert (diffs <= 1e-7).all(), f"not monotonic at rho={rho}"


def test_rho_large_concentrates_near_zero():
    tmin, tmax, n = 0.0, 1.0, 256
    ts_lin = power_law_ts(tmin, tmax, n, rho=1.0)
    ts_big = power_law_ts(tmin, tmax, n, rho=3.0)
    threshold = 0.1
    n_low_lin = int((ts_lin <= threshold).sum())
    n_low_big = int((ts_big <= threshold).sum())
    assert n_low_big > n_low_lin


def test_rejects_bad_args():
    with pytest.raises(ValueError):
        power_law_ts(1e-3, 1.0, tstep=1, rho=1.0)
    with pytest.raises(ValueError):
        power_law_ts(1e-3, 1.0, tstep=10, rho=0.0)
    with pytest.raises(ValueError):
        power_law_ts(1e-3, 1.0, tstep=10, rho=-1.0)
