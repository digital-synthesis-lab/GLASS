"""Lightweight tests for the HPO driver in ``scripts/hpo.py``.

The full study needs a trained checkpoint and reference structures. These
tests exercise the parts that do not:

- The weighted-objective aggregation.
- The search-space sampler with an Optuna ``FixedTrial``, to catch bounds and
  naming regressions.
- Device-list parsing.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import importlib.util
import pathlib

import pytest

optuna = pytest.importorskip("optuna")

HPO_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "hpo.py"


def _load_hpo_module():
    spec = importlib.util.spec_from_file_location("glass_hpo", HPO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mode_objective_weights():
    hpo = _load_hpo_module()
    errors = {
        "coordination_emd": 0.2,
        "pdf_rmse": 0.1,
        "adf_rmse": 0.04,
    }
    expected = hpo.W_COORD * 0.2 + hpo.W_PDF * 0.1 + hpo.W_ADF * 0.04
    assert hpo._mode_obj(errors) == pytest.approx(expected)


def test_sample_params_respects_bounds():
    hpo = _load_hpo_module()
    fixed = optuna.trial.FixedTrial({
        "tstep": 256,
        "tmin": 5e-3,
        "tmax": 0.9,
        "t_rho": 1.0,
        "tersoff_lambda": 0.07,
        "tersoff_schedule": "sigmoid",
        "tersoff_t_gate": 0.55,
        "n_corr": 2,
        "corr_step_size": 0.2,
        "corr_t_gate": 0.5,
        "N_anneal": 100,
        "T0": 5e-3,
        "anneal_lr": 1e-3,
        "rho": 300.0,
    })
    params = hpo._sample_params(fixed)

    assert set(params) == {
        "tstep", "tmin", "tmax", "t_rho",
        "tersoff_lambda", "tersoff_schedule", "tersoff_t_gate",
        "n_corr", "corr_step_size", "corr_t_gate",
        "N_anneal", "T0", "anneal_lr", "rho",
    }
    assert params["tersoff_schedule"] in {"constant", "linear", "sigmoid"}
    assert params["n_corr"] in {0, 1, 2}
    assert params["N_anneal"] in {0, 50, 100, 200}


def test_parse_devices_roundtrip():
    hpo = _load_hpo_module()
    assert hpo._parse_devices("cuda:0") == ["cuda:0"]
    assert hpo._parse_devices("cuda:0,cuda:1,cuda:2") == ["cuda:0", "cuda:1", "cuda:2"]
    assert hpo._parse_devices(" cuda:0 , cuda:1 ") == ["cuda:0", "cuda:1"]
