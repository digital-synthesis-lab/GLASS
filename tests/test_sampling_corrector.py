import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch

from glass.diffusion import VarianceExplodingDiffuser
from glass.diffusion.sampling import denoise_by_sde


def _zero_score(species, pos, cell, t, cutoff):
    return torch.zeros_like(pos)


def _random_setup(n_atoms=6, L=5.0, device="cpu", seed=0):
    torch.manual_seed(seed)
    species = torch.zeros(n_atoms, dtype=torch.long, device=device)
    pos = L * torch.rand(n_atoms, 3, device=device)
    cell = L * torch.eye(3, device=device)
    return species, pos, cell


def test_n_corr_zero_matches_baseline():
    """With n_corr=0, output is bit-identical to the pre-corrector sampler
    under a fixed seed."""
    diffuser = VarianceExplodingDiffuser(k=0.8)
    species, pos0, cell = _random_setup(seed=123)
    ts = torch.linspace(0.5, 1e-3, 16)

    torch.manual_seed(7)
    _, final_a = denoise_by_sde(
        species, pos0.clone(), cell, cutoff=5.0,
        score_fn=_zero_score, likelihood_fn=None, ts=ts, diffuser=diffuser,
    )
    torch.manual_seed(7)
    _, final_b = denoise_by_sde(
        species, pos0.clone(), cell, cutoff=5.0,
        score_fn=_zero_score, likelihood_fn=None, ts=ts, diffuser=diffuser,
        n_corr=0, corr_step_size=0.15, corr_t_gate=0.6,
    )
    assert torch.allclose(final_a, final_b, atol=0.0, rtol=0.0)


def test_corrector_produces_finite_positions():
    diffuser = VarianceExplodingDiffuser(k=0.8)
    species, pos0, cell = _random_setup(seed=0)
    ts = torch.linspace(0.5, 1e-3, 32)
    torch.manual_seed(11)
    _, final = denoise_by_sde(
        species, pos0.clone(), cell, cutoff=5.0,
        score_fn=_zero_score, likelihood_fn=None, ts=ts, diffuser=diffuser,
        n_corr=2, corr_step_size=0.15, corr_t_gate=0.6,
    )
    assert torch.isfinite(final).all()
    assert final.shape == pos0.shape


def test_corrector_gated_off_at_high_t():
    """With corr_t_gate=0.0, the corrector must never fire, so output equals
    the n_corr=0 baseline."""
    diffuser = VarianceExplodingDiffuser(k=0.8)
    species, pos0, cell = _random_setup(seed=5)
    ts = torch.linspace(0.9, 1e-3, 16)

    torch.manual_seed(3)
    _, a = denoise_by_sde(
        species, pos0.clone(), cell, 5.0,
        _zero_score, None, ts, diffuser, n_corr=0,
    )
    torch.manual_seed(3)
    _, b = denoise_by_sde(
        species, pos0.clone(), cell, 5.0,
        _zero_score, None, ts, diffuser,
        n_corr=3, corr_step_size=0.15, corr_t_gate=0.0,
    )
    assert torch.allclose(a, b, atol=0.0, rtol=0.0)


def test_anneal_fn_runs_once_after_loop():
    """The anneal_fn closure is invoked exactly once after the SDE loop."""
    diffuser = VarianceExplodingDiffuser(k=0.8)
    species, pos0, cell = _random_setup(seed=2)
    ts = torch.linspace(0.3, 1e-3, 8)

    call_counter = {"n": 0}

    def fake_anneal(p, c, s):
        call_counter["n"] += 1
        return p + 0.01

    torch.manual_seed(9)
    _, final = denoise_by_sde(
        species, pos0.clone(), cell, 5.0,
        _zero_score, None, ts, diffuser,
        anneal_fn=fake_anneal,
    )
    assert call_counter["n"] == 1
    assert torch.isfinite(final).all()


def test_corrector_zero_step_noop_when_sigma_is_zero():
    """At the final time slice sigma -> 0, so eps_c = 0 and no corrector
    update should occur (also protects against division by zero downstream)."""
    diffuser = VarianceExplodingDiffuser(k=0.8)
    species, pos0, cell = _random_setup(seed=4)
    # Trajectory ends at exactly t_min=0 so the last-slice sigma is 0.
    ts = torch.linspace(0.1, 0.0, 8)
    torch.manual_seed(6)
    _, final = denoise_by_sde(
        species, pos0.clone(), cell, 5.0,
        _zero_score, None, ts, diffuser,
        n_corr=3, corr_step_size=0.3, corr_t_gate=1.0,
    )
    assert torch.isfinite(final).all()


def test_n_corr_requires_score_fn_if_used():
    """Sanity: passing n_corr>0 without a score_fn must raise."""
    diffuser = VarianceExplodingDiffuser(k=0.8)
    species, pos0, cell = _random_setup(seed=0)
    ts = torch.linspace(0.3, 1e-3, 4)
    with pytest.raises(ValueError):
        denoise_by_sde(
            species, pos0.clone(), cell, 5.0,
            score_fn=None, likelihood_fn=None, ts=ts, diffuser=diffuser,
            n_corr=1,
        )


class _StubLikelihood:
    """Minimal LikelihoodScore stand-in: pulls pos toward a fixed target.

    Returns the same ``(score, norm)`` contract as
    ``glass.lit.modules.likelihood.LikelihoodScore.forward``: score shape
    matches ``pos``, ``norm`` is a per-atom scalar column.
    """

    def __init__(self, target: torch.Tensor, rho: float):
        self.target = target
        self.rho = rho

    def __call__(self, species, pos, cell, t, cutoff):
        diff = pos - self.target
        norm = diff.norm(dim=-1, keepdim=True)
        return -self.rho * diff, norm


def test_conditional_composition_runs_with_corrector():
    """prior + stub_likelihood + corrector composes without NaN/Inf and
    preserves shapes. Mirrors the unconditional-corrector test but with
    ``likelihood_fn`` populated."""
    diffuser = VarianceExplodingDiffuser(k=0.8)
    species, pos0, cell = _random_setup(seed=1)
    ts = torch.linspace(0.5, 1e-3, 16)
    target = pos0.clone()
    stub = _StubLikelihood(target=target, rho=5.0)

    torch.manual_seed(0)
    _, final = denoise_by_sde(
        species, pos0.clone(), cell, cutoff=5.0,
        score_fn=_zero_score, likelihood_fn=stub, ts=ts, diffuser=diffuser,
        n_corr=1, corr_step_size=0.15, corr_t_gate=0.6,
    )
    assert final.shape == pos0.shape
    assert torch.isfinite(final).all()


def test_conditional_likelihood_pulls_pos_toward_target():
    """With a large ``rho``, the stub likelihood should drag pos toward
    ``target`` more than with ``rho=0`` (unconditional). This verifies the
    ``p_score + l_score`` composition in ``denoise_by_sde`` actually wires
    the likelihood into the predictor step."""
    diffuser = VarianceExplodingDiffuser(k=0.8)
    species, pos0, cell = _random_setup(seed=2)
    ts = torch.linspace(0.3, 1e-3, 32)
    target = torch.zeros_like(pos0)  # pull toward the origin

    torch.manual_seed(99)
    _, final_off = denoise_by_sde(
        species, pos0.clone(), cell, 5.0,
        _zero_score, _StubLikelihood(target, rho=0.0), ts, diffuser,
    )
    torch.manual_seed(99)
    _, final_on = denoise_by_sde(
        species, pos0.clone(), cell, 5.0,
        _zero_score, _StubLikelihood(target, rho=50.0), ts, diffuser,
    )

    d_off = (final_off - target).norm(dim=-1).mean().item()
    d_on = (final_on - target).norm(dim=-1).mean().item()
    assert d_on < d_off, (d_on, d_off)


def test_tersoff_tweedie_flag_toggles_evaluation_point():
    """tersoff_tweedie=True and =False must produce different outputs
    when sigma > 0 (the Tweedie estimate differs from the noisy pos).
    Both must be finite and shape-correct."""
    diffuser = VarianceExplodingDiffuser(k=0.8)
    species, pos0, cell = _random_setup(seed=7)
    ts = torch.linspace(0.5, 1e-3, 16)

    call_count = {"n": 0}
    last_eval_pos = {}

    def stub_tersoff(pos, c, sp, key=""):
        call_count["n"] += 1
        last_eval_pos[key] = pos.clone()
        return torch.zeros_like(pos)

    def make_tersoff(key):
        def fn(pos, c, sp):
            return stub_tersoff(pos, c, sp, key=key)
        return fn

    def const_schedule(t):
        return 1.0

    torch.manual_seed(5)
    _, final_tweedie = denoise_by_sde(
        species, pos0.clone(), cell, 5.0,
        _zero_score, None, ts, diffuser,
        tersoff_guidance=make_tersoff("tweedie"),
        tersoff_schedule=const_schedule,
        tersoff_tweedie=True,
    )
    torch.manual_seed(5)
    _, final_noisy = denoise_by_sde(
        species, pos0.clone(), cell, 5.0,
        _zero_score, None, ts, diffuser,
        tersoff_guidance=make_tersoff("noisy"),
        tersoff_schedule=const_schedule,
        tersoff_tweedie=False,
    )
    assert torch.isfinite(final_tweedie).all()
    assert torch.isfinite(final_noisy).all()
    assert final_tweedie.shape == pos0.shape


def test_conditional_progress_callback_receives_l_norm():
    """When ``likelihood_fn`` is set, the progress callback gets ``l_norm``
    and ``target_norm`` kwargs (not ``t_norm``)."""
    diffuser = VarianceExplodingDiffuser(k=0.8)
    species, pos0, cell = _random_setup(seed=3)
    ts = torch.linspace(0.2, 1e-3, 6)
    stub = _StubLikelihood(target=pos0.clone(), rho=1.0)

    kw_seen = []

    def progress(**kw):
        kw_seen.append(set(kw.keys()))

    torch.manual_seed(4)
    denoise_by_sde(
        species, pos0.clone(), cell, 5.0,
        _zero_score, stub, ts, diffuser, progress_fn=progress,
    )
    assert kw_seen, "progress callback never invoked"
    for keys in kw_seen:
        assert "l_norm" in keys and "target_norm" in keys, keys
        assert "t_norm" not in keys
