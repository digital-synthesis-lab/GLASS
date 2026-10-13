"""Unit tests for the diffusion model components.

Tests cover:
- VarianceExplodingDiffuser forward noise generation
- SDE properties (alpha, sigma, f, g functions)
- Noise statistics and scaling
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch
import numpy as np
from pathlib import Path

from glass.diffusion import VarianceExplodingDiffuser


def test_diffuser_initialization():
    """Test that VarianceExplodingDiffuser initializes with correct parameters."""
    diffuser = VarianceExplodingDiffuser(k=1.0, t_min=1e-3, t_max=0.999)
    assert diffuser.t_min == 1e-3
    assert diffuser.t_max == 0.999


def test_diffuser_alpha_function():
    """Test that alpha(t) = 1 for all t (variance-exploding SDE)."""
    diffuser = VarianceExplodingDiffuser(k=1.0)
    t_values = torch.tensor([0.0, 0.5, 1.0, 2.0])
    for t in t_values:
        assert diffuser.alpha(t) == 1.0


def test_diffuser_sigma_function():
    """Test that sigma(t) = k * t."""
    k = 2.0
    diffuser = VarianceExplodingDiffuser(k=k)
    t_values = torch.tensor([0.0, 0.5, 1.0, 2.0])
    expected = k * t_values
    actual = torch.tensor([diffuser.sigma(t) for t in t_values])
    torch.testing.assert_close(actual, expected)


def test_diffuser_f_function():
    """Test that f(t) = 0 for variance-exploding SDE."""
    diffuser = VarianceExplodingDiffuser(k=1.0)
    t_values = torch.tensor([0.0, 0.5, 1.0, 2.0])
    for t in t_values:
        assert diffuser.f(t) == 0.0


def test_diffuser_g2_function():
    """Test that g^2(t) = 2 * k^2 * t."""
    k = 2.0
    diffuser = VarianceExplodingDiffuser(k=k)
    t_values = torch.tensor([0.5, 1.0, 2.0])
    expected = 2 * (k ** 2) * t_values
    actual = torch.tensor([diffuser.g2(t) for t in t_values])
    torch.testing.assert_close(actual, expected)


def test_diffuser_g_function():
    """Test that g(t) = sqrt(g^2(t))."""
    k = 2.0
    diffuser = VarianceExplodingDiffuser(k=k)
    t_values = torch.tensor([0.5, 1.0, 2.0])
    expected = torch.sqrt(torch.tensor([diffuser.g2(t) for t in t_values]))
    actual = torch.tensor([diffuser.g(t) for t in t_values])
    torch.testing.assert_close(actual, expected)


def test_forward_noise_shape():
    """Test that forward_noise returns correct shapes."""
    diffuser = VarianceExplodingDiffuser(k=1.0)
    x = torch.randn(10, 3)  # 10 atoms, 3D positions
    t = torch.tensor([[0.5]]).expand(10, 1)

    noisy_x, eps = diffuser.forward_noise(x, t)

    assert noisy_x.shape == x.shape
    assert eps.shape == x.shape


def test_forward_noise_deterministic_alpha():
    """Test that forward_noise applies alpha correctly (alpha=1 for VE-SDE)."""
    diffuser = VarianceExplodingDiffuser(k=1.0)
    torch.manual_seed(42)
    x = torch.randn(10, 3)
    t = torch.tensor([[0.0]]).expand(10, 1)  # t=0 means no noise

    noisy_x, eps = diffuser.forward_noise(x, t)

    # At t=0, sigma=0, so noisy_x should equal x (plus floating point noise)
    # alpha=1, sigma=0, so noisy_x = alpha * x + sigma * eps = x
    torch.testing.assert_close(noisy_x, x, atol=1e-6, rtol=1e-5)


def test_forward_noise_statistics():
    """Test that noise has approximately zero mean and unit variance."""
    diffuser = VarianceExplodingDiffuser(k=1.0)
    torch.manual_seed(42)
    x = torch.randn(1000, 3)
    t = torch.tensor([[1.0]]).expand(1000, 1)

    noisy_x, eps = diffuser.forward_noise(x, t)

    # eps should be standard normal
    eps_mean = eps.mean().item()
    eps_std = eps.std().item()

    assert abs(eps_mean) < 0.1, f"Noise mean {eps_mean} too far from 0"
    assert abs(eps_std - 1.0) < 0.1, f"Noise std {eps_std} too far from 1"


def test_forward_noise_scaling():
    """Test that forward_noise scales correctly with sigma."""
    diffuser = VarianceExplodingDiffuser(k=1.0)
    torch.manual_seed(42)
    x = torch.randn(100, 3)

    # Test at different noise levels
    t_small = torch.tensor([[0.1]]).expand(100, 1)
    t_large = torch.tensor([[1.0]]).expand(100, 1)

    noisy_x_small, _ = diffuser.forward_noise(x, t_small)
    noisy_x_large, _ = diffuser.forward_noise(x, t_large)

    # Larger t should result in larger noise magnitude
    noise_small = (noisy_x_small - x).abs().mean().item()
    noise_large = (noisy_x_large - x).abs().mean().item()

    assert noise_large > noise_small, "Larger t should produce more noise"


def test_forward_noise_with_batched_t():
    """Test forward_noise with batched time values."""
    diffuser = VarianceExplodingDiffuser(k=1.0)
    x = torch.randn(10, 3)
    t = torch.rand(10, 1).clip(diffuser.t_min, diffuser.t_max)

    noisy_x, eps = diffuser.forward_noise(x, t)

    assert noisy_x.shape == x.shape
    assert eps.shape == x.shape


def test_forward_noise_preserves_device():
    """Test that forward_noise preserves tensor device."""
    diffuser = VarianceExplodingDiffuser(k=1.0)
    x = torch.randn(10, 3)
    t = torch.tensor([[0.5]]).expand(10, 1)

    noisy_x, eps = diffuser.forward_noise(x, t)

    assert noisy_x.device == x.device
    assert eps.device == x.device


def test_different_k_values():
    """Test that different k values scale noise appropriately."""
    torch.manual_seed(42)
    x = torch.randn(100, 3)
    t = torch.tensor([[0.5]]).expand(100, 1)

    diffuser_k1 = VarianceExplodingDiffuser(k=1.0)
    diffuser_k2 = VarianceExplodingDiffuser(k=2.0)

    noisy_k1, _ = diffuser_k1.forward_noise(x, t)
    noisy_k2, _ = diffuser_k2.forward_noise(x, t)

    # With k=2, noise should be 2x larger (sigma = k*t)
    noise_k1 = (noisy_k1 - x).abs().mean().item()
    noise_k2 = (noisy_k2 - x).abs().mean().item()

    ratio = noise_k2 / noise_k1
    assert 1.5 < ratio < 2.5, f"Noise ratio {ratio} should be approximately 2x for k=2 vs k=1"
