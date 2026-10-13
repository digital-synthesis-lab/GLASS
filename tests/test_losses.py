"""Validation strategies and unit tests for different loss functions.

This module provides:
- Loss function validation strategies (MSE, MAE, Huber)
- Score matching loss validation
- Training loss monitoring patterns
- Loss convergence and stability checks
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from ase.io import read

from glass.lit.modules import LitScoreNet, LitSpecNet
from glass.diffusion import VarianceExplodingDiffuser


DATA_DIR = Path(__file__).resolve().parent / "data"


class LossValidator:
    """Validation strategies for different loss functions.

    This class provides methods to validate that loss functions behave
    as expected during training and evaluation.
    """

    @staticmethod
    def validate_mse_loss(predictions: torch.Tensor, targets: torch.Tensor,
                           expected_reduction: str = "mean") -> dict:
        """Validate MSE loss computation.

        Returns:
            Dictionary with loss value and validation metrics.
        """
        if expected_reduction == "mean":
            loss = F.mse_loss(predictions, targets, reduction="mean")
        elif expected_reduction == "sum":
            loss = F.mse_loss(predictions, targets, reduction="sum")
        elif expected_reduction == "none":
            loss = F.mse_loss(predictions, targets, reduction="none")
        else:
            raise ValueError(f"Unknown reduction: {expected_reduction}")

        # Compute manual verification
        manual_loss = ((predictions - targets) ** 2)
        if expected_reduction == "mean":
            manual_loss = manual_loss.mean()
        elif expected_reduction == "sum":
            manual_loss = manual_loss.sum()

        return {
            "loss": loss,
            "manual_loss": manual_loss,
            "matches": torch.allclose(loss, manual_loss, rtol=1e-5),
            "max_error": (predictions - targets).abs().max().item(),
            "mean_error": (predictions - targets).abs().mean().item(),
        }

    @staticmethod
    def validate_mae_loss(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """Validate MAE (L1) loss computation.

        Returns:
            Dictionary with loss value and validation metrics.
        """
        loss = F.l1_loss(predictions, targets, reduction="mean")
        manual_loss = (predictions - targets).abs().mean()

        return {
            "loss": loss,
            "manual_loss": manual_loss,
            "matches": torch.allclose(loss, manual_loss, rtol=1e-5),
            "max_error": (predictions - targets).abs().max().item(),
        }

    @staticmethod
    def validate_huber_loss(predictions: torch.Tensor, targets: torch.Tensor,
                            delta: float = 1.0) -> dict:
        """Validate Huber loss computation.

        Returns:
            Dictionary with loss value and validation metrics.
        """
        loss = F.smooth_l1_loss(predictions, targets, beta=delta, reduction="mean")

        # Manual Huber loss computation
        # PyTorch smooth_l1_loss formula:
        # loss = 0.5 * (diff^2) / beta      if diff < beta
        # loss = diff - 0.5 * beta          if diff >= beta
        diff = (predictions - targets).abs()
        manual_loss = torch.where(
            diff < delta,
            0.5 * diff ** 2 / delta,
            diff - 0.5 * delta
        ).mean()

        return {
            "loss": loss,
            "manual_loss": manual_loss,
            "matches": torch.allclose(loss, manual_loss, rtol=1e-5),
            "delta": delta,
        }

    @staticmethod
    def validate_score_matching_loss(score: torch.Tensor, sigma: torch.Tensor,
                                     eps: torch.Tensor) -> dict:
        """Validate score matching loss: loss = mean((score * sigma + eps)^2).

        This is the denoising score matching objective used in the diffusion model.

        Returns:
            Dictionary with loss value and validation metrics.
        """
        # Compute loss
        loss = (score * sigma + eps).pow(2).sum(dim=-1).mean()

        # Manual verification
        manual_residual = score * sigma + eps
        manual_loss = manual_residual.pow(2).sum(dim=-1).mean()

        # Check properties
        residual_norm = manual_residual.norm(dim=-1)

        return {
            "loss": loss,
            "manual_loss": manual_loss,
            "matches": torch.allclose(loss, manual_loss, rtol=1e-5),
            "mean_residual_norm": residual_norm.mean().item(),
            "max_residual_norm": residual_norm.max().item(),
            "is_finite": torch.isfinite(loss).item(),
        }

    @staticmethod
    def check_loss_properties(losses: list, check_decreasing: bool = True,
                              max_allowed: float = None) -> dict:
        """Check loss properties over training steps.

        Args:
            losses: List of loss values over training steps
            check_decreasing: Whether to check if loss is generally decreasing
            max_allowed: Maximum allowed loss value

        Returns:
            Dictionary with validation results.
        """
        losses = torch.tensor(losses)

        results = {
            "initial": losses[0].item(),
            "final": losses[-1].item(),
            "min": losses.min().item(),
            "max": losses.max().item(),
            "mean": losses.mean().item(),
            "std": losses.std().item(),
            "is_finite": torch.all(torch.isfinite(losses)).item(),
        }

        if check_decreasing:
            # Check if loss is generally decreasing (allow for some noise)
            window = min(10, len(losses) // 4)
            if window > 0:
                early_mean = losses[:window].mean()
                late_mean = losses[-window:].mean()
                results["decreasing"] = late_mean < early_mean

        if max_allowed is not None:
            results["within_max"] = losses.max().item() < max_allowed

        return results


# =============================================================================
# Test Functions
# =============================================================================

def test_mse_loss_validation():
    """Test MSE loss validation with known values."""
    torch.manual_seed(42)
    predictions = torch.randn(100, 10)
    targets = torch.randn(100, 10)

    # Test mean reduction
    result = LossValidator.validate_mse_loss(predictions, targets, "mean")
    assert result["matches"], "MSE loss should match manual computation"
    assert result["loss"] >= 0, "MSE loss should be non-negative"

    # Test sum reduction
    result_sum = LossValidator.validate_mse_loss(predictions, targets, "sum")
    assert result_sum["matches"], "MSE loss (sum) should match manual computation"

    # Test no reduction
    result_none = LossValidator.validate_mse_loss(predictions, targets, "none")
    assert result_none["matches"], "MSE loss (none) should match manual computation"


def test_mse_loss_perfect_prediction():
    """Test MSE loss is zero for perfect predictions."""
    targets = torch.randn(50, 5)
    predictions = targets.clone()

    result = LossValidator.validate_mse_loss(predictions, targets)
    assert abs(result["loss"].item()) < 1e-6, "MSE loss should be ~0 for perfect predictions"


def test_mae_loss_validation():
    """Test MAE loss validation."""
    torch.manual_seed(42)
    predictions = torch.randn(100, 10)
    targets = torch.randn(100, 10)

    result = LossValidator.validate_mae_loss(predictions, targets)
    assert result["matches"], "MAE loss should match manual computation"
    assert result["loss"] >= 0, "MAE loss should be non-negative"


def test_huber_loss_validation():
    """Test Huber loss validation."""
    torch.manual_seed(42)
    predictions = torch.randn(100, 10)
    targets = torch.randn(100, 10)

    for delta in [0.5, 1.0, 2.0]:
        result = LossValidator.validate_huber_loss(predictions, targets, delta=delta)
        assert result["matches"], f"Huber loss should match manual computation for delta={delta}"
        assert result["loss"] >= 0, "Huber loss should be non-negative"


def test_huber_vs_mse_small_errors():
    """Test that Huber approximates (0.5/delta) * MSE for small errors."""
    delta = 1.0
    # Small errors - should be similar to (0.5/delta) * MSE
    predictions = torch.randn(100, 5) * 0.1
    targets = torch.zeros(100, 5)

    huber_result = LossValidator.validate_huber_loss(predictions, targets, delta=delta)
    mse_result = LossValidator.validate_mse_loss(predictions, targets)

    # For small errors, Huber ~ (0.5/delta) * MSE
    ratio = huber_result["loss"].item() / (0.5 / delta * mse_result["loss"].item())
    assert 0.9 < ratio < 1.1, f"Huber should approximate (0.5/delta)*MSE for small errors, got ratio {ratio}"


def test_score_matching_loss_validation():
    """Test score matching loss used in diffusion model."""
    torch.manual_seed(42)
    batch_size = 100
    dim = 3

    score = torch.randn(batch_size, dim)
    sigma = torch.rand(batch_size, 1) + 0.01
    eps = torch.randn(batch_size, dim)

    result = LossValidator.validate_score_matching_loss(score, sigma, eps)

    assert result["matches"], "Score matching loss should match manual computation"
    assert result["is_finite"], "Score matching loss should be finite"
    assert result["loss"] >= 0, "Score matching loss should be non-negative"


def test_score_matching_loss_perfect():
    """Test score matching loss when score = -eps/sigma (perfect prediction)."""
    torch.manual_seed(42)
    batch_size = 100
    dim = 3

    eps = torch.randn(batch_size, dim)
    sigma = torch.rand(batch_size, 1) + 0.01

    # Perfect score: score = -eps / sigma
    score = -eps / sigma

    result = LossValidator.validate_score_matching_loss(score, sigma, eps)

    assert abs(result["loss"].item()) < 1e-5, "Loss should be ~0 for perfect score"


def test_score_matching_loss_random():
    """Test score matching loss with random inputs."""
    torch.manual_seed(42)

    for _ in range(5):
        batch_size = np.random.randint(10, 100)
        dim = 3

        score = torch.randn(batch_size, dim)
        sigma = torch.rand(batch_size, 1) * 2 + 0.01
        eps = torch.randn(batch_size, dim)

        result = LossValidator.validate_score_matching_loss(score, sigma, eps)

        assert result["is_finite"], "Score matching loss should be finite"
        assert result["loss"] >= 0, "Score matching loss should be non-negative"


def test_loss_properties_decreasing():
    """Test loss property checker with decreasing loss."""
    # Simulate a decreasing loss curve
    losses = [10.0 / (i + 1) + np.random.randn() * 0.1 for i in range(50)]
    losses = [max(0.1, l) for l in losses]

    result = LossValidator.check_loss_properties(losses, check_decreasing=True)

    assert result["is_finite"], "All losses should be finite"
    assert result["decreasing"], "Loss should be generally decreasing"


def test_loss_properties_constant():
    """Test loss property checker with constant loss."""
    losses = [1.0] * 50

    result = LossValidator.check_loss_properties(losses, check_decreasing=False)

    assert result["is_finite"], "All losses should be finite"
    assert result["std"] < 1e-6, "Constant loss should have zero std"


def test_loss_properties_exploding():
    """Test loss property checker detects exploding loss."""
    losses = [1.0 * (1.1 ** i) for i in range(20)]

    result = LossValidator.check_loss_properties(losses, check_decreasing=False)

    assert result["is_finite"], "All losses should be finite"
    assert result["final"] > result["initial"] * 5, "Exploding loss should have large increase"


def test_litspecnet_mse_loss():
    """Test MSE loss in LitSpecNet (spectral prediction model)."""
    model = LitSpecNet(
        num_species=1,
        num_convs=2,
        dim=32,
        out_dim=100,
        ema_decay=0.999,
        learn_rate=1e-4,
    )
    model.eval()

    # Create dummy batch
    batch_size = 10
    batch = type('Batch', (), {})()
    batch.z = torch.randn(batch_size, 1)
    batch.edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    batch.edge_attr = torch.randn(5, 4)
    batch.y = torch.randn(batch_size, 100)
    batch.train_mask = torch.tensor([True] * 5 + [False] * 5)
    batch.num_graphs = 1

    # Get predictions
    with torch.no_grad():
        pred_y = model.model(batch.z, batch.edge_index, batch.edge_attr)

    # Compute losses
    train_loss = F.mse_loss(pred_y[batch.train_mask], batch.y[batch.train_mask])
    valid_loss = F.mse_loss(pred_y[~batch.train_mask], batch.y[~batch.train_mask])

    # Validate
    assert train_loss.item() >= 0, "Training loss should be non-negative"
    assert valid_loss.item() >= 0, "Validation loss should be non-negative"
    assert torch.isfinite(train_loss), "Training loss should be finite"
    assert torch.isfinite(valid_loss), "Validation loss should be finite"


def test_scorenet_training_loss_behavior():
    """Test that LitScoreNet produces expected loss behavior."""
    model = LitScoreNet(
        num_species=1,
        num_convs=2,
        dim=32,
        ema_decay=0.999,
        learn_rate=1e-4,
    )
    model.train()

    # Collect losses over multiple steps
    losses = []

    for step in range(10):
        batch_size = 20
        batch = type('Batch', (), {})()
        batch.z = torch.randn(batch_size, 1)
        batch.edge_index = torch.randint(0, batch_size, (2, 50))
        batch.edge_attr = torch.randn(50, 4)
        batch.t = torch.rand(batch_size, 1).clip(1e-3, 0.999)
        batch.sigma_r = torch.rand(batch_size, 1)
        batch.eps_r = torch.randn(batch_size, 3)
        batch.num_graphs = 1

        loss = model.training_step(batch, step)
        losses.append(loss.item())

    # Validate loss properties
    result = LossValidator.check_loss_properties(losses, check_decreasing=False)

    assert result["is_finite"], "All losses should be finite"
    assert result["min"] >= 0, "Losses should be non-negative"


def test_loss_with_real_data():
    """Test loss computation with real Si data."""
    if not (DATA_DIR / "Si_2.5_00.xyz").exists():
        pytest.skip("Test data file not found")

    from ase import Atoms
    atoms = read(DATA_DIR / "Si_2.5_00.xyz")

    # Create simple dataset
    num_atoms = len(atoms)
    z = torch.tensor([[1.0]] * num_atoms, dtype=torch.float)
    pos = torch.tensor(atoms.positions, dtype=torch.float)
    cell = torch.tensor(np.array(atoms.cell), dtype=torch.float)

    # Create graph
    from glass.nn import periodic_radius_graph
    edge_index, edge_vec = periodic_radius_graph(pos, 5.0, cell=cell)
    edge_len = edge_vec.norm(dim=-1, keepdim=True)
    edge_attr = torch.hstack([edge_vec, edge_len])

    # Create model and diffuse
    model = LitScoreNet(num_species=1, num_convs=2, dim=32, ema_decay=0.999, learn_rate=1e-4)
    diffuser = VarianceExplodingDiffuser(k=1.0)

    # Apply diffusion
    t = torch.rand(num_atoms, 1).clip(1e-3, 0.999)
    noisy_pos, eps = diffuser.forward_noise(pos, t)
    sigma = torch.tensor(diffuser.sigma(t), dtype=torch.float)

    # Forward pass
    model.eval()
    with torch.no_grad():
        score = model.model(z, edge_index, edge_attr, t, sigma)

    # Compute loss
    result = LossValidator.validate_score_matching_loss(score, sigma, eps)

    assert result["is_finite"], "Loss should be finite"
    assert result["loss"] >= 0, "Loss should be non-negative"


def test_loss_gradient_stability():
    """Test that loss gradients remain stable."""
    model = LitScoreNet(
        num_species=1,
        num_convs=2,
        dim=32,
        ema_decay=0.999,
        learn_rate=1e-4,
    )
    model.train()

    gradient_norms = []

    for _ in range(5):
        batch_size = 20
        batch = type('Batch', (), {})()
        batch.z = torch.randn(batch_size, 1)
        batch.edge_index = torch.randint(0, batch_size, (2, 50))
        batch.edge_attr = torch.randn(50, 4)
        batch.t = torch.rand(batch_size, 1).clip(1e-3, 0.999)
        batch.sigma_r = torch.rand(batch_size, 1)
        batch.eps_r = torch.randn(batch_size, 3)
        batch.num_graphs = 1

        model.zero_grad()
        loss = model.training_step(batch, 0)
        loss.backward()

        # Compute gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        gradient_norms.append(total_norm)

    # Check that gradients are reasonable
    max_grad = max(gradient_norms)
    assert max_grad < 1000, f"Gradient norm {max_grad} is too large, possible instability"

    # Check that not all gradients are zero
    assert any(g > 0 for g in gradient_norms), "Gradients should not all be zero"
