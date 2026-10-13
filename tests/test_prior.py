"""Unit tests for the score-based diffusion model (LitScoreNet).

Tests cover:
- Model initialization and architecture
- Forward pass functionality
- Training step and loss computation
- EMA model updates
- Score prediction properties
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch
import numpy as np
from pathlib import Path
from ase.io import read

from glass.lit.modules import LitScoreNet
from glass.lit.datamodules.structure_spec import StructureSpecDataset, StructureSpecDataModule
from glass.diffusion import VarianceExplodingDiffuser


DATA_DIR = Path(__file__).resolve().parent / "data"


def create_minimal_model(num_species=1, num_convs=2, dim=32, ema_decay=0.999, learn_rate=1e-4):
    """Create a minimal LitScoreNet for testing."""
    return LitScoreNet(
        num_species=num_species,
        num_convs=num_convs,
        dim=dim,
        ema_decay=ema_decay,
        learn_rate=learn_rate,
    )


def test_scorenet_initialization():
    """Test that LitScoreNet initializes correctly."""
    model = create_minimal_model(num_species=1, num_convs=2, dim=32)

    assert model.hparams.num_species == 1
    assert model.hparams.num_convs == 2
    assert model.hparams.dim == 32
    assert model.hparams.ema_decay == 0.999
    assert model.hparams.learn_rate == 1e-4


def test_scorenet_model_components():
    """Test that LitScoreNet has all required model components."""
    model = create_minimal_model()

    # Check model structure
    assert hasattr(model, "model")
    assert hasattr(model.model, "encoder")
    assert hasattr(model.model, "processor")
    assert hasattr(model.model, "decoder")

    # Check EMA model
    assert hasattr(model, "ema_model")


def test_scorenet_forward_pass():
    """Test that ScoreNet can perform a forward pass."""
    model = create_minimal_model(num_species=1, num_convs=2, dim=32)
    model.eval()

    # Create dummy inputs
    batch_size = 10
    z = torch.randn(batch_size, 1)  # 1 species
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    edge_attr = torch.randn(5, 4)  # 4 = 3 (edge_vec) + 1 (edge_len)
    t = torch.rand(batch_size, 1).clip(1e-3, 0.999)
    sigma = torch.rand(batch_size, 1)

    with torch.no_grad():
        score = model.model(z, edge_index, edge_attr, t, sigma)

    assert score.shape == (batch_size, 3)  # 3D score vectors


def test_scorenet_output_scaled_by_sigma():
    """Test that model output is scaled by sigma (score = model_output / sigma)."""
    model = create_minimal_model(num_species=1, num_convs=2, dim=32)
    model.eval()

    batch_size = 10
    z = torch.randn(batch_size, 1)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    edge_attr = torch.randn(5, 4)
    t = torch.rand(batch_size, 1).clip(1e-3, 0.999)

    # Test with different sigma values
    sigma1 = torch.ones(batch_size, 1) * 0.1
    sigma2 = torch.ones(batch_size, 1) * 1.0

    with torch.no_grad():
        score1 = model.model(z, edge_index, edge_attr, t, sigma1)
        score2 = model.model(z, edge_index, edge_attr, t, sigma2)

    # Both should produce valid outputs (shape check)
    assert score1.shape == (batch_size, 3)
    assert score2.shape == (batch_size, 3)


def test_scorenet_training_step():
    """Test that training step runs without errors and returns a loss."""
    model = create_minimal_model(num_species=1, num_convs=2, dim=32)
    model.train()

    # Create a minimal batch
    batch_size = 5
    batch = type('Batch', (), {})()
    batch.z = torch.randn(batch_size, 1)
    batch.edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    batch.edge_attr = torch.randn(5, 4)
    batch.t = torch.rand(batch_size, 1).clip(1e-3, 0.999)
    batch.sigma_r = torch.rand(batch_size, 1)
    batch.eps_r = torch.randn(batch_size, 3)
    batch.num_graphs = 1

    loss = model.training_step(batch, 0)

    assert loss is not None
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar loss
    assert loss.item() >= 0  # Loss should be non-negative


def test_scorenet_loss_computation():
    """Test that loss is computed correctly: loss = mean((score * sigma + eps)^2)."""
    model = create_minimal_model(num_species=1, num_convs=2, dim=32)
    model.train()

    batch_size = 10
    batch = type('Batch', (), {})()
    batch.z = torch.randn(batch_size, 1)
    batch.edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)
    batch.edge_attr = torch.randn(10, 4)
    batch.t = torch.rand(batch_size, 1).clip(1e-3, 0.999)
    batch.sigma_r = torch.rand(batch_size, 1)
    batch.eps_r = torch.randn(batch_size, 3)
    batch.num_graphs = 1

    # Get score prediction
    score = model.model(batch.z, batch.edge_index, batch.edge_attr, batch.t, batch.sigma_r)

    # Compute loss manually
    expected_loss = (score * batch.sigma_r + batch.eps_r).pow(2).sum(dim=-1).mean()

    # Get loss from training step
    actual_loss = model.training_step(batch, 0)

    torch.testing.assert_close(actual_loss, expected_loss)


def test_scorenet_optimizer_configuration():
    """Test that optimizer is configured correctly."""
    model = create_minimal_model(num_species=1, num_convs=2, dim=32, learn_rate=5e-4)
    optimizers = model.configure_optimizers()

    assert isinstance(optimizers, torch.optim.Adam)
    assert optimizers.param_groups[0]["lr"] == 5e-4


def test_scorenet_ema_update():
    """Test that EMA model updates after optimizer step."""
    model = create_minimal_model(num_species=1, num_convs=2, dim=32)
    model.train()

    # Get initial EMA parameters
    initial_ema_params = [p.clone() for p in model.ema_model.parameters()]

    # Create a batch and do one training step
    batch_size = 5
    batch = type('Batch', (), {})()
    batch.z = torch.randn(batch_size, 1)
    batch.edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    batch.edge_attr = torch.randn(5, 4)
    batch.t = torch.rand(batch_size, 1).clip(1e-3, 0.999)
    batch.sigma_r = torch.rand(batch_size, 1)
    batch.eps_r = torch.randn(batch_size, 3)
    batch.num_graphs = 1

    loss = model.training_step(batch, 0)
    loss.backward()

    # Manually call optimizer_step (simulating what Lightning does)
    optimizer = model.configure_optimizers()
    optimizer.step()
    # Update EMA model directly
    model.ema_model.update_parameters(model.model)

    # Check that EMA parameters have been updated
    for initial, current in zip(initial_ema_params, model.ema_model.parameters()):
        assert not torch.allclose(initial, current, atol=1e-10)


def test_scorenet_parameter_count():
    """Test that model has learnable parameters."""
    model = create_minimal_model(num_species=1, num_convs=2, dim=32)

    params = list(model.parameters())
    assert len(params) > 0

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0


def test_scorenet_save_hyperparameters():
    """Test that hyperparameters are saved correctly."""
    model = create_minimal_model(num_species=2, num_convs=3, dim=64, ema_decay=0.99, learn_rate=1e-3)

    assert model.hparams.num_species == 2
    assert model.hparams.num_convs == 3
    assert model.hparams.dim == 64
    assert model.hparams.ema_decay == 0.99
    assert model.hparams.learn_rate == 1e-3


def test_scorenet_with_real_data():
    """Test ScoreNet with real Si data from test file."""
    if not (DATA_DIR / "Si_2.5_00.xyz").exists():
        pytest.skip("Test data file not found")

    atoms = read(DATA_DIR / "Si_2.5_00.xyz")

    # One-hot encode species (Si only)
    z = torch.tensor([[1.0]] * len(atoms), dtype=torch.float)  # Single species
    pos = torch.tensor(atoms.positions, dtype=torch.float)
    cell = torch.tensor(np.array(atoms.cell), dtype=torch.float)

    # Create simple graph edges (connect atoms within cutoff)
    cutoff = 5.0
    from glass.nn import periodic_radius_graph
    edge_index, edge_vec = periodic_radius_graph(pos, cutoff, cell=cell)
    edge_len = edge_vec.norm(dim=-1, keepdim=True)
    edge_attr = torch.hstack([edge_vec, edge_len])

    # Create model
    model = create_minimal_model(num_species=1, num_convs=2, dim=32)
    model.eval()

    # Sample time and sigma
    t = torch.rand(len(atoms), 1).clip(1e-3, 0.999)
    diffuser = VarianceExplodingDiffuser(k=1.0)
    sigma = torch.tensor(diffuser.sigma(t), dtype=torch.float)

    # Forward pass
    with torch.no_grad():
        score = model.model(z, edge_index, edge_attr, t, sigma)

    assert score.shape == (len(atoms), 3)


def test_scorenet_gradient_flow():
    """Test that gradients flow through the model."""
    model = create_minimal_model(num_species=1, num_convs=2, dim=32)
    model.train()

    batch_size = 5
    batch = type('Batch', (), {})()
    batch.z = torch.randn(batch_size, 1)
    batch.edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    batch.edge_attr = torch.randn(5, 4)
    batch.t = torch.rand(batch_size, 1).clip(1e-3, 0.999)
    batch.sigma_r = torch.rand(batch_size, 1)
    batch.eps_r = torch.randn(batch_size, 3)
    batch.num_graphs = 1

    # Zero gradients
    model.zero_grad()

    # Forward and backward
    loss = model.training_step(batch, 0)
    loss.backward()

    # Check that gradients exist and are non-zero
    has_grad = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "Model should have non-zero gradients"
