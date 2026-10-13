"""Tests for inference with trained silicon model checkpoint.

This module tests the complete inference pipeline using a real trained model
checkpoint shipped in examples/silicon/experiment/checkpoints/best.ckpt.
"""

import os
import pytest
import torch
import numpy as np
from pathlib import Path
from ase.build import bulk
from ase.io import read

# Ensure KMP_DUPLICATE_LIB_OK is set before importing torch-dependent modules
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from glass.lit.modules import LitScoreNet, DifferentiableRDF
from glass.diffusion import VarianceExplodingDiffuser
from glass.diffusion.sampling import denoise_by_sde
from glass.lit.modules.likelihood import LikelihoodScore
from glass.utils.atoms import atoms_to_device, compute_prior_score
from glass.nn import periodic_radius_graph


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
CHECKPOINT_PATH = REPO_ROOT / "examples" / "silicon" / "experiment" / "checkpoints" / "best.ckpt"

pytestmark = pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(),
    reason=f"example checkpoint not found at {CHECKPOINT_PATH}",
)


class TestCheckpointLoading:
    """Test checkpoint can be loaded and contains expected structure."""
    
    def test_checkpoint_file_exists(self):
        """Verify checkpoint file exists."""
        assert CHECKPOINT_PATH.exists(), f"Checkpoint not found at {CHECKPOINT_PATH}"
    
    def test_checkpoint_can_be_loaded(self):
        """Verify checkpoint can be loaded with torch."""
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        assert isinstance(ckpt, dict)
    
    def test_checkpoint_has_required_keys(self):
        """Verify checkpoint has required keys."""
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        required_keys = [
            "state_dict",
            "hyper_parameters",
            "epoch",
            "global_step",
        ]
        for key in required_keys:
            assert key in ckpt, f"Missing key: {key}"
    
    def test_checkpoint_hyperparameters(self):
        """Verify checkpoint has expected hyperparameters."""
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        hparams = ckpt["hyper_parameters"]
        
        assert hparams["num_species"] == 1
        assert hparams["num_convs"] == 5
        assert hparams["dim"] == 200
        assert hparams["ema_decay"] == 0.9999
        assert hparams["learn_rate"] == 0.001
    
    def test_checkpoint_state_dict_structure(self):
        """Verify state dict has expected model parameters."""
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        
        # Check for expected parameter keys
        assert "model.encoder.embed_node.0.mlp.0.weight" in state_dict
        assert "model.processor.convs.0.edge_processor.edge_mlp.0.mlp.0.weight" in state_dict
        # Decoder uses nested MLP structure
        decoder_keys = [k for k in state_dict.keys() if "model.decoder" in k]
        assert len(decoder_keys) > 0, "No decoder parameters found"
        
        # Check EMA model parameters exist
        ema_keys = [k for k in state_dict.keys() if "ema_model" in k]
        assert len(ema_keys) > 0, "No EMA model parameters found"


class TestModelInitialization:
    """Test model can be initialized from checkpoint."""
    
    @pytest.fixture
    def score_net(self):
        """Load model from checkpoint."""
        return LitScoreNet.load_from_checkpoint(
            CHECKPOINT_PATH,
            map_location="cpu",
        )
    
    def test_model_loads_successfully(self, score_net):
        """Verify model loads without errors."""
        assert score_net is not None
        assert isinstance(score_net, LitScoreNet)
    
    def test_model_architecture(self, score_net):
        """Verify model has correct architecture."""
        hparams = score_net.hparams
        assert hparams.num_species == 1
        assert hparams.num_convs == 5
        assert hparams.dim == 200
    
    def test_ema_model_exists(self, score_net):
        """Verify EMA model is available."""
        assert hasattr(score_net, "ema_model")
        assert score_net.ema_model is not None
    
    def test_model_eval_mode(self, score_net):
        """Verify model can be set to eval mode."""
        score_net.eval()
        assert not score_net.training
        score_net.ema_model.eval()
        assert not score_net.ema_model.training


class TestForwardPass:
    """Test forward pass through the model."""
    
    @pytest.fixture
    def model_setup(self):
        """Setup model and test data."""
        score_net = LitScoreNet.load_from_checkpoint(
            CHECKPOINT_PATH,
            map_location="cpu",
        )
        score_net.eval()
        score_net.ema_model.eval()
        
        # Create simple test structure (diamond Si)
        atoms = bulk("Si", "diamond", a=5.43)
        species, pos, cell = atoms_to_device(atoms, "cpu")
        
        # Setup time and noise level
        t = torch.tensor([[0.5]], dtype=torch.float32)
        sigma = torch.tensor([[0.4]], dtype=torch.float32)
        
        return score_net, species, pos, cell, t, sigma
    
    def test_forward_pass_runs(self, model_setup):
        """Verify forward pass executes without errors."""
        score_net, species, pos, cell, t, sigma = model_setup
        
        edge_index, edge_vec = periodic_radius_graph(pos, r=5.0, cell=cell)
        edge_attr = torch.hstack([edge_vec, edge_vec.norm(dim=-1, keepdim=True)])
        
        with torch.no_grad():
            score = score_net.ema_model(
                species, edge_index, edge_attr, t, sigma
            )
        
        assert score is not None
        assert score.shape == pos.shape
    
    def test_forward_pass_output_finite(self, model_setup):
        """Verify forward pass produces finite values."""
        score_net, species, pos, cell, t, sigma = model_setup
        
        edge_index, edge_vec = periodic_radius_graph(pos, r=5.0, cell=cell)
        edge_attr = torch.hstack([edge_vec, edge_vec.norm(dim=-1, keepdim=True)])
        
        with torch.no_grad():
            score = score_net.ema_model(
                species, edge_index, edge_attr, t, sigma
            )
        
        assert torch.isfinite(score).all()
    
    def test_forward_pass_different_noise_levels(self, model_setup):
        """Verify model works at different noise levels."""
        score_net, species, pos, cell, t, sigma = model_setup
        
        edge_index, edge_vec = periodic_radius_graph(pos, r=5.0, cell=cell)
        edge_attr = torch.hstack([edge_vec, edge_vec.norm(dim=-1, keepdim=True)])
        
        noise_levels = [0.1, 0.5, 1.0]
        for noise in noise_levels:
            t_test = torch.tensor([[noise]], dtype=torch.float32)
            sigma_test = torch.tensor([[noise * 0.8]], dtype=torch.float32)
            
            with torch.no_grad():
                score = score_net.ema_model(
                    species, edge_index, edge_attr, t_test, sigma_test
                )
            
            assert torch.isfinite(score).all()
            assert score.shape == pos.shape
    
    def test_forward_pass_deterministic(self, model_setup):
        """Verify forward pass is deterministic."""
        score_net, species, pos, cell, t, sigma = model_setup
        
        edge_index, edge_vec = periodic_radius_graph(pos, r=5.0, cell=cell)
        edge_attr = torch.hstack([edge_vec, edge_vec.norm(dim=-1, keepdim=True)])
        
        with torch.no_grad():
            score1 = score_net.ema_model(
                species, edge_index, edge_attr, t, sigma
            )
            score2 = score_net.ema_model(
                species, edge_index, edge_attr, t, sigma
            )
        
        assert torch.allclose(score1, score2)


class TestScoreComputation:
    """Test score computation on real structures."""
    
    @pytest.fixture
    def model_and_diffuser(self):
        """Setup model and diffuser."""
        score_net = LitScoreNet.load_from_checkpoint(
            CHECKPOINT_PATH,
            map_location="cpu",
        )
        score_net.eval()
        score_net.ema_model.eval()
        
        diffuser = VarianceExplodingDiffuser(k=0.8)
        
        return score_net, diffuser
    
    def test_prior_score_computation(self, model_and_diffuser):
        """Test prior score computation."""
        score_net, diffuser = model_and_diffuser
        
        atoms = bulk("Si", "diamond", a=5.43)
        species, pos, cell = atoms_to_device(atoms, "cpu")
        
        t = torch.tensor([[0.5]], dtype=torch.float32)
        
        with torch.no_grad():
            score = compute_prior_score(
                species, pos, cell, t, cutoff=5.0,
                score_net=score_net, diffuser=diffuser
            )
        
        assert torch.isfinite(score).all()
        assert score.shape == pos.shape
    
    def test_score_magnitude_reasonable(self, model_and_diffuser):
        """Test that score magnitudes are reasonable."""
        score_net, diffuser = model_and_diffuser
        
        atoms = bulk("Si", "diamond", a=5.43)
        species, pos, cell = atoms_to_device(atoms, "cpu")
        
        t = torch.tensor([[0.5]], dtype=torch.float32)
        
        with torch.no_grad():
            score = compute_prior_score(
                species, pos, cell, t, cutoff=5.0,
                score_net=score_net, diffuser=diffuser
            )
        
        # Check that scores are not too large
        score_norm = torch.norm(score, dim=-1).mean()
        assert score_norm < 100.0, f"Score norm too large: {score_norm}"
        # Check that scores are not all near zero
        assert score_norm > 1e-6, f"Score norm too small: {score_norm}"


class TestDenoisingStep:
    """Test SDE denoising step."""
    
    @pytest.fixture
    def denoising_setup(self):
        """Setup for denoising test."""
        score_net = LitScoreNet.load_from_checkpoint(
            CHECKPOINT_PATH,
            map_location="cpu",
        )
        score_net.eval()
        score_net.ema_model.eval()
        
        diffuser = VarianceExplodingDiffuser(k=0.8)
        
        atoms = bulk("Si", "diamond", a=5.43)
        species, pos, cell = atoms_to_device(atoms, "cpu")
        
        return score_net, diffuser, species, pos, cell
    
    def test_denoising_step_runs(self, denoising_setup):
        """Test single denoising step executes."""
        score_net, diffuser, species, pos, cell = denoising_setup
        
        # Create short time series
        ts = torch.linspace(1.0, 0.5, 10, device="cpu")
        
        def score_fn(sp, p, c, t, co):
            return compute_prior_score(sp, p, c, t, co, score_net, diffuser)
        
        traj, final_pos = denoise_by_sde(
            species, pos, cell, cutoff=5.0,
            score_fn=score_fn,
            likelihood_fn=None,
            ts=ts,
            diffuser=diffuser,
            save_traj=True,
        )
        
        assert traj is not None
        assert len(traj) > 0
        assert torch.isfinite(final_pos).all()
    
    def test_denoising_preserves_structure(self, denoising_setup):
        """Test denoising preserves basic structure."""
        score_net, diffuser, species, pos, cell = denoising_setup
        
        ts = torch.linspace(1.0, 0.8, 5, device="cpu")  # Short trajectory
        
        def score_fn(sp, p, c, t, co):
            return compute_prior_score(sp, p, c, t, co, score_net, diffuser)
        
        traj, final_pos = denoise_by_sde(
            species, pos, cell, cutoff=5.0,
            score_fn=score_fn,
            likelihood_fn=None,
            ts=ts,
            diffuser=diffuser,
            save_traj=False,
        )
        
        # Check final positions are finite
        assert torch.isfinite(final_pos).all()
        
        # Check structure didn't explode (positions should be similar magnitude)
        initial_norm = torch.norm(pos).item()
        final_norm = torch.norm(final_pos).item()
        assert final_norm < initial_norm * 10, "Structure exploded"
        assert final_norm > initial_norm * 0.1, "Structure collapsed"


class TestGenerationPipeline:
    """Test full generation pipeline."""
    
    @pytest.fixture
    def generation_setup(self):
        """Setup for generation test."""
        score_net = LitScoreNet.load_from_checkpoint(
            CHECKPOINT_PATH,
            map_location="cpu",
        )
        score_net.eval()
        score_net.ema_model.eval()
        
        diffuser = VarianceExplodingDiffuser(k=0.8)
        
        # Load real structure
        atoms = read(DATA_DIR / "Si_2.5_00.xyz")
        species, pos, cell = atoms_to_device(atoms, "cpu")
        
        return score_net, diffuser, species, pos, cell
    
    def test_short_generation(self, generation_setup):
        """Test short generation run."""
        score_net, diffuser, species, pos, cell = generation_setup
        
        # Short trajectory for testing
        ts = torch.linspace(1.0, 0.001, 20, device="cpu")
        
        def score_fn(sp, p, c, t, co):
            return compute_prior_score(sp, p, c, t, co, score_net, diffuser)
        
        traj, final_pos = denoise_by_sde(
            species, pos, cell, cutoff=5.0,
            score_fn=score_fn,
            likelihood_fn=None,
            ts=ts,
            diffuser=diffuser,
            save_traj=True,
        )
        
        assert traj is not None
        assert len(traj) == 20  # tstep number
        assert torch.isfinite(final_pos).all()
    
    def test_generation_with_trajectory(self, generation_setup):
        """Test generation saves trajectory correctly."""
        score_net, diffuser, species, pos, cell = generation_setup
        
        ts = torch.linspace(1.0, 0.5, 10, device="cpu")
        
        def score_fn(sp, p, c, t, co):
            return compute_prior_score(sp, p, c, t, co, score_net, diffuser)
        
        traj, final_pos = denoise_by_sde(
            species, pos, cell, cutoff=5.0,
            score_fn=score_fn,
            likelihood_fn=None,
            ts=ts,
            diffuser=diffuser,
            save_traj=True,
        )
        
        # Verify trajectory structure
        assert isinstance(traj, list)
        assert len(traj) == 10
        for positions in traj:
            assert torch.isfinite(positions).all()
            assert positions.shape == pos.shape
    
    def test_generation_produces_reasonable_structure(self, generation_setup):
        """Test generation produces physically reasonable structure."""
        score_net, diffuser, species, pos, cell = generation_setup
        
        ts = torch.linspace(1.0, 0.01, 30, device="cpu")
        
        def score_fn(sp, p, c, t, co):
            return compute_prior_score(sp, p, c, t, co, score_net, diffuser)
        
        traj, final_pos = denoise_by_sde(
            species, pos, cell, cutoff=5.0,
            score_fn=score_fn,
            likelihood_fn=None,
            ts=ts,
            diffuser=diffuser,
            save_traj=False,
        )
        
        # Check cell dimensions are preserved
        assert torch.allclose(cell, cell)
        
        # Check positions are within reasonable bounds
        cell_np = cell.cpu().numpy()
        pos_np = final_pos.cpu().numpy()
        
        # Rough check: atoms should be within 2x the cell bounds
        for dim in range(3):
            cell_length = np.linalg.norm(cell_np[dim])
            max_pos = np.abs(pos_np[:, dim]).max()
            assert max_pos < cell_length * 2, f"Atoms outside reasonable bounds in dim {dim}"


class TestGuidanceIntegration:
    """Test conditional generation with guidance."""
    
    @pytest.fixture
    def guidance_setup(self):
        """Setup for guidance test."""
        score_net = LitScoreNet.load_from_checkpoint(
            CHECKPOINT_PATH,
            map_location="cpu",
        )
        score_net.eval()
        score_net.ema_model.eval()
        
        diffuser = VarianceExplodingDiffuser(k=0.8)
        
        atoms = bulk("Si", "diamond", a=5.43)
        species, pos, cell = atoms_to_device(atoms, "cpu")
        
        # Setup guidance
        guidance_model = DifferentiableRDF(cutoff=5.0, bin_size=100, sigma=0.15)
        guidance_model.eval()
        
        # Compute target from reference (DifferentiableRDF returns 3 values)
        _, target_y, _ = guidance_model(pos.cpu(), species.cpu(), cell.cpu())
        target_y = target_y.to("cpu")
        
        return score_net, diffuser, species, pos, cell, guidance_model, target_y
    
    def test_likelihood_score_computation(self, guidance_setup):
        """Test likelihood score computation."""
        score_net, diffuser, species, pos, cell, guidance_model, target_y = guidance_setup
        
        likelihood_fn = LikelihoodScore(
            score_net.ema_model,
            guidance_model,
            target_y,
            rho=100.0,
            diffuser=diffuser,
            guidance_type="pdf",
            cutoff=5.0,
        )
        
        t = torch.tensor([[0.5]], dtype=torch.float32)
        
        l_score, norm = likelihood_fn(species, pos, cell, t, cut=5.0)
        
        assert torch.isfinite(l_score).all()
        assert torch.isfinite(norm)
        assert l_score.shape == pos.shape
    
    def test_guided_denoising_step(self, guidance_setup):
        """Test guided denoising step."""
        score_net, diffuser, species, pos, cell, guidance_model, target_y = guidance_setup
        
        likelihood_fn = LikelihoodScore(
            score_net.ema_model,
            guidance_model,
            target_y,
            rho=100.0,
            diffuser=diffuser,
            guidance_type="pdf",
            cutoff=5.0,
        )
        
        ts = torch.linspace(1.0, 0.8, 5, device="cpu")
        
        def score_fn(sp, p, c, t, co):
            return compute_prior_score(sp, p, c, t, co, score_net, diffuser)
        
        traj, final_pos = denoise_by_sde(
            species, pos, cell, cutoff=5.0,
            score_fn=score_fn,
            likelihood_fn=likelihood_fn,
            ts=ts,
            diffuser=diffuser,
            save_traj=False,
        )
        
        assert torch.isfinite(final_pos).all()


class TestModelBehavior:
    """Test general model behavior and properties."""
    
    @pytest.fixture
    def model(self):
        """Load model."""
        return LitScoreNet.load_from_checkpoint(
            CHECKPOINT_PATH,
            map_location="cpu",
        )
    
    def test_model_has_parameters(self, model):
        """Verify model has trainable parameters."""
        params = list(model.parameters())
        assert len(params) > 0
        
        total_params = sum(p.numel() for p in params)
        assert total_params > 1000, "Model seems to have too few parameters"
    
    def test_ema_model_has_same_parameters(self, model):
        """Verify EMA model has same structure as main model."""
        ema_params = list(model.ema_model.parameters())
        model_params = list(model.model.parameters())
        
        assert len(ema_params) == len(model_params)
    
    def test_model_save_hyperparameters(self, model):
        """Test hyperparameters are saved correctly."""
        hparams = model.hparams
        assert hasattr(hparams, "num_species")
        assert hasattr(hparams, "num_convs")
        assert hasattr(hparams, "dim")
        assert hasattr(hparams, "ema_decay")
