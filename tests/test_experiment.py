"""Tests for the unified experiment management system."""

import os
import pytest
import tempfile
import shutil
from pathlib import Path

from glass.experiment import Experiment, ExperimentConfig


class TestExperimentConfig:
    """Test ExperimentConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ExperimentConfig()
        assert config.name == "experiment"
        assert config.model_type == "score"
        assert config.num_species == 1
        assert config.num_convs == 5
        assert config.dim == 200
        assert config.ema_decay == 0.9999
        assert config.checkpoint == "best"
    
    def test_config_update(self):
        """Test updating configuration."""
        config = ExperimentConfig()
        config.update(name="test", num_species=2, max_epochs=100)
        assert config.name == "test"
        assert config.num_species == 2
        assert config.max_epochs == 100
        # Other values should remain
        assert config.dim == 200
    
    def test_config_yaml_roundtrip(self):
        """Test saving and loading config from YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            
            config = ExperimentConfig()
            config.name = "test_roundtrip"
            config.num_species = 3
            config.model_type = "spec"
            config.spec_type = "exafs"
            
            config.to_yaml(config_path)
            loaded = ExperimentConfig.from_yaml(config_path)
            
            assert loaded.name == "test_roundtrip"
            assert loaded.num_species == 3
            assert loaded.model_type == "spec"
            assert loaded.spec_type == "exafs"


    def test_default_params_yaml_matches_dataclass(self):
        """glass/default_params.yaml must mirror ExperimentConfig exactly."""
        import yaml
        from dataclasses import asdict

        import glass

        path = Path(glass.__file__).parent / "default_params.yaml"
        with open(path) as fh:
            from_yaml = yaml.safe_load(fh)
        assert from_yaml == asdict(ExperimentConfig())

    def test_from_yaml_ignores_unknown_keys(self, tmp_path):
        """Old config.yaml files with removed keys still load."""
        import yaml

        path = tmp_path / "config.yaml"
        with open(path, "w") as fh:
            yaml.dump({"dim": 64, "entropy_guidance": False}, fh)
        with pytest.warns(UserWarning, match="entropy_guidance"):
            loaded = ExperimentConfig.from_yaml(path)
        assert loaded.dim == 64

    def test_preprint_generation_defaults(self):
        config = ExperimentConfig()
        assert (config.uncond_tmax, config.uncond_tmin, config.uncond_tstep) == (1.0, 1e-3, 512)
        assert (config.tmax, config.tmin, config.tstep) == (0.6, 1e-3, 512)
        assert config.uncond_prepass and config.n_restart == 1
        assert not config.tersoff_guidance and config.n_corr == 0
        assert config.sa_n_steps == 0


    def test_default_rho_covers_every_guidance_type(self):
        from glass.cli.generate import generate
        from glass.experiment import DEFAULT_RHO

        choice = next(p for p in generate.params if p.name == "guidance_type")
        assert set(DEFAULT_RHO) == set(choice.type.choices)
        assert ExperimentConfig().rho is None


class TestExperiment:
    """Test Experiment class."""
    
    def test_create_structure(self):
        """Test creating experiment directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(tmpdir, create=True)
            
            assert exp.root.exists()
            assert exp.data_dir.exists()
            assert exp.checkpoints_dir.exists()
            assert exp.inits_dir.exists()
            assert exp.outputs_dir.exists()
            assert exp.logs_dir.exists()
    
    def test_exists_property(self):
        """Test exists property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test non-existent subdirectory
            non_existent = os.path.join(tmpdir, "non_existent")
            exp = Experiment(non_existent)
            assert not exp.exists
            
            exp._create_structure()
            assert exp.exists
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(tmpdir, create=True)
            
            config = ExperimentConfig(name="test_save", num_species=2)
            exp.save_config(config)
            
            assert exp.config_path.exists()
            
            loaded = exp.load_config()
            assert loaded.name == "test_save"
            assert loaded.num_species == 2
    
    def test_load_config_not_found(self):
        """Test loading config when it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(tmpdir)
            with pytest.raises(FileNotFoundError):
                exp.load_config()
    
    def test_get_data_files_flat_structure(self):
        """Test finding data files in flat structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(tmpdir, create=True)
            
            # Create some xyz files
            (exp.data_dir / "struct1.xyz").touch()
            (exp.data_dir / "struct2.xyz").touch()
            
            files = exp.get_data_files()
            assert len(files) == 2
            assert all(f.suffix == ".xyz" for f in files)
    
    def test_get_data_files_old_structure(self):
        """Test finding data files in old structure fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(tmpdir, create=True)
            
            # Create old structure
            structures_dir = exp.data_dir / "structures"
            structures_dir.mkdir()
            (structures_dir / "train" / "struct1.xyz").parent.mkdir(parents=True, exist_ok=True)
            (structures_dir / "train" / "struct1.xyz").touch()
            
            files = exp.get_data_files()
            assert len(files) == 1
    
    def test_get_init_files(self):
        """Test finding initialization files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(tmpdir, create=True)
            
            (exp.inits_dir / "init1.xyz").touch()
            (exp.inits_dir / "init2.xyz").touch()
            
            files = exp.get_init_files()
            assert len(files) == 2
    
    def test_find_checkpoint_best(self):
        """Test finding best checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(tmpdir, create=True)
            
            # Create best checkpoint
            best_ckpt = exp.checkpoints_dir / "best.ckpt"
            best_ckpt.touch()
            
            found = exp.find_checkpoint("best")
            assert found == best_ckpt
    
    def test_find_checkpoint_last(self):
        """Test finding last checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(tmpdir, create=True)
            
            # Create last checkpoint
            last_ckpt = exp.checkpoints_dir / "last.ckpt"
            last_ckpt.touch()
            
            found = exp.find_checkpoint("last")
            assert found == last_ckpt
    
    def test_find_checkpoint_specific(self):
        """Test finding specific checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(tmpdir, create=True)
            
            # Create specific checkpoint
            specific_ckpt = exp.checkpoints_dir / "epoch_0050.ckpt"
            specific_ckpt.touch()
            
            found = exp.find_checkpoint("epoch_0050.ckpt")
            assert found == specific_ckpt
    
    def test_find_checkpoint_not_found(self):
        """Test finding checkpoint when it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(tmpdir, create=True)
            
            with pytest.raises(FileNotFoundError):
                exp.find_checkpoint("best")
    
    def test_list_checkpoints(self):
        """Test listing all checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(tmpdir, create=True)
            
            (exp.checkpoints_dir / "best.ckpt").touch()
            (exp.checkpoints_dir / "last.ckpt").touch()
            (exp.checkpoints_dir / "epoch_0001.ckpt").touch()
            
            checkpoints = exp.list_checkpoints()
            assert len(checkpoints) == 3
    
    def test_get_data_dir_for_datamodule(self):
        """Test getting data directory path for datamodule."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(tmpdir, create=True)
            
            data_dir = exp.get_data_dir_for_datamodule()
            assert data_dir.endswith("/")
            assert str(exp.data_dir) in data_dir


class TestExperimentIntegration:
    """Integration tests for experiment workflows."""
    
    def test_full_workflow_creation(self):
        """Test creating a complete experiment with data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create experiment
            exp = Experiment(tmpdir, create=True)
            
            # Create config
            config = ExperimentConfig(
                name="Si_test",
                model_type="score",
                num_species=1,
                max_epochs=10,
            )
            exp.save_config(config)
            
            # Add data files
            (exp.data_dir / "structure_01.xyz").touch()
            (exp.data_dir / "structure_02.xyz").touch()
            
            # Add init files
            (exp.inits_dir / "init_01.xyz").touch()
            
            # Verify
            assert len(exp.get_data_files()) == 2
            assert len(exp.get_init_files()) == 1
            assert exp.load_config().name == "Si_test"
    
    def test_experiment_repr(self):
        """Test string representation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment(tmpdir)
            assert "Experiment" in repr(exp)
            assert tmpdir in repr(exp)
