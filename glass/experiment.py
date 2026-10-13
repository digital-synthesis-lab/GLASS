"""Unified experiment management for glass training and generation.

This module provides a simplified interface for managing machine learning experiments,
organizing data, checkpoints, logs, and outputs in a single directory structure.
"""

import os
import glob
import warnings
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict, field, fields as dataclass_fields


# Guidance weight used when ``rho`` is not set, per guidance type. Chosen
# with the amorphous Si example (examples/silicon, README): PDF and ADF
# targets are matched closely at these values without distorting the local
# structure; the diffraction intensities need much weaker guidance.
DEFAULT_RHO = {
    "pdf": 300.0,
    "adf": 1000.0,
    "xrd": 1.0,
    "nd": 1.0,
    "exafs": 300.0,
    "xanes": 300.0,
}


@dataclass
class ExperimentConfig:
    """Configuration for a glass experiment.
    
    This dataclass stores all parameters needed for training or generation,
    with sensible defaults for most options.
    """
    # Experiment metadata
    name: str = "experiment"
    model_type: str = "score"  # "score" or "spec"
    
    # Model architecture
    num_species: int = 1
    num_convs: int = 5
    dim: int = 200
    ema_decay: float = 0.9999
    
    # Training parameters. Each epoch draws ``dup`` noisy copies of every
    # training structure. Batches should hold on the order of a thousand atoms
    # or more: with a single 216-atom cell per step the score model stays at
    # the trivial zero-output solution (loss = 3).
    max_epochs: int = 8000
    batch_size: int = 16
    learning_rate: float = 1e-3
    cutoff: float = 5.0
    k: float = 0.8
    dup: int = 64
    train_size: float = 0.9
    scale_y: float = 1.0
    num_workers: int = 8
    
    # Spec model specific
    spec_type: Optional[str] = None  # "exafs" or "xanes"
    out_dim: Optional[int] = None  # 400 for exafs, 100 for xanes
    
    # Checkpoint settings
    save_top_k: int = 3

    # Generation parameters. The defaults follow the protocol of the preprint
    # (arXiv:2603.23210): an unconditional pass from t=1.0 to t=0.001 in 512
    # steps, followed (for guided runs) by a guided pass from t=0.6 to
    # t=0.001 in 512 steps. No empirical potential is used while sampling.
    checkpoint: str = "best"  # "best", "last", or specific filename
    n_runs: int = 10
    save_traj: bool = False
    device: str = "cuda:0"

    # Unconditional pass (unconditional runs and prepass of guided runs)
    uncond_tmin: float = 1.0e-3
    uncond_tmax: float = 1.0
    uncond_tstep: int = 512
    # Run the unconditional pass before the guided pass.
    uncond_prepass: bool = True

    # Guided pass
    tmin: float = 1.0e-3
    tmax: float = 0.6
    tstep: int = 512

    # Power-law exponent of the time grid (1.0 = uniform steps)
    t_schedule_rho: float = 1.0

    # Guidance weight (w in the preprint); None selects DEFAULT_RHO[guidance]
    rho: Optional[float] = None

    # Guidance-specific parameters
    pdf_cutoff: float = 8.0  # PDF, maximum r (Å)
    bin_size: int = 100  # PDF
    angle_bins: int = 100  # ADF
    adf_sigma: float = 0.1  # ADF
    adf_cutoff: float = 3.5  # ADF
    element_names: list = field(default_factory=list)  # XRD/ND
    qmin: float = 1.0  # XRD/ND
    qmax: float = 20.0  # XRD/ND
    qstep: float = 0.1  # XRD/ND
    biso: float = 1.5  # XRD/ND

    # [exafs/xanes] Compare atom-averaged spectra instead of atom i with atom i
    xas_average: bool = False

    # Optional hard minimum-distance veto (Å); None or 0 disables it.
    min_dist: Optional[float] = None

    # Scale factor on the prior score at each SDE step.
    # 1.0 = unmodified prior; 0.0 turns off the prior entirely.
    lambda_prior: float = 1.0

    # Number of guided passes; each pass starts from the previous output.
    n_restart: int = 1

    # Optional Tersoff-energy guidance (pure Si only). Off in the preprint
    # protocol; see examples/silicon/hpo_params.yaml for tuned settings.
    tersoff_guidance: bool = False
    tersoff_lambda: float = 0.30
    tersoff_schedule: str = "sigmoid"
    tersoff_t_gate: float = 0.50
    tersoff_clamp: float = 10.0

    # Optional Langevin corrector (0 disables it)
    n_corr: int = 0
    corr_step_size: float = 0.50
    corr_use_tersoff: bool = True
    corr_t_gate: float = 0.50

    # Optional simulated-annealing tail on the Tersoff PES (0 disables it)
    sa_n_steps: int = 0
    sa_T0: float = 1e-2
    sa_T_end: float = 1e-5
    sa_lr: float = 1e-3
    sa_lr_clamp: float = 0.2

    # Infrastructure
    accelerator: str = "gpu"
    strategy: str = "ddp_find_unused_parameters_true"
    matmul_precision: str = "medium"
    refresh_rate: int = 10
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        valid_keys = {f.name for f in dataclass_fields(cls)}
        unknown = sorted(set(data) - valid_keys)
        if unknown:
            warnings.warn(
                f"Ignoring unknown keys in {path}: {', '.join(unknown)}"
            )
        return cls(**{k: v for k, v in data.items() if k in valid_keys})
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
    
    def update(self, **kwargs) -> "ExperimentConfig":
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def update_from_yaml(self, path: Union[str, Path]) -> "ExperimentConfig":
        """Apply partial YAML overrides.

        Only keys present in the file are updated; unmentioned fields keep
        their current values.  Unknown keys are silently ignored so that
        partial override files stay forward-compatible.
        """
        valid_keys = {f.name for f in dataclass_fields(self)}
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        for key, value in data.items():
            if key in valid_keys:
                setattr(self, key, value)
        return self


class Experiment:
    """Manages experiment directory structure and resources.
    
    An experiment is organized as:
        experiment_root/
        ├── config.yaml          # Experiment configuration
        ├── data/                # Training data (.xyz files)
        ├── checkpoints/         # Model checkpoints
        │   ├── best.ckpt
        │   ├── last.ckpt
        │   └── epoch=*.ckpt
        ├── inits/               # Initial structures for generation
        ├── outputs/             # Generated structures
        └── logs/                # TensorBoard logs
            └── version_*/
    
    Args:
        root: Path to experiment directory (created if doesn't exist)
        create: If True, create directory structure on init
    """
    
    def __init__(self, root: Union[str, Path], create: bool = False):
        self.root = Path(root).resolve()
        self.config_path = self.root / "config.yaml"
        
        # Define subdirectories
        self.data_dir = self.root / "data"
        self.checkpoints_dir = self.root / "checkpoints"
        self.inits_dir = self.root / "inits"
        self.outputs_dir = self.root / "outputs"
        self.logs_dir = self.root / "logs"
        
        # Config cache
        self._config: Optional[ExperimentConfig] = None
        
        if create:
            self._create_structure()
    
    def _create_structure(self) -> None:
        """Create experiment directory structure."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.inits_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
    
    @property
    def exists(self) -> bool:
        """Check if experiment directory exists."""
        return self.root.exists()
    
    def load_config(self) -> ExperimentConfig:
        """Load configuration from config.yaml."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        self._config = ExperimentConfig.from_yaml(self.config_path)
        return self._config
    
    def save_config(self, config: Optional[ExperimentConfig] = None) -> None:
        """Save configuration to config.yaml."""
        if config is None:
            config = self._config
        if config is None:
            raise ValueError("No config to save")
        self.root.mkdir(parents=True, exist_ok=True)
        config.to_yaml(self.config_path)
        self._config = config
    
    def get_config(self) -> ExperimentConfig:
        """Get cached or loaded config."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def find_checkpoint(self, checkpoint: str = "best") -> Path:
        """Find checkpoint file.
        
        Args:
            checkpoint: "best", "last", or specific filename/pattern
            
        Returns:
            Path to checkpoint file
            
        Raises:
            FileNotFoundError: If no matching checkpoint found
        """
        if checkpoint == "best":
            path = self.checkpoints_dir / "best.ckpt"
            if path.exists():
                return path
            # Fallback: find checkpoint with "best" in name
            candidates = sorted(self.checkpoints_dir.glob("*best*.ckpt"))
            if candidates:
                return candidates[-1]
            # Final fallback: any checkpoint sorted by name
            candidates = sorted(self.checkpoints_dir.glob("*.ckpt"))
            if candidates:
                return candidates[-1]
                
        elif checkpoint == "last":
            path = self.checkpoints_dir / "last.ckpt"
            if path.exists():
                return path
            # Fallback: find checkpoint with "last" in name
            candidates = sorted(self.checkpoints_dir.glob("*last*.ckpt"))
            if candidates:
                return candidates[-1]
            # Final fallback: any checkpoint sorted by name
            candidates = sorted(self.checkpoints_dir.glob("*.ckpt"))
            if candidates:
                return candidates[-1]
        else:
            # Specific filename or pattern
            path = self.checkpoints_dir / checkpoint
            if path.exists():
                return path
            # Try as glob pattern
            candidates = sorted(self.checkpoints_dir.glob(checkpoint))
            if candidates:
                return candidates[-1]
        
        raise FileNotFoundError(
            f"No checkpoint found for '{checkpoint}' in {self.checkpoints_dir}"
        )
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        return sorted(self.checkpoints_dir.glob("*.ckpt"))
    
    def get_init_files(self, init_dir: Optional[Union[str, Path]] = None) -> list:
        """Get list of initialization .xyz files.
        
        Args:
            init_dir: Override directory (default: self.inits_dir)
            
        Returns:
            List of Path objects to .xyz files
        """
        search_dir = Path(init_dir) if init_dir else self.inits_dir
        if not search_dir.exists():
            return []
        return sorted(search_dir.glob("*.xyz"))
    
    def get_data_files(self) -> list:
        """Get list of training data .xyz files.
        
        Searches in order:
        1. data_dir/*.xyz (flat structure)
        2. data_dir/structures/*.xyz (old structure)
        3. data_dir/structures/train/*.xyz (old structure)
        
        Returns:
            List of Path objects to .xyz files
        """
        # Flat structure (preferred)
        files = sorted(self.data_dir.glob("*.xyz"))
        if files:
            return files
        
        # Old structure fallbacks
        files = sorted(self.data_dir.glob("structures/*.xyz"))
        if files:
            return files
            
        files = sorted(self.data_dir.glob("structures/train/*.xyz"))
        return files
    
    def get_data_dir_for_datamodule(self) -> str:
        """Get data directory path suitable for StructureSpecDataModule.
        
        The datamodule expects a directory that may contain subdirectories
        for different data types. We return the data_dir with trailing slash.
        """
        return str(self.data_dir) + "/"
    
    def __repr__(self) -> str:
        return f"Experiment('{self.root}')"
