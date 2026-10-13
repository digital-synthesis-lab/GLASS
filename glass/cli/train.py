import os
import click
import yaml
import torch
import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from glass.experiment import Experiment, ExperimentConfig
from glass.lit.datamodules import StructureSpecDataModule
from glass.lit.modules import LitScoreNet, LitSpecNet


class BestCheckpoint(ModelCheckpoint):
    """ModelCheckpoint with unique state key for best checkpoint."""
    @property
    def state_key(self) -> str:
        return "best_checkpoint"


class LastCheckpoint(ModelCheckpoint):
    """ModelCheckpoint with unique state key for last checkpoint."""
    @property
    def state_key(self) -> str:
        return "last_checkpoint"


@click.command(
    "train",
    help="""
Train a score-based generative model or spectral surrogate model.

This unified command creates a complete experiment with organized directories
for data, checkpoints, logs, and outputs.

\b
EXPERIMENT STRUCTURE:
    ./my_experiment/
    ├── config.yaml          # All training parameters
    ├── data/                # Training structures (*.xyz files)
    ├── checkpoints/         # Model checkpoints
    │   ├── best.ckpt       # Best validation checkpoint
    │   ├── last.ckpt       # Most recent checkpoint
    │   └── epoch_*.ckpt    # Intermediate checkpoints
    └── logs/               # TensorBoard logs
        └── version_*/

EXAMPLES:

\b
  # Create new experiment and train
  glass train ./my_experiment --model-type score --num-species 2

\b
  # Resume training from checkpoint
  glass train ./my_experiment --resume

\b
  # Train spectral surrogate (EXAFS)
  glass train ./my_experiment --model-type spec --spec-type exafs --num-species 1

\b
  # Override specific parameters
  glass train ./my_experiment --max-epochs 5000 --lr 0.0005

KEY PARAMETERS:

\b
  --model-type     "score" (structure generation) or "spec" (spectral surrogate)
  --num-species    Number of atomic species in the system
  --cutoff         Graph cutoff radius (Å), default: 5.0
  --k              Noise level for diffusion, default: 0.8
  --dim            Hidden dimension, default: 200
  --n-conv         Number of convolution layers, default: 5
  --ema-decay      EMA decay for model stability, default: 0.9999
""",
)
@click.argument("experiment_path", type=click.Path())
@click.option(
    "--model-type",
    type=click.Choice(["score", "spec"]),
    default=None,
    help="Model type: 'score' for structure generation, 'spec' for spectral surrogate.",
)
@click.option(
    "--num-species",
    type=int,
    default=None,
    help="Number of atomic species.",
)
@click.option(
    "--spec-type",
    type=click.Choice(["exafs", "xanes"]),
    default=None,
    help="[spec model only] Spectral type.",
)
@click.option(
    "--resume/--new",
    "resume",
    default=False,
    help="Resume from last checkpoint or start new training.",
)
@click.option(
    "--cutoff",
    type=float,
    default=None,
    help="Graph cutoff radius (Å).",
)
@click.option(
    "--k",
    type=float,
    default=None,
    help="Maximum noise level.",
)
@click.option(
    "--dup",
    type=int,
    default=None,
    help="Dataset duplication factor.",
)
@click.option(
    "--train-size",
    type=float,
    default=None,
    help="Train/validation split fraction.",
)
@click.option(
    "--scale-y",
    type=float,
    default=None,
    help="Scale factor for spectroscopy curve.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Batch size.",
)
@click.option(
    "--num-workers",
    type=int,
    default=None,
    help="Number of dataloader workers.",
)
@click.option(
    "--n-conv",
    type=int,
    default=None,
    help="Number of convolution layers.",
)
@click.option(
    "--dim",
    type=int,
    default=None,
    help="Hidden dimension.",
)
@click.option(
    "--ema-decay",
    type=float,
    default=None,
    help="EMA decay.",
)
@click.option(
    "--lr",
    "--learn-rate",
    type=float,
    default=None,
    help="Learning rate.",
)
@click.option(
    "--max-epochs",
    type=int,
    default=None,
    help="Maximum number of training epochs.",
)
@click.option(
    "--out-dim",
    type=int,
    default=None,
    help="[spec model only] Output dimension (default: 400 for exafs, 100 for xanes).",
)
@click.option(
    "--accelerator",
    type=str,
    default=None,
    help="Lightning accelerator.",
)
@click.option(
    "--strategy",
    type=str,
    default=None,
    help="Lightning distributed training strategy.",
)
@click.option(
    "--refresh-rate",
    type=int,
    default=None,
    help="Progress bar refresh rate.",
)
@click.option(
    "--matmul-precision",
    type=click.Choice(["highest", "high", "medium"]),
    default=None,
    help="Torch float32 matmul precision.",
)
@click.option(
    "--save-top-k",
    type=int,
    default=None,
    help="Number of best checkpoints to keep.",
)
@click.option(
    "--params",
    "params_file",
    type=click.Path(exists=True),
    default=None,
    help="YAML file with hyperparameters to override experiment config. "
         "Only keys present in the file are applied; explicit CLI flags take precedence.",
)
def train(
    experiment_path,
    model_type,
    num_species,
    spec_type,
    resume,
    cutoff,
    k,
    dup,
    train_size,
    scale_y,
    batch_size,
    num_workers,
    n_conv,
    dim,
    ema_decay,
    lr,
    max_epochs,
    out_dim,
    accelerator,
    strategy,
    refresh_rate,
    matmul_precision,
    save_top_k,
    params_file,
):
    """Train a model with unified experiment structure."""

    # Initialize experiment
    experiment = Experiment(experiment_path)

    # Load or create config. An existing config.yaml always wins: `resume` only
    # decides whether last.ckpt is fed to trainer.fit (below), and rebuilding a
    # default ExperimentConfig here would resume a trained checkpoint with the
    # wrong architecture (default dim/num_convs).
    config_had_max_epochs = False
    is_new_experiment = not experiment.config_path.exists()
    if experiment.config_path.exists():
        click.echo(f"Loading existing config from {experiment.config_path}")
        with open(experiment.config_path, "r") as fh:
            existing_data = yaml.safe_load(fh) or {}
        config_had_max_epochs = "max_epochs" in existing_data
        config = experiment.load_config()
    else:
        # Validate required parameters for new experiment
        if model_type is None:
            raise click.ClickException(
                "--model-type is required for new experiments (score or spec)"
            )
        if num_species is None:
            raise click.ClickException(
                "--num-species is required for new experiments"
            )
        if model_type == "spec" and spec_type is None:
            raise click.ClickException(
                "--spec-type is required for spec models (exafs or xanes)"
            )
        
        # Create new experiment structure
        experiment._create_structure()
        config = ExperimentConfig()
        config.model_type = model_type
        config.num_species = num_species
        if spec_type:
            config.spec_type = spec_type
            config.out_dim = 400 if spec_type == "exafs" else 100
    
    # Apply params-file overrides (between experiment config and explicit CLI flags)
    if params_file:
        with open(params_file, "r") as fh:
            params_data = yaml.safe_load(fh) or {}
        if "max_epochs" in params_data:
            config_had_max_epochs = True
        config.update_from_yaml(params_file)
        click.echo(f"Applied params overrides from {params_file}")

    # Apply CLI overrides
    cli_overrides = {}
    if is_new_experiment:
        # The model identity given on the command line wins over --params.
        cli_overrides["model_type"] = model_type
        cli_overrides["num_species"] = num_species
        if spec_type:
            cli_overrides["spec_type"] = spec_type
            cli_overrides["out_dim"] = 400 if spec_type == "exafs" else 100
    if cutoff is not None:
        cli_overrides["cutoff"] = cutoff
    if k is not None:
        cli_overrides["k"] = k
    if dup is not None:
        cli_overrides["dup"] = dup
    if train_size is not None:
        cli_overrides["train_size"] = train_size
    if scale_y is not None:
        cli_overrides["scale_y"] = scale_y
    if batch_size is not None:
        cli_overrides["batch_size"] = batch_size
    if num_workers is not None:
        cli_overrides["num_workers"] = num_workers
    if n_conv is not None:
        cli_overrides["num_convs"] = n_conv
    if dim is not None:
        cli_overrides["dim"] = dim
    if ema_decay is not None:
        cli_overrides["ema_decay"] = ema_decay
    if lr is not None:
        cli_overrides["learning_rate"] = lr
    if max_epochs is not None:
        cli_overrides["max_epochs"] = max_epochs
    if out_dim is not None:
        cli_overrides["out_dim"] = out_dim
    if accelerator is not None:
        cli_overrides["accelerator"] = accelerator
    if strategy is not None:
        cli_overrides["strategy"] = strategy
    if refresh_rate is not None:
        cli_overrides["refresh_rate"] = refresh_rate
    if matmul_precision is not None:
        cli_overrides["matmul_precision"] = matmul_precision
    if save_top_k is not None:
        cli_overrides["save_top_k"] = save_top_k
    
    config.update(**cli_overrides)
    
    # Set default max_epochs for spec models, unless a CLI flag or the loaded
    # config.yaml already supplied one (the dataclass default of max_epochs can't be
    # used as a sentinel, so presence is tracked from the raw YAML keys).
    if config.model_type == "spec" and max_epochs is None and not config_had_max_epochs:
        if config.spec_type == "exafs":
            config.max_epochs = 8000
        elif config.spec_type == "xanes":
            config.max_epochs = 3000
    
    # Save updated config
    experiment.save_config(config)
    
    # Set torch precision
    if config.matmul_precision:
        torch.set_float32_matmul_precision(config.matmul_precision)
    
    click.echo(f"Experiment: {experiment.root}")
    click.echo(f"Model type: {config.model_type}")
    click.echo(f"Number of species: {config.num_species}")
    if config.model_type == "spec":
        click.echo(f"Spec type: {config.spec_type}")
        click.echo(f"Output dim: {config.out_dim}")
    
    # Get data files
    data_files = experiment.get_data_files()
    if not data_files:
        raise click.ClickException(
            f"No .xyz files found in {experiment.data_dir}\n"
            "Place training structures (*.xyz) in the data/ folder."
        )
    click.echo(f"Found {len(data_files)} training structures")
    
    # Setup datamodule
    guide_type = config.spec_type if config.model_type == "spec" else None
    datamodule = StructureSpecDataModule(
        data_dir=experiment.get_data_dir_for_datamodule(),
        cutoff=config.cutoff,
        train_prior=True,
        k=config.k,
        train_size=config.train_size,
        scale_y=config.scale_y,
        dup=config.dup,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        guide_type=guide_type,
    )
    
    # Setup model
    checkpoint_path = None
    if resume:
        try:
            checkpoint_path = experiment.find_checkpoint("last")
            click.echo(f"Resuming from checkpoint: {checkpoint_path}")
        except FileNotFoundError:
            click.echo("No checkpoint found, starting fresh training")
    
    if config.model_type == "score":
        model = LitScoreNet(
            num_species=config.num_species,
            num_convs=config.num_convs,
            dim=config.dim,
            ema_decay=config.ema_decay,
            learn_rate=config.learning_rate,
        )
        name = "score"
    else:  # spec
        model = LitSpecNet(
            num_species=config.num_species,
            num_convs=config.num_convs,
            dim=config.dim,
            out_dim=config.out_dim,
            ema_decay=config.ema_decay,
            learn_rate=config.learning_rate,
        )
        name = f"{config.spec_type}"
    
    if checkpoint_path:
        model = model.__class__.load_from_checkpoint(checkpoint_path)
    
    # Setup checkpoint callbacks with unique state keys
    # Main checkpoint callback - saves top k checkpoints by epoch number
    checkpoint_callback = ModelCheckpoint(
        dirpath=experiment.checkpoints_dir,
        filename="{epoch:04d}",
        save_top_k=config.save_top_k,
        monitor="train_loss",
        mode="min",
    )
    
    # Best checkpoint with fixed name - uses custom class for unique state_key
    best_checkpoint_callback = BestCheckpoint(
        dirpath=experiment.checkpoints_dir,
        filename="best",
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        # Overwrite best.ckpt when resuming instead of writing best-v1.ckpt
        enable_version_counter=False,
    )
    
    # Last checkpoint with fixed name - uses custom class for unique state_key
    last_checkpoint_callback = LastCheckpoint(
        dirpath=experiment.checkpoints_dir,
        filename="last",
        save_top_k=1,
        monitor="step",
        mode="max",
        every_n_epochs=1,
        enable_version_counter=False,
    )
    
    # Setup trainer
    trainer = L.Trainer(
        accelerator=config.accelerator,
        max_epochs=config.max_epochs,
        logger=TensorBoardLogger(
            save_dir=experiment.logs_dir,
            name="",
            version="0",
        ),
        callbacks=[
            TQDMProgressBar(refresh_rate=config.refresh_rate),
            checkpoint_callback,
            best_checkpoint_callback,
            last_checkpoint_callback,
        ],
        strategy=config.strategy,
    )
    
    # Train
    trainer.fit(model, datamodule, ckpt_path=checkpoint_path)
    
    click.echo(f"\nTraining complete!")
    click.echo(f"Checkpoints saved to: {experiment.checkpoints_dir}")
    click.echo(f"Logs saved to: {experiment.logs_dir}")
