import os
import click
import torch
import numpy as np
from tqdm import tqdm

from glass.experiment import DEFAULT_RHO, Experiment
from glass.diffusion import VarianceExplodingDiffuser
from glass.lit.modules import LitScoreNet
from glass.diffusion.sampling import denoise_by_sde
from glass.lit.modules.likelihood import LikelihoodScore
from glass.lit.modules.guidance import create_guidance_model, load_experimental_data
from glass.lit.modules.tersoff_guidance import (
    TersoffEnergyGuidance,
    TersoffSchedule,
)
from glass.diffusion.schedules import power_law_ts
from glass.diffusion.annealing import make_anneal_fn
from glass.utils.atoms import (
    atoms_to_device,
    compute_prior_score,
    compute_target_from_reference,
)


@click.command(
    "generate",
    help="""
Generate atomic structures using a trained score model.

Runs denoising from initial structures to generate new configurations.
Supports both unconditional and conditional (guided) generation.

Unconditional runs use the --uncond-tmax/--uncond-tmin/--uncond-tstep
schedule. Guided runs first do one unconditional pass with that schedule
(disable with --no-uncond-prepass) and then a guided pass with the
--tmax/--tmin/--tstep schedule. Every frame of every *.xyz file in --inits
is used as an initial structure.

\b
EXPERIMENT STRUCTURE:
    ./my_experiment/
    ├── config.yaml          # Training configuration
    ├── checkpoints/         # Model checkpoints
    │   ├── best.ckpt
    │   ├── last.ckpt
    │   └── epoch=*.ckpt
    ├── inits/               # Initial structures (*.xyz)
    └── outputs/             # Generated structures

EXAMPLES:

\b
  # Unconditional generation (uses best checkpoint by default)
  glass generate ./my_experiment/ --inits ./my_experiment/inits/

\b
  # Use specific checkpoint
  glass generate ./my_experiment/ --inits ./inits/ --checkpoint last.ckpt

\b
  # Conditional generation with PDF guidance (averaged over all refs)
  glass generate ./my_experiment/ --inits ./inits/ \\
      --guidance-type pdf --ref-path ./reference/

\b
  # Conditional generation from a single reference structure
  glass generate ./my_experiment/ --inits ./inits/ \\
      --guidance-type pdf --ref-structure ./reference/Si_2.5_00.xyz

\b
  # Multiple runs per structure
  glass generate ./my_experiment/ --inits ./inits/ --n-runs 20

\b
GUIDANCE TYPES:
  pdf    -- Pair Distribution Function (computational or experimental)
  adf    -- Angular Distribution Function
  xrd    -- X-ray Diffraction
  nd     -- Neutron Diffraction
  exafs  -- Extended X-ray Absorption Fine Structure
  xanes  -- X-ray Absorption Near Edge Structure
""",
)
@click.argument("experiment_path", type=click.Path(exists=True))
@click.option(
    "--inits",
    type=click.Path(),
    required=True,
    help="Directory containing initial structures (*.xyz files).",
)
@click.option(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint to use: 'best', 'last', or filename (default: from config, 'best').",
)
@click.option(
    "--outdir",
    type=click.Path(),
    default=None,
    help="Output directory (default: EXPERIMENT/outputs/).",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Torch device (default: from config or cuda:0).",
)
@click.option(
    "--tmin",
    type=float,
    default=None,
    help="[guided pass] Final time of the reverse SDE.",
)
@click.option(
    "--tmax",
    type=float,
    default=None,
    help="[guided pass] Starting time of the reverse SDE.",
)
@click.option(
    "--tstep",
    type=int,
    default=None,
    help="[guided pass] Number of SDE time steps.",
)
@click.option(
    "--uncond-tmin",
    type=float,
    default=None,
    help="[unconditional pass] Final time of the reverse SDE.",
)
@click.option(
    "--uncond-tmax",
    type=float,
    default=None,
    help="[unconditional pass] Starting time of the reverse SDE.",
)
@click.option(
    "--uncond-tstep",
    type=int,
    default=None,
    help="[unconditional pass] Number of SDE time steps.",
)
@click.option(
    "--min-dist",
    type=float,
    default=None,
    help="Minimum interatomic distance (Å). Moves that bring an atom closer "
         "than this to a neighbour are rejected for that step. 0 disables.",
)
@click.option(
    "--cutoff",
    type=float,
    default=None,
    help="Graph cutoff radius (Å).",
)
@click.option(
    "--n-runs",
    type=int,
    default=None,
    help="Number of independent runs per structure.",
)
@click.option(
    "--save-traj/--no-save-traj",
    default=None,
    help="Save full trajectory or only final frame.",
)
# Conditional generation options
@click.option(
    "--guidance-type",
    type=click.Choice(["pdf", "adf", "xrd", "nd", "exafs", "xanes"]),
    default=None,
    help="Enable conditional generation with guidance.",
)
@click.option(
    "--ref-path",
    type=click.Path(),
    default=None,
    help="Directory of reference structures; all *.xyz files are averaged into one target.",
)
@click.option(
    "--ref-structure",
    type=click.Path(),
    default=None,
    help="Single reference structure file (.xyz) for computational guidance.",
)
@click.option(
    "--exp-data",
    type=click.Path(),
    default=None,
    help="JSON file with experimental data for guidance.",
)
@click.option(
    "--rho",
    type=float,
    default=None,
    help="Guidance weight (w in the preprint). Default depends on the guidance "
         "type: pdf 300, adf 1000, xrd 1, nd 1.",
)
@click.option(
    "--spec-model-path",
    type=click.Path(),
    default=None,
    help="[exafs/xanes] Path to spectral model checkpoint.",
)
@click.option(
    "--xas-average/--xas-atomwise",
    default=None,
    help="[exafs/xanes] Compare the spectrum averaged over all atoms "
         "(needed for --exp-data and for references of a different size) "
         "instead of the spectra of atoms with the same index (default).",
)
# PDF options
@click.option(
    "--pdf-cutoff",
    type=float,
    default=None,
    help="[pdf] Maximum r of the PDF (Å). Keep it below half the cell length.",
)
@click.option(
    "--bin-size",
    type=int,
    default=None,
    help="[pdf] Number of RDF bins.",
)
# ADF options
@click.option(
    "--angle-bins",
    type=int,
    default=None,
    help="[adf] Number of angle bins.",
)
@click.option(
    "--adf-sigma",
    type=float,
    default=None,
    help="[adf] Gaussian kernel width.",
)
@click.option(
    "--adf-cutoff",
    type=float,
    default=None,
    help="[adf] Cutoff radius for triplet search.",
)
# XRD/ND options
@click.option(
    "--element-names",
    "element_names",
    multiple=True,
    default=None,
    help="[xrd/nd] Element names (can be repeated).",
)
@click.option(
    "--qmin",
    type=float,
    default=None,
    help="[xrd/nd] Minimum q value.",
)
@click.option(
    "--qmax",
    type=float,
    default=None,
    help="[xrd/nd] Maximum q value.",
)
@click.option(
    "--qstep",
    type=float,
    default=None,
    help="[xrd/nd] Q step size.",
)
@click.option(
    "--biso",
    type=float,
    default=None,
    help="[xrd/nd] Debye-Waller B factor.",
)
# Tersoff (empirical-potential) guidance
@click.option(
    "--tersoff-guidance/--no-tersoff-guidance",
    default=None,
    help="Add Tersoff-energy gradient as an auxiliary score term during the reverse SDE.",
)
@click.option(
    "--tersoff-lambda",
    type=float,
    default=None,
    help="[tersoff] Weight lambda_0 for the Tersoff guidance schedule.",
)
@click.option(
    "--tersoff-schedule",
    type=click.Choice(["constant", "linear", "sigmoid"]),
    default=None,
    help="[tersoff] Shape of lambda(t).",
)
@click.option(
    "--tersoff-t-gate",
    type=float,
    default=None,
    help="[tersoff] Gate time for the sigmoid schedule.",
)
@click.option(
    "--tersoff-clamp",
    type=float,
    default=None,
    help="[tersoff] Per-atom guidance-norm clamp (Å units of autograd/N).",
)
@click.option(
    "--tersoff-tweedie/--no-tersoff-tweedie",
    default=None,
    help="[tersoff] Evaluate Tersoff on the Tweedie denoised estimate x̂₀ = x_t + σ²·score "
         "rather than on the noisy x_t. Default: True.",
)
# Langevin predictor-corrector
@click.option(
    "--n-corr",
    type=int,
    default=None,
    help="Number of Langevin corrector steps per predictor step. 0 disables.",
)
@click.option(
    "--corr-step-size",
    type=float,
    default=None,
    help="Corrector step size (effective step = corr_step_size * sigma(t)^2).",
)
@click.option(
    "--corr-use-tersoff/--no-corr-use-tersoff",
    default=None,
    help="Include Tersoff gradient inside the corrector loop.",
)
@click.option(
    "--corr-t-gate",
    type=float,
    default=None,
    help="Skip corrector when t > corr_t_gate * t_max.",
)
# Non-linear t schedule
@click.option(
    "--t-schedule-rho",
    type=float,
    default=None,
    help="Power-law exponent for the t trajectory (1.0 = linspace, >1 concentrates near t=0).",
)
# Simulated-annealing post-relaxation
@click.option(
    "--sa-n-steps",
    type=int,
    default=None,
    help="Number of simulated-annealing steps after the reverse SDE. 0 disables.",
)
@click.option(
    "--sa-t0",
    type=float,
    default=None,
    help="[SA] Initial temperature.",
)
@click.option(
    "--sa-t-end",
    type=float,
    default=None,
    help="[SA] Final temperature.",
)
@click.option(
    "--sa-lr",
    type=float,
    default=None,
    help="[SA] Step size on the Tersoff force.",
)
@click.option(
    "--sa-lr-clamp",
    type=float,
    default=None,
    help="[SA] Per-atom per-step displacement cap (Å).",
)
@click.option(
    "--n-restart",
    type=int,
    default=None,
    show_default=True,
    help="Number of full denoising passes per structure. 1 = single pass (default). "
         "Each restart starts from the previous pass output (same cell/species/guidance).",
)
@click.option(
    "--uncond-prepass/--no-uncond-prepass",
    default=None,
    help="Run one unconditional denoising pass on the init structure before "
         "the conditional passes. Only applies when guidance_type is set. "
         "Default: enabled.",
)
@click.option(
    "--lambda-prior",
    type=float,
    default=None,
    help="Scale factor applied to the prior score network output at every SDE step. "
         "1.0 (default) = unmodified prior. 0.0 turns off the prior entirely, "
         "leaving only guidance terms. Values > 1 amplify the prior.",
)
@click.option(
    "--params",
    "params_file",
    type=click.Path(exists=True),
    default=None,
    help="YAML file with generation hyperparameters to override experiment config. "
         "Only keys present in the file are applied; explicit CLI flags take precedence.",
)
def generate(
    experiment_path,
    inits,
    checkpoint,
    outdir,
    device,
    tmin,
    tmax,
    tstep,
    uncond_tmin,
    uncond_tmax,
    uncond_tstep,
    min_dist,
    cutoff,
    n_runs,
    save_traj,
    guidance_type,
    ref_path,
    ref_structure,
    exp_data,
    rho,
    spec_model_path,
    xas_average,
    pdf_cutoff,
    bin_size,
    angle_bins,
    adf_sigma,
    adf_cutoff,
    element_names,
    qmin,
    qmax,
    qstep,
    biso,
    tersoff_guidance,
    tersoff_lambda,
    tersoff_schedule,
    tersoff_t_gate,
    tersoff_clamp,
    tersoff_tweedie,
    n_corr,
    corr_step_size,
    corr_use_tersoff,
    corr_t_gate,
    t_schedule_rho,
    sa_n_steps,
    sa_t0,
    sa_t_end,
    sa_lr,
    sa_lr_clamp,
    n_restart,
    uncond_prepass,
    lambda_prior,
    params_file,
):
    """Generate structures using trained score model."""
    import ase
    from ase.io import read

    # Set CUDA memory config
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Load experiment.  If the experiment has no config.yaml, fall back to
    # the built-in ExperimentConfig defaults (mirrors glass/default_params.yaml)
    # so that `glass generate` works on any experiment dir that just contains
    # checkpoints/ and inits/.
    experiment = Experiment(experiment_path)
    if experiment.config_path.exists():
        config = experiment.load_config()
    else:
        from glass.experiment import ExperimentConfig
        click.echo(
            f"No config.yaml in {experiment.root}; using built-in defaults."
        )
        config = ExperimentConfig()

    # Apply params-file overrides (between experiment config and explicit CLI flags)
    if params_file:
        config.update_from_yaml(params_file)
        click.echo(f"Applied params overrides from {params_file}")

    # Verify this is a score model experiment
    if config.model_type != "score":
        raise click.ClickException(
            f"Experiment '{experiment_path}' is not a score model (type: {config.model_type}).\n"
            "Generation requires a score model experiment."
        )
    
    # Apply CLI overrides. Use `is not None` (not `or`) so an explicit 0 on
    # the command line is honoured instead of silently falling back to config.
    device = device or config.device
    checkpoint = checkpoint or config.checkpoint
    tmin = tmin if tmin is not None else config.tmin
    tmax = tmax if tmax is not None else config.tmax
    tstep = tstep if tstep is not None else config.tstep
    uncond_tmin = uncond_tmin if uncond_tmin is not None else config.uncond_tmin
    uncond_tmax = uncond_tmax if uncond_tmax is not None else config.uncond_tmax
    uncond_tstep = uncond_tstep if uncond_tstep is not None else config.uncond_tstep
    min_dist = min_dist if min_dist is not None else config.min_dist
    cutoff = cutoff if cutoff is not None else config.cutoff
    n_runs = n_runs if n_runs is not None else config.n_runs
    save_traj = save_traj if save_traj is not None else config.save_traj
    rho = rho if rho is not None else config.rho
    if rho is None and guidance_type:
        rho = DEFAULT_RHO[guidance_type]
    bin_size = bin_size if bin_size is not None else config.bin_size
    pdf_cutoff = pdf_cutoff if pdf_cutoff is not None else config.pdf_cutoff
    xas_average = xas_average if xas_average is not None else config.xas_average
    is_xas = guidance_type in ("exafs", "xanes")
    if is_xas and exp_data and not xas_average:
        raise click.ClickException(
            "Experimental EXAFS/XANES targets are atom averages; use --xas-average."
        )
    angle_bins = angle_bins if angle_bins is not None else config.angle_bins
    adf_sigma = adf_sigma if adf_sigma is not None else config.adf_sigma
    adf_cutoff = adf_cutoff if adf_cutoff is not None else config.adf_cutoff
    element_names = list(element_names) if element_names else config.element_names
    qmin = qmin if qmin is not None else config.qmin
    qmax = qmax if qmax is not None else config.qmax
    qstep = qstep if qstep is not None else config.qstep
    biso = biso if biso is not None else config.biso
    # Tersoff guidance
    tersoff_guidance = (
        tersoff_guidance if tersoff_guidance is not None else config.tersoff_guidance
    )
    tersoff_lambda = (
        tersoff_lambda if tersoff_lambda is not None else config.tersoff_lambda
    )
    tersoff_schedule = tersoff_schedule or config.tersoff_schedule
    tersoff_t_gate = (
        tersoff_t_gate if tersoff_t_gate is not None else config.tersoff_t_gate
    )
    tersoff_clamp = (
        tersoff_clamp if tersoff_clamp is not None else config.tersoff_clamp
    )
    tersoff_tweedie = (
        tersoff_tweedie if tersoff_tweedie is not None else getattr(config, "tersoff_tweedie", True)
    )
    # Sampler refinements
    n_corr = n_corr if n_corr is not None else config.n_corr
    corr_step_size = corr_step_size if corr_step_size is not None else config.corr_step_size
    corr_use_tersoff = (
        corr_use_tersoff if corr_use_tersoff is not None else config.corr_use_tersoff
    )
    corr_t_gate = corr_t_gate if corr_t_gate is not None else config.corr_t_gate
    t_schedule_rho = (
        t_schedule_rho if t_schedule_rho is not None else config.t_schedule_rho
    )
    sa_n_steps = sa_n_steps if sa_n_steps is not None else config.sa_n_steps
    sa_t0 = sa_t0 if sa_t0 is not None else config.sa_T0
    sa_t_end = sa_t_end if sa_t_end is not None else config.sa_T_end
    sa_lr = sa_lr if sa_lr is not None else config.sa_lr
    sa_lr_clamp = sa_lr_clamp if sa_lr_clamp is not None else config.sa_lr_clamp
    n_restart = n_restart if n_restart is not None else getattr(config, "n_restart", 1)
    n_restart = max(1, int(n_restart))
    uncond_prepass = (
        uncond_prepass if uncond_prepass is not None
        else getattr(config, "uncond_prepass", True)
    )
    lambda_prior = (
        lambda_prior if lambda_prior is not None
        else getattr(config, "lambda_prior", 1.0)
    )

    # Resolve output directory
    if outdir is None:
        outdir = experiment.outputs_dir
    os.makedirs(outdir, exist_ok=True)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    click.echo(f"Using device: {device}")
    if torch.cuda.is_available():
        click.echo(f"GPU: {torch.cuda.get_device_name(device.index or 0)}")
    
    # Load checkpoint
    ckpt_path = experiment.find_checkpoint(checkpoint)
    click.echo(f"Loading checkpoint: {ckpt_path}")
    
    # Load model
    score_net = LitScoreNet.load_from_checkpoint(ckpt_path, map_location=device)
    score_net.eval()
    score_net.ema_model.to(device)
    score_net.ema_model.eval()
    
    # The noise schedule only depends on k, so no training data is needed here.
    diffuser = VarianceExplodingDiffuser(k=config.k)

    # Setup Tersoff-energy guidance if requested (independent of conditional guidance).
    tersoff_guidance_fn = None
    tersoff_schedule_fn = None
    if tersoff_guidance:
        click.echo(
            f"Tersoff guidance: schedule={tersoff_schedule} "
            f"lambda0={tersoff_lambda} t_gate={tersoff_t_gate} "
            f"clamp={tersoff_clamp}"
        )
        tersoff_guidance_fn = TersoffEnergyGuidance(
            clamp_norm=tersoff_clamp,
        )
        tersoff_schedule_fn = TersoffSchedule(
            schedule=tersoff_schedule,
            lambda_0=tersoff_lambda,
            tmax=(tmax if guidance_type else uncond_tmax),
            t_gate=tersoff_t_gate,
        )

    # Simulated-annealing post-relaxation: SA always runs on the Tersoff PES,
    # so auto-create a guidance instance if the user did not enable Tersoff.
    anneal_fn = None
    sa_guidance_fn = tersoff_guidance_fn
    if sa_n_steps and sa_n_steps > 0:
        if sa_guidance_fn is None:
            sa_guidance_fn = TersoffEnergyGuidance(clamp_norm=tersoff_clamp)
        click.echo(
            f"SA tail: n_steps={sa_n_steps} T0={sa_t0} T_end={sa_t_end} "
            f"lr={sa_lr} lr_clamp={sa_lr_clamp}"
        )
        anneal_fn = make_anneal_fn(
            tersoff_guidance=sa_guidance_fn,
            n_steps=sa_n_steps,
            T0=sa_t0,
            T_end=sa_t_end,
            lr=sa_lr,
            lr_clamp=sa_lr_clamp,
        )

    if n_corr and n_corr > 0:
        click.echo(
            f"Corrector: n_corr={n_corr} step_size={corr_step_size} "
            f"use_tersoff={corr_use_tersoff} t_gate={corr_t_gate}"
        )
    if t_schedule_rho != 1.0:
        click.echo(f"Non-linear t schedule: rho={t_schedule_rho}")

    # Setup guidance model if conditional
    guidance_model = None
    if guidance_type:
        click.echo(f"Guidance type: {guidance_type}")
        
        guidance_model = create_guidance_model(
            guidance_type=guidance_type,
            device=device,
            cutoff=cutoff,
            bin_size=bin_size,
            angle_bins=angle_bins,
            adf_sigma=adf_sigma,
            adf_cutoff=adf_cutoff,
            element_names=element_names,
            qmin=qmin,
            qmax=qmax,
            qstep=qstep,
            biso=biso,
            spec_model_path=spec_model_path,
            pdf_cutoff=pdf_cutoff,
        )
        
        # Validate reference source options
        n_ref_sources = sum(bool(x) for x in [ref_path, ref_structure, exp_data])
        if n_ref_sources == 0:
            raise click.ClickException(
                "Conditional generation requires --ref-path, --ref-structure, or --exp-data"
            )
        if n_ref_sources > 1:
            raise click.ClickException(
                "--ref-path, --ref-structure, and --exp-data are mutually exclusive"
            )

    # Load experimental target if specified
    exp_target_y = None
    if exp_data:
        exp_target_y = load_experimental_data(
            exp_data_path=exp_data,
            guidance_type=guidance_type,
            cutoff=pdf_cutoff,
            bin_size=bin_size,
            angle_bins=angle_bins,
            qmin=qmin,
            qmax=qmax,
            qstep=qstep,
            device=device,
        )
        click.echo(f"Loaded experimental target from {exp_data}")

    # Pre-compute averaged target from all structures in --ref-path
    ref_path_target_y = None
    if ref_path and guidance_type:
        import glob as _glob
        ref_files = sorted(_glob.glob(os.path.join(ref_path, "*.xyz")))
        if not ref_files:
            raise click.ClickException(f"No .xyz files found in --ref-path {ref_path}")
        click.echo(f"Computing averaged target from {len(ref_files)} reference structures in {ref_path}")
        targets = []
        for rf in ref_files:
            ref_atoms = read(rf, "-1")
            y = compute_target_from_reference(ref_atoms, guidance_model, guidance_type, cutoff, device)
            if is_xas and xas_average:
                y = y.mean(dim=0, keepdim=True)
            targets.append(y)
        if len({tuple(y.shape) for y in targets}) > 1:
            raise click.ClickException(
                f"Reference structures in {ref_path} give targets of different "
                "shapes (different numbers of atoms). Use --xas-average for "
                "EXAFS/XANES, or references of the same size."
            )
        ref_path_target_y = torch.stack(targets).mean(dim=0)

    # Pre-compute target from single --ref-structure file
    ref_structure_target_y = None
    if ref_structure and guidance_type:
        if not os.path.exists(ref_structure):
            raise click.ClickException(f"--ref-structure file not found: {ref_structure}")
        click.echo(f"Computing target from reference structure: {ref_structure}")
        ref_atoms = read(ref_structure, "-1")
        ref_structure_target_y = compute_target_from_reference(
            ref_atoms, guidance_model, guidance_type, cutoff, device
        )
        if is_xas and xas_average:
            ref_structure_target_y = ref_structure_target_y.mean(dim=0, keepdim=True)

    # Get initial structures
    init_files = experiment.get_init_files(inits)
    if not init_files:
        raise click.ClickException(f"No .xyz files found in {inits}")
    init_structures = []
    for init_file in init_files:
        frames = read(init_file, ":")
        stem = os.path.basename(init_file).replace(".xyz", "")
        for k, frame in enumerate(frames):
            sub = stem if len(frames) == 1 else f"{stem}_{k:02d}"
            init_structures.append((str(init_file), sub, frame))
    click.echo(
        f"Found {len(init_structures)} initial structures in {len(init_files)} files"
    )

    if guidance_type == "pdf":
        for init_file, sub, frame in init_structures:
            # Minimum-image distances are complete only up to half the
            # smallest cell height.
            half = 0.5 * float(np.min(frame.cell.volume / np.linalg.norm(
                np.cross(frame.cell[[1, 2, 0]], frame.cell[[2, 0, 1]]), axis=1)))
            if pdf_cutoff > half + 0.1:
                click.echo(
                    f"Warning: --pdf-cutoff {pdf_cutoff} Å exceeds half the cell "
                    f"height of {init_file} ({half:.2f} Å); the PDF beyond that "
                    "range misses pairs."
                )
                break

    if tersoff_guidance or (sa_n_steps and sa_n_steps > 0):
        for init_file, sub, frame in init_structures:
            if set(frame.get_chemical_symbols()) != {"Si"}:
                raise click.ClickException(
                    f"Tersoff guidance and the SA tail only support pure Si, but "
                    f"{init_file} contains {sorted(set(frame.get_chemical_symbols()))}. "
                    "Use --no-tersoff-guidance and --sa-n-steps 0."
                )

    # Time steps (power-law; rho=1.0 == linspace). Unconditional runs and the
    # prepass of guided runs use the uncond_* schedule; the guided pass uses
    # tmin/tmax/tstep.
    ts_uncond = power_law_ts(
        uncond_tmin, uncond_tmax, uncond_tstep, rho=t_schedule_rho, device=device
    )
    ts_guided = power_law_ts(tmin, tmax, tstep, rho=t_schedule_rho, device=device)
    ts_torch = ts_guided if guidance_type else ts_uncond
    run_tmax, run_tstep = (tmax, tstep) if guidance_type else (uncond_tmax, uncond_tstep)

    # Process each initial structure
    for init_file, sub_id, init_atoms in init_structures:
        click.echo(f"\nProcessing: {sub_id}")

        # Select pre-computed target for this structure
        target_y = None
        if ref_path_target_y is not None:
            target_y = ref_path_target_y
        elif ref_structure_target_y is not None:
            target_y = ref_structure_target_y
        elif exp_data:
            target_y = exp_target_y
        
        if is_xas and not xas_average and target_y.shape[0] != len(init_atoms):
            raise click.ClickException(
                f"Atom-wise {guidance_type} guidance needs a reference with as many "
                f"atoms as {sub_id} ({target_y.shape[0]} vs {len(init_atoms)}). "
                "Use --xas-average to compare atom-averaged spectra."
            )
        if is_xas and xas_average and exp_data:
            n_out = guidance_model.module.decoder.out_dim
            if target_y.shape[1] != n_out:
                raise click.ClickException(
                    f"--exp-data has {target_y.shape[1]} points, but the surrogate "
                    f"predicts {n_out}."
                )

        # Setup likelihood function if conditional
        likelihood_fn = None
        if guidance_type:
            likelihood_fn = LikelihoodScore(
                score_net.ema_model,
                guidance_model,
                target_y,
                rho,
                diffuser,
                guidance_type,
                cutoff,
                spec_average=bool(is_xas and xas_average),
            )
        
        # Run generation
        tag_parts = []
        if guidance_type:
            rho_str = f"{int(rho)}" if rho == int(rho) else f"{rho:.2f}"
            tag_parts.append(f"{guidance_type}_rho{rho_str}")
        if tersoff_guidance:
            tag_parts.append(
                f"tersoff_{tersoff_schedule}_lam{tersoff_lambda:g}"
                f"_gate{tersoff_t_gate:g}"
            )
        tag_parts.append(f"tmax{run_tmax}_nsteps{run_tstep}")
        run_outdir = os.path.join(outdir, sub_id, "_".join(tag_parts)) \
            if tag_parts else os.path.join(outdir, sub_id)

        extra_tag = []
        if n_corr and n_corr > 0:
            extra_tag.append(f"corr{n_corr}s{corr_step_size:g}")
        if t_schedule_rho != 1.0:
            extra_tag.append(f"rho{t_schedule_rho:g}")
        if sa_n_steps and sa_n_steps > 0:
            extra_tag.append(f"sa{sa_n_steps}lr{sa_lr:g}")
        if min_dist:
            extra_tag.append(f"mind{min_dist:g}")
        if extra_tag:
            run_outdir = os.path.join(run_outdir, "_".join(extra_tag))
        os.makedirs(run_outdir, exist_ok=True)

        # Persist the full generation-hparam vector next to the outputs so
        # downstream analysis can attribute each xyz to its exact config.
        # The path-tag encoding is lossy (omits flags at default, rounds
        # floats); params.json is the authoritative record.
        import json as _json
        params_record = {
            "mode": "cond" if guidance_type else "uncond",
            "guidance_type": guidance_type,
            "rho": rho if guidance_type else None,
            "pdf_cutoff": pdf_cutoff if guidance_type == "pdf" else None,
            "bin_size": bin_size if guidance_type == "pdf" else None,
            "spec_model_path": spec_model_path if is_xas else None,
            "xas_average": bool(xas_average) if is_xas else None,
            "ref_path": ref_path,
            "ref_structure": ref_structure,
            "exp_data": exp_data,
            "tmin": tmin if guidance_type else None,
            "tmax": tmax if guidance_type else None,
            "tstep": tstep if guidance_type else None,
            "uncond_tmin": uncond_tmin,
            "uncond_tmax": uncond_tmax,
            "uncond_tstep": uncond_tstep,
            "min_dist": min_dist,
            "t_schedule_rho": t_schedule_rho,
            "cutoff": cutoff,
            "tersoff_guidance": bool(tersoff_guidance),
            "tersoff_lambda": tersoff_lambda if tersoff_guidance else None,
            "tersoff_schedule": tersoff_schedule if tersoff_guidance else None,
            "tersoff_t_gate": tersoff_t_gate if tersoff_guidance else None,
            "tersoff_clamp": tersoff_clamp if tersoff_guidance else None,
            "tersoff_tweedie": tersoff_tweedie if tersoff_guidance else None,
            "n_corr": n_corr,
            "corr_step_size": corr_step_size if n_corr else None,
            "corr_use_tersoff": corr_use_tersoff if n_corr else None,
            "corr_t_gate": corr_t_gate if n_corr else None,
            "sa_n_steps": sa_n_steps,
            "sa_T0": sa_t0 if sa_n_steps else None,
            "sa_T_end": sa_t_end if sa_n_steps else None,
            "sa_lr": sa_lr if sa_n_steps else None,
            "sa_lr_clamp": sa_lr_clamp if sa_n_steps else None,
            "checkpoint": str(ckpt_path),
            "n_runs": n_runs,
            "n_restart": n_restart,
            "uncond_prepass": bool(uncond_prepass) if guidance_type else None,
            "lambda_prior": lambda_prior,
            "init_file": init_file,
            "sub_id": sub_id,
        }
        with open(os.path.join(run_outdir, "params.json"), "w") as _pf:
            _json.dump(params_record, _pf, indent=2, default=str)

        # Prior function wrapper
        _lp = float(lambda_prior)
        def prior_fn(sp, p, c, t, co, _sn=score_net, _df=diffuser, _lp=_lp):
            return _lp * compute_prior_score(sp, p, c, t, co, _sn, _df)
        
        for i in range(n_runs):
            import copy
            species, pos, cell = atoms_to_device(copy.deepcopy(init_atoms), device)
            cell_np = cell.detach().cpu().numpy()

            pos_current = pos
            # Frames from every pass (prepass, restarts, SA) when save_traj.
            traj = [pos_current.detach().cpu().clone()] if save_traj else None

            # Optional unconditional prepass: a single full denoising pass with
            # no likelihood guidance to give the conditional restarts a more
            # physical starting point than the raw init.
            if guidance_type and uncond_prepass:
                pre_pbar = tqdm(
                    total=len(ts_uncond),
                    desc=f"  run {i+1}/{n_runs} uncond-prepass ",
                    unit="step",
                    leave=False,
                    dynamic_ncols=True,
                )

                def pre_progress(step, t, p_norm, _pbar=pre_pbar, **_kw):
                    _pbar.update(1)
                    _pbar.set_postfix(p=f"{p_norm:.3f}")

                pre_traj, pos_current = denoise_by_sde(
                    species=species,
                    pos=pos_current,
                    cell=cell,
                    cutoff=cutoff,
                    score_fn=prior_fn,
                    likelihood_fn=None,
                    ts=ts_uncond,
                    diffuser=diffuser,
                    save_traj=save_traj,
                    progress_fn=pre_progress,
                    tersoff_guidance=tersoff_guidance_fn,
                    tersoff_schedule=tersoff_schedule_fn,
                    tersoff_tweedie=tersoff_tweedie,
                    n_corr=n_corr,
                    corr_step_size=corr_step_size,
                    corr_use_tersoff=corr_use_tersoff,
                    corr_t_gate=corr_t_gate,
                    anneal_fn=None,
                    min_dist=min_dist,
                )
                pre_pbar.close()
                if save_traj:
                    traj.extend(pre_traj[1:])

            for _restart in range(n_restart):
                last_restart = (_restart == n_restart - 1)
                total_steps = len(ts_torch)
                restart_label = f"restart {_restart+1}/{n_restart} " if n_restart > 1 else ""
                pbar = tqdm(
                    total=total_steps,
                    desc=f"  run {i+1}/{n_runs} {restart_label}",
                    unit="step",
                    leave=False,
                    dynamic_ncols=True,
                )

                def progress_callback(step, t, p_norm, l_norm=None, target_norm=None, _pbar=pbar, **_kw):
                    _pbar.update(1)
                    if l_norm is not None:
                        _pbar.set_postfix(p=f"{p_norm:.3f}", l=f"{l_norm:.3f}", tgt=f"{target_norm:.4f}")
                    else:
                        _pbar.set_postfix(p=f"{p_norm:.3f}")

                pass_traj, pos_current = denoise_by_sde(
                    species=species,
                    pos=pos_current,
                    cell=cell,
                    cutoff=cutoff,
                    score_fn=prior_fn,
                    likelihood_fn=likelihood_fn,
                    ts=ts_torch,
                    diffuser=diffuser,
                    save_traj=save_traj,
                    progress_fn=progress_callback,
                    tersoff_guidance=tersoff_guidance_fn,
                    tersoff_schedule=tersoff_schedule_fn,
                    tersoff_tweedie=tersoff_tweedie,
                    n_corr=n_corr,
                    corr_step_size=corr_step_size,
                    corr_use_tersoff=corr_use_tersoff,
                    corr_t_gate=corr_t_gate,
                    anneal_fn=anneal_fn if last_restart else None,
                    min_dist=min_dist,
                )
                pbar.close()
                if save_traj:
                    traj.extend(pass_traj[1:])
            final_pos = pos_current
            
            if save_traj:
                traj_list = []
                for p in traj:
                    a = ase.Atoms(
                        numbers=init_atoms.numbers,
                        positions=p.numpy(),
                        cell=cell_np,
                        pbc=[True] * 3,
                    )
                    a.wrap()
                    traj_list.append(a)
                ase.io.write(f"{run_outdir}/{i:02}_traj.xyz", traj_list)
                ase.io.write(f"{run_outdir}/{i:02}_final.xyz", traj_list[-1])
            else:
                final = ase.Atoms(
                    numbers=init_atoms.numbers,
                    positions=final_pos.cpu().numpy(),
                    cell=cell_np,
                    pbc=[True] * 3,
                )
                final.wrap()
                ase.io.write(f"{run_outdir}/{i:02}_final.xyz", final)
            
            click.echo(f"  Run {i+1}/{n_runs} complete")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    click.echo(f"\nGeneration complete! Results saved to: {outdir}")
