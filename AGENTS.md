# AGENTS.md

Notes for coding agents working on this repository.
User-facing documentation lives in `README.md`.

## What this repo is

`glass` is the code for "Generative Inversion of Spectroscopic Data for
Amorphous Structure Elucidation" (arXiv:2603.23210). A score-based diffusion
model learns a structural prior from atomistic structures; sampling can be
guided by differentiable spectra (PDF, ADF, XRD, ND, EXAFS, XANES).

## Setup and tests

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e ".[test,plot,hpo]"
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/pytest -q          # ~2 min, needs no GPU
```

`tests/test_inference.py` and `tests/test_generation.py` load the example
model in `examples/silicon/experiment/checkpoints/best.ckpt` and skip when it
is missing.

## Layout

```
glass/
├── cli/                  # click commands; entry point glass.cli:glass
│   ├── train.py          # glass train
│   ├── generate.py       # glass generate (unconditional + guided)
│   ├── initialize.py     # glass initialize (random packing)
│   ├── metrics.py        # glass metrics / compare / pdf / coordination / rings
│   └── crn.py            # glass crn (WWW continuous random network, Si)
├── diffusion/
│   ├── __init__.py       # VarianceExplodingDiffuser, sigma(t) = k t
│   ├── sampling.py       # denoise_by_sde: predictor, corrector, min-dist veto
│   ├── schedules.py      # power_law_ts
│   └── annealing.py      # optional SA tail on the Tersoff PES
├── experiment.py         # ExperimentConfig (all defaults) + Experiment dirs
├── default_params.yaml   # mirror of ExperimentConfig (tested)
├── lit/
│   ├── datamodules/structure_spec.py   # noisy graphs for training
│   └── modules/
│       ├── prior.py      # LitScoreNet (score model)
│       ├── forward.py    # LitSpecNet (EXAFS/XANES surrogate)
│       ├── likelihood.py # guidance term on the Tweedie estimate x0
│       ├── guidance.py   # builds the differentiable observables
│       ├── differentiable_{rdf,adf,xrd,nd}.py
│       └── tersoff_guidance.py
├── nn/                   # MeshGraphNets blocks, periodic radius graph
├── metrics/              # non-differentiable analysis (PDF, ADF, CN, rings, Tersoff)
├── potentials/tersoff/   # PyTorch Tersoff Si (energy, md CLI)
├── potentials/keating/   # Keating potential used by the CRN builder
├── algorithms/crn/       # WWW bond-switching CRN builder (numba)
└── utils/                # ASE <-> tensor helpers, packing
examples/silicon/         # data, trained model, scripts, figures
scripts/hpo.py            # Optuna search over the optional sampler extensions
```

## Defaults and where they live

`glass/experiment.py::ExperimentConfig` is the single source of defaults;
`glass/default_params.yaml` must mirror it (`tests/test_experiment.py`).
Precedence: CLI flag > `--params` file > experiment `config.yaml` > defaults.
Note that `glass train` writes every field, including generation settings,
into the experiment `config.yaml`, which then overrides the built-in defaults.

Generation follows the preprint protocol:

- unconditional pass `t = 1.0 -> 0.001`, 512 steps (`uncond_*`)
- guided pass `t = 0.6 -> 0.001`, 512 steps (`tmin/tmax/tstep`), started
  from the unconditional output without re-noising
- guidance weight `rho` (w in the paper): `None` resolves to
  `experiment.DEFAULT_RHO` (pdf 300, adf 1000, xrd/nd 1); PDF range `pdf_cutoff = 8 Å`
- Tersoff guidance, Langevin corrector, restarts and SA are off. Tuned
  values for them are in `examples/silicon/hpo_params.yaml`.

Training: `batch_size = 16` structures per GPU. A batch of a single 216-atom
cell leaves the model at the zero-output solution (loss stuck at 3.0); aim
for roughly a thousand atoms or more per batch.

## Conventions

- Guidance score: `-(rho / sum(norm)) * grad_x0 ||target - F(x0)||^2`, with
  `x0 = x + sigma^2 * score` (see `likelihood.py`).
- XRD/ND are open-boundary Debye sums over the atoms in the cell; positions
  are wrapped into the cell first.
- The PDF uses minimum-image distances, valid up to half the cell height.
- Tersoff guidance, the SA tail and Tersoff metrics are pure-Si only.
- EXAFS/XANES: the surrogate (LitSpecNet) is trained on per-atom spectra in
  `data/<exafs|xanes>/train/<stem>.txt` next to `data/structures/<stem>.xyz`.
  Guidance compares atom i with atom i of the reference by default (as in the
  preprint code); `--xas-average` compares atom-averaged spectra and enables
  `--exp-data`. Pretrained Si surrogates: `examples/silicon/xas/`.

## Adding a guidance type

1. Add `DifferentiableXXX` in `glass/lit/modules/` with
   `forward(pos, species, cell) -> feature`.
2. Register it in `guidance.py::create_guidance_model`.
3. Add a branch in `likelihood.py::LikelihoodScore.forward` and in
   `utils/atoms.py::compute_target_from_reference`.
4. Add the name to the `--guidance-type` choice in `cli/generate.py`.
5. Add a test next to `tests/test_differentiable_xrd_nd.py`.
