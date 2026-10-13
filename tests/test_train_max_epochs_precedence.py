"""Regression: a spec model's hard-coded max_epochs default (8000 exafs /
3000 xanes) must not clobber a value already supplied by config.yaml.

Guards against the pre-fix state where the default fired any time
``--max-epochs`` wasn't passed on the CLI, silently overwriting a stored
config.yaml value. Precedence must be: existing config.yaml > --params YAML >
explicit CLI flags > hard-coded default.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import yaml
from click.testing import CliRunner

from glass.cli.train import train
from glass.experiment import Experiment, ExperimentConfig


def test_explicit_config_max_epochs_survives(tmp_path):
    exp = Experiment(str(tmp_path / "exp"))
    exp._create_structure()
    cfg = ExperimentConfig()
    cfg.model_type = "spec"
    cfg.spec_type = "exafs"
    cfg.num_species = 1
    cfg.max_epochs = 500
    exp.save_config(cfg)

    CliRunner().invoke(train, [str(exp.root), "--resume"])

    saved = ExperimentConfig.from_yaml(exp.config_path)
    assert saved.max_epochs == 500, (
        "hard-coded spec default overwrote the max_epochs stored in config.yaml"
    )


def test_default_applies_when_config_omits_max_epochs(tmp_path):
    exp = Experiment(str(tmp_path / "exp"))
    exp._create_structure()
    # Hand-write a partial config.yaml, as e.g. a template would, deliberately
    # omitting max_epochs so the dataclass default never appears here.
    with open(exp.config_path, "w") as fh:
        yaml.dump({"model_type": "spec", "spec_type": "exafs", "num_species": 1}, fh)

    CliRunner().invoke(train, [str(exp.root), "--resume"])

    saved = ExperimentConfig.from_yaml(exp.config_path)
    assert saved.max_epochs == 8000, (
        "spec default did not apply when config.yaml had no max_epochs key"
    )


def test_params_file_does_not_override_model_identity(tmp_path):
    """A --params file (e.g. a copy of default_params.yaml) must not reset
    --model-type / --num-species / --spec-type given on the command line."""
    params = tmp_path / "params.yaml"
    with open(params, "w") as fh:
        yaml.dump(
            {"model_type": "score", "num_species": 1, "spec_type": None,
             "out_dim": None, "max_epochs": 7},
            fh,
        )
    exp_dir = tmp_path / "exp"
    # The run itself fails (no training data); only the saved config matters.
    CliRunner().invoke(
        train,
        [str(exp_dir), "--model-type", "spec", "--spec-type", "xanes",
         "--num-species", "2", "--params", str(params)],
    )

    saved = ExperimentConfig.from_yaml(exp_dir / "config.yaml")
    assert saved.model_type == "spec"
    assert saved.spec_type == "xanes"
    assert saved.num_species == 2
    assert saved.out_dim == 100
    assert saved.max_epochs == 7
