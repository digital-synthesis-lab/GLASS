"""Regression: `glass train --resume` must load the experiment's saved config.

Guards against the pre-fix state where the config was loaded only when
``not resume``, so ``--resume`` fell into the new-experiment branch: it demanded
--model-type/--num-species and rebuilt a DEFAULT ExperimentConfig, yet still
handed last.ckpt to trainer.fit — i.e. it resumed a trained checkpoint with the
wrong architecture (default dim/num_convs instead of the trained ones).
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from click.testing import CliRunner

from glass.cli.train import train
from glass.experiment import Experiment, ExperimentConfig


def _experiment_with_config(root):
    """A saved experiment: config.yaml on disk, data/ deliberately left empty."""
    exp = Experiment(str(root))
    exp._create_structure()
    cfg = ExperimentConfig()
    cfg.model_type = "score"
    cfg.num_species = 3
    exp.save_config(cfg)
    return exp


def test_resume_loads_saved_config(tmp_path):
    # data/ is empty, so the run stops at the "no .xyz" guard — right after the
    # config has been resolved, which is all this test is about.
    exp = _experiment_with_config(tmp_path / "exp")

    result = CliRunner().invoke(train, [str(exp.root), "--resume"])

    assert "--model-type is required" not in result.output, (
        "--resume took the new-experiment branch and discarded config.yaml"
    )
    assert "Loading existing config" in result.output
    # The saved config drove the run even though no flags were passed.
    assert "Model type: score" in result.output
    assert "No .xyz files found" in result.output


def test_new_still_loads_saved_config(tmp_path):
    # Unchanged behaviour: --new on an existing experiment reuses its config.
    exp = _experiment_with_config(tmp_path / "exp")

    result = CliRunner().invoke(train, [str(exp.root), "--new"])

    assert "Loading existing config" in result.output
