# Fix for macOS multiprocessing issue with PyTorch Lightning
# Python 3.8+ on macOS uses 'spawn' by default which doesn't work well with
# complex dataset objects. We use 'fork' which is more compatible.
import platform
import multiprocessing
if platform.system() == "Darwin":  # macOS
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass  # Already set, ignore

import click


@click.group()
def glass():
    """CLI for training and generating atomic structures with glass."""
    pass


from glass.cli.train import train
from glass.cli.generate import generate
from glass.cli.initialize import initialize
from glass.cli.metrics import metrics, compute_pdf_command, compute_coordination_command, compute_rings_command, compare_command

glass.add_command(train)
glass.add_command(generate)
glass.add_command(initialize)
glass.add_command(metrics)
glass.add_command(compute_pdf_command, name="pdf")
glass.add_command(compute_coordination_command, name="coordination")
glass.add_command(compute_rings_command, name="rings")
glass.add_command(compare_command, name="compare")

# Register Tersoff potential subcommands (md, energy) under the `glass` group.
from glass.potentials.tersoff.cli import md as _tersoff_md
from glass.potentials.tersoff.cli import energy as _tersoff_energy

glass.add_command(_tersoff_md, name="md")
glass.add_command(_tersoff_energy, name="energy")

# Register CRN generation command
from glass.cli.crn import crn

glass.add_command(crn)


if __name__ == "__main__":
    glass()