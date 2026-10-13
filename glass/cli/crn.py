"""`glass crn` CLI command for CRN generation."""

from __future__ import annotations

import os
from pathlib import Path

import click

# Suppress OpenMP duplicate library warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


@click.command("crn")
@click.option(
    "--n-atoms",
    type=int,
    default=216,
    show_default=True,
    help="Number of Si atoms.",
)
@click.option(
    "--density",
    type=float,
    default=2.33,
    show_default=True,
    help="Target density in g/cm³.",
)
@click.option("--seed", type=int, default=0, show_default=True, help="Random seed.")
@click.option(
    "--n-cycles",
    type=int,
    default=5,
    show_default=True,
    help="Number of anneal/quench cycles.",
)
@click.option(
    "--n-anneal-per-atom",
    type=int,
    default=50,
    show_default=True,
    help="Trial transpositions per atom per anneal phase.",
)
@click.option(
    "--kT",
    "kT",
    type=float,
    default=0.25,
    show_default=True,
    help="Annealing temperature in eV.",
)
@click.option(
    "--quench-attempts-per-atom",
    type=int,
    default=18,
    show_default=True,
    help="T=0 quench attempts per atom.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False),
    required=True,
    help="Output XYZ file.",
)
@click.option(
    "--bonds-output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Optional path to write bond list (i j per line, 0-indexed).",
)
@click.option("--quiet", is_flag=True, help="Suppress per-cycle log messages.")
def crn(
    n_atoms: int,
    density: float,
    seed: int,
    n_cycles: int,
    n_anneal_per_atom: int,
    kT: float,
    quench_attempts_per_atom: int,
    output_path: str,
    bonds_output: str | None,
    quiet: bool,
):
    """Generate CRN structures using WWW algorithm.

    The Wooten-Winer-Weaire (WWW) algorithm, as improved by Barkema-Mousseau (2000),
    generates 4-coordinated continuous random networks of silicon atoms using bond
    transposition moves combined with simulated annealing.

    The algorithm produces amorphous structures with zero crystalline memory and low
    structural energy under the Keating potential.

    Example:

        glass crn --n-atoms 216 --n-cycles 15 --output crn216.xyz

    For high-quality structures, use n-cycles=15 and n-anneal-per-atom=60.
    """
    import sys
    from ase import Atoms
    from ase.io import write as ase_write

    from glass.algorithms.crn import generate_crn

    # Use click.echo with err=True to write to stderr (unbuffered by default)
    # This ensures progress messages appear immediately even when stdout is redirected
    def log_fn(msg):
        click.echo(msg, err=True)
        sys.stderr.flush()  # Force immediate flush

    log = (lambda _msg: None) if quiet else log_fn

    net, stats = generate_crn(
        n_atoms=n_atoms,
        density=density,
        seed=seed,
        n_cycles=n_cycles,
        n_anneal_per_atom=n_anneal_per_atom,
        kT=kT,
        quench_attempts_per_atom=quench_attempts_per_atom,
        log=log,
    )

    atoms = Atoms(
        symbols=["Si"] * net.n_atoms,
        positions=net.positions,
        cell=net.cell,
        pbc=True,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ase_write(output_path, atoms)
    click.echo(f"wrote {output_path} ({net.n_atoms} atoms)")

    if bonds_output:
        with open(bonds_output, "w") as fh:
            for i, j in net.bonds:
                fh.write(f"{int(i)} {int(j)}\n")
        click.echo(f"wrote {bonds_output} ({len(net.bonds)} bonds)")

    click.echo(
        f"E_final = {stats.final_energy:.3f} eV "
        f"({stats.final_energy / n_atoms:.4f} eV/atom); "
        f"acceptance = {100.0 * stats.n_accepted / max(stats.n_proposed, 1):.2f}%"
    )
