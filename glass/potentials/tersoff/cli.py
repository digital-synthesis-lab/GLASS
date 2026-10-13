"""Command-line interface for running MD with the PyTorch Tersoff calculator."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import click
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from .ase_calc import silicon_calculator


@click.group()
def main():
    """torch-tersoff: PyTorch Tersoff potential tools."""


@main.command("md")
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Input structure (e.g., extxyz). Read via ase.io.read.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default="md.traj",
    show_default=True,
    help="Output ASE trajectory file.",
)
@click.option(
    "--ensemble",
    type=click.Choice(["nve", "nvt"], case_sensitive=False),
    default="nve",
    show_default=True,
    help="MD ensemble: NVE (Velocity Verlet) or NVT (Langevin).",
)
@click.option(
    "--timestep", type=float, default=1.0, show_default=True, help="Timestep in fs."
)
@click.option(
    "--steps", type=int, default=100, show_default=True, help="Number of MD steps."
)
@click.option(
    "--temperature",
    type=float,
    default=300.0,
    show_default=True,
    help="Target / initial temperature in Kelvin.",
)
@click.option(
    "--friction",
    type=float,
    default=0.01,
    show_default=True,
    help="Langevin friction (1/fs). Used only for --ensemble nvt.",
)
@click.option(
    "--log-interval",
    type=int,
    default=10,
    show_default=True,
    help="Print/save interval in steps.",
)
@click.option(
    "--seed", type=int, default=42, show_default=True, help="RNG seed for velocities."
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    show_default=True,
    help="torch device (e.g. cpu, cuda).",
)
@click.option(
    "--init-velocities/--no-init-velocities",
    default=True,
    show_default=True,
    help="Initialize Maxwell-Boltzmann velocities at --temperature.",
)
def md(
    input_path: Path,
    output_path: Path,
    ensemble: str,
    timestep: float,
    steps: int,
    temperature: float,
    friction: float,
    log_interval: int,
    seed: int,
    device: str,
    init_velocities: bool,
):
    """Run MD on INPUT_PATH using the PyTorch Tersoff Si potential."""
    atoms = read(input_path)
    atoms.calc = silicon_calculator(device=device)

    if init_velocities:
        import numpy as np

        np.random.seed(seed)
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

    dt = timestep * units.fs
    ensemble = ensemble.lower()
    if ensemble == "nve":
        dyn = VelocityVerlet(atoms, timestep=dt)
    else:
        dyn = Langevin(
            atoms,
            timestep=dt,
            temperature_K=temperature,
            friction=friction / units.fs,
            rng=__import__("numpy").random.default_rng(seed),
        )

    traj = Trajectory(str(output_path), "w", atoms)
    dyn.attach(traj.write, interval=log_interval)

    def _log():
        ekin = atoms.get_kinetic_energy()
        epot = atoms.get_potential_energy()
        T = ekin / (1.5 * units.kB * len(atoms))
        click.echo(
            f"step={dyn.nsteps:6d}  E_pot={epot:12.4f} eV  "
            f"E_kin={ekin:10.4f} eV  T={T:8.2f} K  "
            f"E_tot={epot + ekin:12.4f} eV"
        )

    dyn.attach(_log, interval=log_interval)

    _log()
    dyn.run(steps)
    traj.close()
    click.echo(f"Wrote trajectory to {output_path}")


@main.command("energy")
@click.argument(
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--device", default="cpu", show_default=True)
def energy(input_path: Path, device: str):
    """Print single-point energy and max |force|."""
    atoms = read(input_path)
    atoms.calc = silicon_calculator(device=device)
    E = atoms.get_potential_energy()
    F = atoms.get_forces()
    E = E / len(atoms)
    click.echo(f"E = {E:.8f} eV/atom")
    click.echo(f"max |F| = {abs(F).max():.6e} eV/A")


if __name__ == "__main__":
    main()
