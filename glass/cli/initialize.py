"""Utility to initialize random atomic structures for denoising.

This module provides a command to generate initial configurations with
specified density, composition, and minimum distance constraints. It uses
a cell-list Poisson-disk sampler with a Metropolis Monte-Carlo anneal
fallback (``glass.utils.packing.pack``), so realistic densities are
feasible in under a second per structure for modest system sizes.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from typing import List, Optional, Tuple

import click
import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers, chemical_symbols
from ase.io import write

from glass.utils.packing import pack


# ---------------------------------------------------------------------------
# Mass / volume arithmetic
# ---------------------------------------------------------------------------


# 1 g/cm^3 == 0.602214 amu/Å^3
_DENSITY_AMU_PER_GCC = 0.602214


def _total_mass_amu(species_list: List[str], counts_list: List[int]) -> float:
    return float(
        sum(
            counts_list[i] * atomic_masses[atomic_numbers[species_list[i]]]
            for i in range(len(species_list))
        )
    )


def _calculate_cell_volume(
    n_atoms: int,
    density: float,
    species_list: List[str],
    counts_list: List[int],
) -> float:
    """Cell volume (Å^3) from mass and density (g/cm^3)."""
    total_mass = _total_mass_amu(species_list, counts_list)
    return total_mass / (density * _DENSITY_AMU_PER_GCC)


def _density_from_cell_and_counts(
    cell: np.ndarray, species_list: List[str], counts_list: List[int]
) -> float:
    """Back-compute g/cm^3 implied by an explicit cell + composition."""
    volume = float(abs(np.linalg.det(cell)))
    total_mass = _total_mass_amu(species_list, counts_list)
    return total_mass / (volume * _DENSITY_AMU_PER_GCC)


# ---------------------------------------------------------------------------
# Argument resolution
# ---------------------------------------------------------------------------


def _build_cell_from_scalar_or_triplet(
    cell_a: Optional[float], cell_abc: Optional[Tuple[float, float, float]]
) -> np.ndarray:
    if cell_abc is not None:
        a, b, c = cell_abc
        return np.diag([float(a), float(b), float(c)])
    return np.eye(3) * float(cell_a)


def _resolve_cell_and_counts(
    density: Optional[float],
    cell_a: Optional[float],
    cell_abc: Optional[Tuple[float, float, float]],
    species_list: List[str],
    counts_list: Optional[List[int]],
    box_shape: str,
) -> Tuple[np.ndarray, List[int], float]:
    """Decide the final (cell, counts, density) from user inputs.

    Rules (per plan):
    - ``density + counts`` → compute cell (current behaviour).
    - ``cell + counts`` (no density) → compute density implicitly.
    - ``cell + density`` (no counts) → compute counts from
      volume * density / atomic_mass (single-species only).
    - ``cell + density + counts`` → validate consistency (0.1 % slack).
    - No cell & no density → error.
    """
    cell_given = cell_a is not None or cell_abc is not None
    if cell_given and cell_a is not None and cell_abc is not None:
        raise click.ClickException(
            "--cell-a and --cell-abc are mutually exclusive."
        )

    if not cell_given and density is None:
        raise click.ClickException(
            "Provide at least one of --density or a cell flag "
            "(--cell-a / --cell-abc)."
        )

    # --- Branch 1: no cell given → need density + counts → compute cell.
    if not cell_given:
        if counts_list is None:
            raise click.ClickException(
                "--num-atoms is required when no cell is specified."
            )
        if density is None:
            raise click.ClickException(
                "--density is required when no cell is specified."
            )
        volume = _calculate_cell_volume(
            sum(counts_list), density, species_list, counts_list
        )
        a = volume ** (1.0 / 3.0)
        cell = np.eye(3) * a  # box_shape is currently cosmetic; both -> cube.
        return cell, counts_list, density

    # --- Cell IS given from here on.
    cell = _build_cell_from_scalar_or_triplet(cell_a, cell_abc)
    volume = float(abs(np.linalg.det(cell)))

    # Branch 2: cell + counts (maybe + density): validate or compute density.
    if counts_list is not None:
        implied = _density_from_cell_and_counts(cell, species_list, counts_list)
        if density is not None:
            rel = abs(density - implied) / max(implied, 1e-12)
            if rel > 1e-3:
                raise click.ClickException(
                    "Inconsistent --density / --cell / --num-atoms: "
                    f"requested density={density:.4f} g/cm^3 but the given "
                    f"cell + atom counts imply {implied:.4f} g/cm^3 "
                    f"(relative mismatch {rel:.2%}, tolerance 0.1%)."
                )
        return cell, counts_list, implied

    # Branch 3: cell + density, no counts → derive counts.
    if density is None:
        raise click.ClickException(
            "--num-atoms or --density is required when --cell-* is given."
        )
    if len(species_list) != 1:
        raise click.ClickException(
            "Deriving --num-atoms from --cell + --density is only supported "
            "for a single species. Pass --num-atoms explicitly for "
            "multi-species inputs."
        )
    mass_per_atom = atomic_masses[atomic_numbers[species_list[0]]]
    n_float = (volume * density * _DENSITY_AMU_PER_GCC) / mass_per_atom
    n_rounded = int(round(n_float))
    if n_rounded <= 0:
        raise click.ClickException(
            f"Implied --num-atoms is {n_rounded} from cell + density; "
            "these values are inconsistent with any physical structure."
        )
    # Validate the rounding didn't introduce > 0.1% density error.
    implied = _density_from_cell_and_counts(cell, species_list, [n_rounded])
    rel = abs(density - implied) / max(density, 1e-12)
    if rel > 1e-3:
        raise click.ClickException(
            f"Rounding implied num-atoms {n_float:.3f} -> {n_rounded} "
            f"changes the density from {density:.4f} to {implied:.4f} "
            f"g/cm^3 (> 0.1%). Specify --num-atoms explicitly."
        )
    return cell, [n_rounded], implied


# ---------------------------------------------------------------------------
# Per-structure driver
# ---------------------------------------------------------------------------


def _generate_structure(
    species_list: List[str],
    counts_list: List[int],
    cell: np.ndarray,
    min_distance: float,
    rng: np.random.Generator,
    echo=None,
    verbose: bool = False,
) -> Atoms:
    n_atoms = sum(counts_list)
    positions = pack(
        n_atoms,
        cell,
        min_distance,
        rng=rng,
        verbose=verbose,
        echo=echo,
    )

    symbols = []
    for sp, count in zip(species_list, counts_list):
        symbols.extend([sp] * count)

    # Shuffle positions so species are not spatially correlated with the
    # order in which the packer placed them.
    rng.shuffle(positions)

    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command(
    "initialize",
    help="""
Initialize random atomic structures for denoising.

Generates one or more random structures satisfying a minimum interatomic
distance, and writes them to an XYZ file. The sampler is a batched
Poisson-disk algorithm (cell-list accelerated) with a Metropolis Monte-
Carlo soft-core anneal fallback, so physical densities (e.g. 2.5 g/cm^3
for amorphous Si at min-distance 2 Å) are reachable in under a second.

\b
CELL / DENSITY / COUNT
  Provide any two of --density, --num-atoms, and a cell flag (--cell-a
  or --cell-abc). Passing all three is valid and will be checked for
  consistency (0.1 % tolerance); inconsistent combinations raise an
  error.

\b
NOTES
  - The feasible regime covers up to ~55% packing fraction (MC fallback).
    Above that, the algorithm may emit a warning and return a structure
    whose minimum distance is slightly below the requested value.

EXAMPLES

\b
  # Single-species (Si) at physical density
  glass initialize --output init_Si.xyz --density 2.5 \\
      --species Si --num-atoms 216 --min-distance 2.0

\b
  # Match an exact reference cell (e.g. 15.91 Å cube, Si @ 2.5 g/cm^3)
  glass initialize --output init_Si.xyz --cell-a 15.91 \\
      --species Si --num-atoms 216 --min-distance 2.0

\b
  # Orthorhombic cell
  glass initialize --output init_Si.xyz --cell-abc 15.0 16.0 17.0 \\
      --species Si --num-atoms 200 --min-distance 2.0

\b
  # Multi-species (Si-Ge alloy)
  glass initialize --output init_SiGe.xyz --density 2.0 \\
      -s Si -s Ge -n 108 -n 108 --min-distance 2.0

\b
  # Many structures, seeded for reproducibility
  glass initialize --output init_Si.xyz --density 2.5 --species Si \\
      --num-atoms 216 --min-distance 2.0 --num-structures 10 --seed 42
""",
)
@click.option("--output", "-o", type=str, required=True,
              help="Output XYZ file path.")
@click.option("--density", "-d", type=float, default=None,
              help="Density in g/cm^3.")
@click.option("--species", "-s", multiple=True, required=True,
              help="Chemical species symbols (e.g. Si, Ge). Repeatable.")
@click.option("--num-atoms", "-n", multiple=True, type=int, default=(),
              help="Number of atoms per species. Order matches --species.")
@click.option("--min-distance", "-m", type=float, default=1.0,
              show_default=True,
              help="Minimum pairwise distance in Angstrom.")
@click.option("--num-structures", "-N", type=int, default=1, show_default=True,
              help="Number of structures to generate.")
@click.option("--box-shape", type=click.Choice(["cube", "orthorhombic"]),
              default="cube", show_default=True,
              help="Box shape (used only when cell is derived from density).")
@click.option("--cell-a", type=float, default=None,
              help="Cubic cell edge length in Å.")
@click.option("--cell-abc", type=float, nargs=3, default=None,
              help="Orthorhombic cell edges: --cell-abc a b c (Å).")
@click.option("--seed", type=int, default=None,
              help="Random seed (default: random).")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output.")
def initialize(
    output,
    density,
    species,
    num_atoms,
    min_distance,
    num_structures,
    box_shape,
    cell_a,
    cell_abc,
    seed,
    verbose,
):
    """Initialize random atomic structures for denoising."""
    species_list = list(species)

    # Validate species symbols
    for s in species_list:
        if s not in chemical_symbols:
            raise click.ClickException(
                f"Unknown species symbol: '{s}'. Must be a valid chemical symbol."
            )

    if len(species_list) == 0:
        raise click.ClickException("At least one --species must be given.")

    # Normalise --num-atoms
    counts_list: Optional[List[int]]
    if len(num_atoms) == 0:
        counts_list = None
    else:
        if len(species_list) != len(num_atoms):
            raise click.ClickException(
                f"Number of species ({len(species_list)}) must match number of "
                f"--num-atoms values ({len(num_atoms)})."
            )
        counts_list = [int(n) for n in num_atoms]
        for i, n in enumerate(counts_list):
            if n <= 0:
                raise click.ClickException(
                    f"Number of atoms must be positive. Got {n} for "
                    f"species {species_list[i]}."
                )

    if density is not None and density <= 0:
        raise click.ClickException(f"Density must be positive. Got {density}.")
    if min_distance <= 0:
        raise click.ClickException(
            f"Minimum distance must be positive. Got {min_distance}."
        )

    # Normalise --cell-abc tuple (Click hands back () when unset because
    # nargs=3 does not accept None cleanly).
    cell_abc_clean: Optional[Tuple[float, float, float]]
    if cell_abc in (None, (), (0.0, 0.0, 0.0)):
        cell_abc_clean = None
    else:
        cell_abc_clean = tuple(float(v) for v in cell_abc)  # type: ignore[assignment]
        if any(v <= 0 for v in cell_abc_clean):
            raise click.ClickException(
                f"--cell-abc values must be positive. Got {cell_abc_clean}."
            )
    if cell_a is not None and cell_a <= 0:
        raise click.ClickException(f"--cell-a must be positive. Got {cell_a}.")

    cell, counts_resolved, density_resolved = _resolve_cell_and_counts(
        density=density,
        cell_a=cell_a,
        cell_abc=cell_abc_clean,
        species_list=species_list,
        counts_list=counts_list,
        box_shape=box_shape,
    )
    total_atoms = sum(counts_resolved)

    # Per-structure RNGs (so one bad first structure does not starve later ones).
    if seed is not None:
        click.echo(f"Using random seed: {seed}")

    click.echo(f"Generating {num_structures} structure(s)...")
    click.echo(f"  Species: {species_list}")
    click.echo(f"  Atoms per species: {counts_resolved}")
    click.echo(f"  Total atoms: {total_atoms}")
    click.echo(f"  Density: {density_resolved:.4f} g/cm^3")
    click.echo(f"  Min distance: {min_distance} Angstrom")
    click.echo(
        "  Cell: "
        + ", ".join(f"{cell[i, i]:.3f}" for i in range(3))
        + " Å"
    )
    click.echo(f"  Output: {output}")

    structures = []
    for i in range(num_structures):
        if verbose:
            click.echo(f"Generating structure {i+1}/{num_structures}...")
        structure_rng = np.random.default_rng(
            seed + i if seed is not None else None
        )
        try:
            atoms = _generate_structure(
                species_list=species_list,
                counts_list=counts_resolved,
                cell=cell,
                min_distance=min_distance,
                rng=structure_rng,
                echo=(click.echo if verbose else None),
                verbose=verbose,
            )
        except RuntimeError as e:
            raise click.ClickException(f"Failed to generate structure {i+1}: {e}")
        structures.append(atoms)
        if verbose:
            click.echo(f"  Cell volume: {atoms.get_volume():.2f} Å^3")

    if len(structures) == 1:
        write(output, structures[0])
    else:
        write(output, structures)

    click.echo(f"✓ Successfully wrote {len(structures)} structure(s) to {output}")


if __name__ == "__main__":
    initialize()
