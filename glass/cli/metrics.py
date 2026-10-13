"""CLI command for computing structural metrics.

This module provides a command-line interface for computing comprehensive
structural metrics for atomic structures.
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import click
import numpy as np
from ase.io import read
from glass.metrics import (
    StructuralMetrics,
    compute_adf,
    compute_all_metrics,
    compute_coordination,
    compute_dihedrals,
    compute_pdf,
    compute_rings,
)
from glass.metrics.errors import (
    compute_all_errors,
    compute_weighted_error,
    coordination_emd,
    coordination_mean_error,
    pdf_mae,
    pdf_rmse,
)
from glass.metrics.utils import load_metrics_from_json


@click.command(
    "metrics",
    help="""
Compute comprehensive structural metrics for atomic structures.

\b
This command calculates various structural analysis metrics including:
  - PDF (Pair Distribution Function)
  - ADF (Angular Distribution Function)
  - Coordination number distribution
  - Dihedral angle distribution (optional)
  - Ring statistics (optional)
  - Tersoff energy and forces (optional, pure Si)

The coordination cutoff can be automatically determined from the PDF first minimum,
or specified manually.

EXAMPLES:

\b
  # Compute all metrics for single structure
  glass metrics structure.xyz

\b
  # Batch process multiple structures
  glass metrics ./structures/*.xyz --output metrics.json

\b
  # Compute with automatic cutoff detection
  glass metrics structure.xyz --auto-cutoff

\b
  # Use specific cutoff
  glass metrics structure.xyz --coord-cutoff 3.2 --adf-cutoff 3.2

\b
  # Include optional metrics
  glass metrics structure.xyz --include-dihedrals --include-rings --include-tersoff

\b
OUTPUT FORMAT:
  The output is a JSON file containing all computed metrics for each structure.
""",
)
@click.argument("structures", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="metrics.json",
    show_default=True,
    help="Output JSON file path.",
)
@click.option(
    "--pdf-cutoff",
    type=float,
    default=8.0,
    show_default=True,
    help="Maximum r for PDF computation (Angstrom).",
)
@click.option(
    "--coord-cutoff",
    type=float,
    default=None,
    help="Cutoff for coordination number (Angstrom). If not set, uses PDF first minimum.",
)
@click.option(
    "--adf-cutoff",
    type=float,
    default=None,
    help="Cutoff for ADF computation (Angstrom). If not set, uses PDF first minimum.",
)
@click.option(
    "--auto-cutoff/--no-auto-cutoff",
    default=True,
    show_default=True,
    help="Automatically determine ADF and coordination cutoffs from PDF first minimum.",
)
@click.option(
    "--include-dihedrals/--no-dihedrals",
    "include_dihedrals",
    default=False,
    help="Compute dihedral angle distribution.",
)
@click.option(
    "--include-rings/--no-rings",
    "include_rings",
    default=False,
    help="Compute ring statistics.",
)
@click.option(
    "--rings-maxlength",
    type=int,
    default=10,
    show_default=True,
    help="Maximum ring size to consider.",
)
@click.option(
    "--include-tersoff/--no-tersoff",
    "include_tersoff",
    default=False,
    help="Compute Tersoff energy and force distribution metrics (Si only).",
)
@click.option(
    "--tersoff-device",
    type=str,
    default="cpu",
    show_default=True,
    help="Torch device for Tersoff computation ('cpu' or 'cuda').",
)
@click.option(
    "--indent",
    type=int,
    default=2,
    show_default=True,
    help="JSON indentation level.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "summary"]),
    default="json",
    show_default=True,
    help="Output format: json (full data) or summary (text summary).",
)
def metrics(
    structures: List[str],
    output: str,
    pdf_cutoff: float,
    coord_cutoff: Optional[float],
    adf_cutoff: Optional[float],
    auto_cutoff: bool,
    include_dihedrals: bool,
    include_rings: bool,
    rings_maxlength: int,
    include_tersoff: bool,
    tersoff_device: str,
    indent: int,
    output_format: str,
):
    """Compute structural metrics for atomic structures."""

    results = {}

    for structure_path in structures:
        path = Path(structure_path)
        click.echo(f"Processing: {path.name}")

        try:
            # Load structure
            atoms = read(structure_path)

            # Compute metrics
            metrics_obj = compute_all_metrics(
                atoms,
                pdf_cutoff=pdf_cutoff,
                adf_cutoff=adf_cutoff,
                coord_cutoff=coord_cutoff,
                auto_cutoff=auto_cutoff,
                include_dihedrals=include_dihedrals,
                include_rings=include_rings,
                rings_maxlength=rings_maxlength,
                include_tersoff=include_tersoff,
                tersoff_device=tersoff_device,
            )

            # Store results
            results[path.name] = metrics_obj.to_dict()

            # Print summary
            click.echo(f"  Atoms: {metrics_obj.n_atoms}")
            click.echo(f"  Composition: {metrics_obj.composition}")
            click.echo(f"  Density: {metrics_obj.density:.4f} atoms/Å³")

            if metrics_obj.pdf.coord_cutoff:
                click.echo(
                    f"  Coord cutoff (auto): {metrics_obj.pdf.coord_cutoff:.3f} Å"
                )
            click.echo(
                f"  Mean coordination: {metrics_obj.coordination.mean_coordination:.2f}"
            )

            if metrics_obj.pdf.first_peak_position:
                click.echo(
                    f"  PDF first peak: {metrics_obj.pdf.first_peak_position:.3f} Å"
                )
            if metrics_obj.adf.dominant_angle_degree:
                click.echo(
                    f"  ADF dominant angle: {metrics_obj.adf.dominant_angle_degree:.1f}°"
                )

            if include_dihedrals and metrics_obj.dihedrals:
                click.echo(
                    f"  Dihedrals: {len(metrics_obj.dihedrals.dihedral_angles)} found"
                )

            if include_rings and metrics_obj.rings:
                click.echo(
                    f"  Rings: {metrics_obj.rings.total_rings:g} found (maxlength={metrics_obj.rings.maxlength})"
                )
                ring_counts = metrics_obj.rings.ring_counts
                if metrics_obj.rings.total_rings > 0:
                    top_rings = [
                        f"{int(i)}-member: {int(cnt)}"
                        for i, cnt in enumerate(ring_counts)
                        if i > 2
                    ]
                    if top_rings:
                        click.echo(f"    Top: {', '.join(top_rings)}")

            if include_tersoff and metrics_obj.tersoff:
                t = metrics_obj.tersoff
                click.echo(
                    f"  Tersoff E/atom: {t.energy_per_atom:.4f} eV/atom  "
                    f"F_rms: {t.forces_rms:.4f} eV/Å  F_max: {t.forces_max:.4f} eV/Å"
                )

            click.echo()

        except Exception as e:
            click.echo(f"  ERROR: {e}", err=True)
            results[path.name] = {"error": str(e)}

    # Output results
    if output_format == "json":
        output_data = {
            "metadata": {
                "n_structures": len(structures),
                "parameters": {
                    "pdf_cutoff": pdf_cutoff,
                    "coord_cutoff": coord_cutoff,
                    "adf_cutoff": adf_cutoff,
                    "auto_cutoff": auto_cutoff,
                    "include_dihedrals": include_dihedrals,
                    "include_rings": include_rings,
                    "rings_maxlength": rings_maxlength,
                    "include_tersoff": include_tersoff,
                    "tersoff_device": tersoff_device,
                },
            },
            "structures": results,
        }

        with open(output, "w") as f:
            json.dump(output_data, f, indent=indent)

        click.echo(f"Saved metrics to {output}")

    else:  # summary format
        click.echo("\n" + "=" * 60)
        click.echo("SUMMARY")
        click.echo("=" * 60)

        for name, data in results.items():
            if "error" in data:
                click.echo(f"\n{name}: ERROR - {data['error']}")
                continue

            click.echo(f"\n{name}:")
            click.echo(f"  {data['n_atoms']} atoms ({data['composition']})")
            click.echo(f"  Density: {data['density']:.4f} atoms/Å³")

            pdf = data.get("pdf", {})
            coord = data.get("coordination", {})

            if pdf.get("coord_cutoff"):
                click.echo(f"  Coord cutoff: {pdf['coord_cutoff']:.3f} Å")
            click.echo(f"  Mean CN: {coord.get('mean_coordination', 'N/A'):.2f}")

            if pdf.get("first_peak_position"):
                click.echo(f"  PDF peak: {pdf['first_peak_position']:.3f} Å")


# Also create individual metric commands
@click.command("pdf", help="Compute PDF for a structure.")
@click.argument("structure", type=click.Path(exists=True))
@click.option("--cutoff", type=float, default=8.0, help="PDF cutoff radius")
@click.option(
    "--output", "-o", type=click.Path(), default="pdf.json", help="Output file"
)
def compute_pdf_command(structure: str, cutoff: float, output: str):
    """Compute PDF for a single structure."""
    atoms = read(structure)
    pdf_metrics = compute_pdf(atoms, cutoff=cutoff)

    data = {
        "r": pdf_metrics.r.tolist(),
        "g_r": pdf_metrics.g_r.tolist(),
        "first_peak_position": pdf_metrics.first_peak_position,
        "first_minima_position": pdf_metrics.first_minima_position,
        "coord_cutoff": pdf_metrics.coord_cutoff,
    }

    with open(output, "w") as f:
        json.dump(data, f, indent=2)

    click.echo(f"PDF saved to {output}")
    click.echo(f"First peak: {pdf_metrics.first_peak_position:.3f} Å")
    click.echo(f"Coord cutoff: {pdf_metrics.coord_cutoff:.3f} Å")


@click.command("coordination", help="Compute coordination numbers.")
@click.argument("structure", type=click.Path(exists=True))
@click.option(
    "--cutoff",
    type=float,
    default=None,
    help="Coordination cutoff (auto-detected if not set)",
)
@click.option(
    "--output", "-o", type=click.Path(), default="coordination.json", help="Output file"
)
def compute_coordination_command(structure: str, cutoff: Optional[float], output: str):
    """Compute coordination for a single structure."""
    atoms = read(structure)
    coord_metrics = compute_coordination(
        atoms, cutoff=cutoff, auto_cutoff=(cutoff is None)
    )

    data = {
        "coordination_numbers": coord_metrics.coordination_numbers.tolist(),
        "mean": coord_metrics.mean_coordination,
        "std": coord_metrics.std_coordination,
        "histogram": coord_metrics.coordination_histogram.tolist(),
        "histogram_bins": coord_metrics.histogram_bins.tolist(),
    }

    with open(output, "w") as f:
        json.dump(data, f, indent=2)

    click.echo(f"Coordination saved to {output}")
    click.echo(
        f"Mean CN: {coord_metrics.mean_coordination:.2f} ± {coord_metrics.std_coordination:.2f}"
    )
    for i, n in enumerate(coord_metrics.coordination_histogram):
        click.echo(
            f"{i:02d}: {n:d}"
        )


@click.command("rings", help="Compute ring statistics.")
@click.argument("structure", type=click.Path(exists=True))
@click.option(
    "--cutoff",
    type=float,
    default=None,
    help="Neighbor cutoff (auto-detected if not set)",
)
@click.option("--maxlength", type=int, default=10, help="Maximum ring size")
@click.option(
    "--output", "-o", type=click.Path(), default="rings.json", help="Output file"
)
def compute_rings_command(
    structure: str, cutoff: Optional[float], maxlength: int, output: str
):
    """Compute ring statistics for a single structure.

    Identifies shortest-path rings using the Franzblau algorithm.
    """
    atoms = read(structure)
    rings_metrics = compute_rings(
        atoms, cutoff=cutoff, maxlength=maxlength, auto_cutoff=(cutoff is None)
    )

    data = {
        "ring_lengths": rings_metrics.ring_lengths.tolist(),
        "ring_counts": rings_metrics.ring_counts.tolist(),
        "ring_fractions": rings_metrics.ring_fractions.tolist(),
        "total_rings": rings_metrics.total_rings,
        "cutoff": rings_metrics.cutoff,
        "maxlength": rings_metrics.maxlength,
    }

    with open(output, "w") as f:
        json.dump(data, f, indent=2)

    click.echo(f"Ring statistics saved to {output}")
    click.echo(f"Total rings: {rings_metrics.total_rings:g}")
    if rings_metrics.total_rings > 0:
        ring_counts = rings_metrics.ring_counts
        for idx, cnt in enumerate(ring_counts):
            if idx < 3:
                continue
            frac = rings_metrics.ring_fractions[idx]
            click.echo(
                f"  {int(idx)}-member: {int(cnt)} ({frac:.1f}%)"
            )


@click.command(
    "compare",
    help="""
Compare structural metrics between reference and target structures.

Computes various error metrics between two structures based on their
PDF, ADF, and coordination distributions. Can load metrics from
JSON files or compute them on the fly.

EXAMPLES:

\b
  # Compare two structure files
  glass compare ref.xyz target.xyz

\b
  # Compare using pre-computed metrics
  glass compare ref_metrics.json target_metrics.json --from-json

\b
  # Compare with specific structure from multi-structure JSON
  glass compare ref_metrics.json target_metrics.json --from-json \
    --ref-structure "Si_2.5_00.xyz" --target-structure "Si_2.5_01.xyz"

\b
  # Compare with detailed output
  glass compare ref.xyz target.xyz --detailed

\b
  # Output results as JSON
  glass compare ref.xyz target.xyz --output errors.json

\b
METRICS COMPUTED:
  - PDF: RMSE, MAE, area between curves, cosine similarity, R-chi²
  - PDF: Peak position and height errors
  - ADF: RMSE, cosine similarity
  - Coordination: EMD, histogram RMSE, mean/std errors
""",
)
@click.argument("reference", type=click.Path(exists=True))
@click.argument("target", type=click.Path(exists=True))
@click.option(
    "--from-json/--from-xyz",
    default=False,
    show_default=True,
    help="Load from JSON metrics files instead of computing from XYZ.",
)
@click.option(
    "--ref-structure",
    type=str,
    default=None,
    help="Structure name to load from reference JSON (for multi-structure files).",
)
@click.option(
    "--target-structure",
    type=str,
    default=None,
    help="Structure name to load from target JSON (for multi-structure files).",
)
@click.option(
    "--pdf-cutoff",
    type=float,
    default=8.0,
    show_default=True,
    help="PDF cutoff when computing from XYZ.",
)
@click.option(
    "--detailed/--summary",
    default=False,
    help="Show detailed per-metric output.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Save errors to JSON file.",
)
@click.option(
    "--indent",
    type=int,
    default=2,
    show_default=True,
    help="JSON indentation level.",
)
def compare_command(
    reference: str,
    target: str,
    from_json: bool,
    ref_structure: Optional[str],
    target_structure: Optional[str],
    pdf_cutoff: float,
    detailed: bool,
    output: Optional[str],
    indent: int,
):
    """Compare structural metrics between reference and target."""

    try:
        if from_json:
            # Load metrics from JSON files
            click.echo(f"Loading reference from {reference}...")
            ref_metrics = load_metrics_from_json(
                reference, structure_name=ref_structure
            )

            click.echo(f"Loading target from {target}...")
            target_metrics = load_metrics_from_json(
                target, structure_name=target_structure
            )
        else:
            # Compute metrics from structure files
            click.echo(f"Computing metrics for reference: {reference}...")
            ref_atoms = read(reference)
            ref_metrics = compute_all_metrics(ref_atoms, pdf_cutoff=pdf_cutoff)

            click.echo(f"Computing metrics for target: {target}...")
            target_atoms = read(target)
            target_metrics = compute_all_metrics(target_atoms, pdf_cutoff=pdf_cutoff)

        # Compute all errors
        errors = compute_all_errors(ref_metrics, target_metrics)

        # Compute weighted error
        weights = {
            "pdf_rmse": 1.0,
            "coordination_emd": 1.0,
            "adf_rmse": 0.5,
        }
        weighted = compute_weighted_error(errors, weights)

        # Print results
        click.echo("\n" + "=" * 60)
        click.echo("COMPARISON RESULTS")
        click.echo("=" * 60)

        click.echo(f"\nReference: {reference}")
        click.echo(f"Target: {target}")
        click.echo(f"\nWeighted Score: {weighted:.4f} (lower is better)")

        # PDF errors
        click.echo("\n--- PDF Errors ---")
        click.echo(f"  RMSE:                 {errors['pdf_rmse']:.6f}")
        click.echo(f"  MAE:                  {errors['pdf_mae']:.6f}")
        click.echo(f"  Area between curves:  {errors['pdf_area']:.6f}")
        click.echo(f"  Cosine similarity:    {errors['pdf_cosine']:.6f}")
        click.echo(f"  R-chi²:               {errors['pdf_r_chi2']:.6f}")

        if errors["pdf_peak_position_error"] is not None:
            click.echo(
                f"  Peak position error:  {errors['pdf_peak_position_error']:.4f} Å"
            )
        if errors["pdf_peak_height_error"] is not None:
            click.echo(
                f"  Peak height error:    {errors['pdf_peak_height_error']:.2f}%"
            )

        # ADF errors
        click.echo("\n--- ADF Errors ---")
        click.echo(f"  RMSE:                 {errors['adf_rmse']:.6f}")
        click.echo(f"  Cosine similarity:    {errors['adf_cosine']:.6f}")

        # Coordination errors
        click.echo("\n--- Coordination Errors ---")
        click.echo(f"  EMD (Wasserstein):    {errors['coordination_emd']:.6f}")
        click.echo(f"  Histogram RMSE:       {errors['coordination_rmse']:.6f}")
        click.echo(f"  Mean CN error:        {errors['coordination_mean_error']:.4f}")
        click.echo(f"  Std CN error:         {errors['coordination_std_error']:.4f}")

        # Detailed output
        if detailed:
            click.echo("\n--- All Error Metrics (detailed) ---")
            for key, value in sorted(errors.items()):
                if value is not None:
                    click.echo(f"  {key}: {value:.6f}")

        # Save to file if requested
        if output:
            output_data = {
                "reference": str(reference),
                "target": str(target),
                "weighted_error": weighted,
                "weights_used": weights,
                "errors": {k: v for k, v in errors.items() if v is not None},
            }
            with open(output, "w") as f:
                json.dump(output_data, f, indent=indent)
            click.echo(f"\nSaved errors to {output}")

        click.echo("\n" + "=" * 60)

    except Exception as e:
        click.echo(f"ERROR: {e}", err=True)
        raise click.Abort()


# Export commands
__all__ = [
    "metrics",
    "compute_pdf_command",
    "compute_coordination_command",
    "compute_rings_command",
    "compare_command",
]
