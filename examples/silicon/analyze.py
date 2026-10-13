"""Compare generated Si structures against a reference ensemble.

For every set of structures this script computes the averaged g(r), the
bond-angle distribution, the coordination numbers, the ring statistics and
the Tersoff energy, and reports how far each set is from the reference.
All bond-based quantities use the same cutoff, taken from the first minimum
of the reference g(r).

Usage:
    python analyze.py --reference data/density_2.5 \
        --set unconditional="results/uncond/*/*/*_final.xyz" \
        --set pdf="results/pdf/*/*/*_final.xyz" \
        --output results/summary.json --figure figures/structure_2.5.png

With --surrogate exafs=xas/Si_exafs.ckpt (repeatable), the atom-averaged
spectrum predicted by the surrogate is also compared with the reference.
"""

import argparse
import glob
import json
from collections import OrderedDict

import numpy as np
import torch
from ase.io import read

from glass.metrics import (
    compute_adf,
    compute_coordination,
    compute_pdf,
    compute_rings,
    compute_tersoff_metrics,
)
from glass.lit.modules.guidance import create_guidance_model
from glass.utils.atoms import initialize_atoms

PDF_CUTOFF = 8.0
PDF_BINS = 200
MAX_CN = 8
MAX_RING = 10
# Debye-equation intensities with the same settings as the guidance
DIFFRACTION = {
    name: create_guidance_model(name, "cpu", cutoff=5.0, element_names=["Si"])
    for name in ("xrd", "nd")
}


def min_distance(atoms):
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


SURROGATES = {}


def load_surrogates(specs):
    from glass.lit.modules import LitSpecNet

    for spec in specs:
        name, path = spec.split("=", 1)
        net = LitSpecNet.load_from_checkpoint(path, map_location="cpu")
        SURROGATES[name] = net.ema_model.eval()


def surrogate_spectrum(model, species, pos, cell):
    from glass.nn import periodic_radius_graph

    edge_index, edge_vec = periodic_radius_graph(pos, 5.0, cell)
    edge_attr = torch.hstack([edge_vec, edge_vec.norm(dim=-1, keepdim=True)])
    return model(species, edge_index, edge_attr).mean(0).numpy()


def analyze_set(files, cutoff):
    """Per-structure quantities for a list of xyz files."""
    out = {"g_r": [], "adf": [], "cn": [], "rings": [], "energy": [], "dmin": [],
           "xrd": [], "nd": []}
    for name in SURROGATES:
        out[name] = []
    for f in files:
        atoms = read(f)
        pdf = compute_pdf(atoms, cutoff=PDF_CUTOFF, bin_size=PDF_BINS)
        adf = compute_adf(atoms, cutoff=cutoff, auto_cutoff=False)
        cn = compute_coordination(atoms, cutoff=cutoff, auto_cutoff=False)
        rings = compute_rings(atoms, cutoff=cutoff, maxlength=MAX_RING, auto_cutoff=False)
        out["g_r"].append(pdf.g_r)
        out["adf"].append(adf.adf)
        out["cn"].append(cn.coordination_numbers)
        out["rings"].append(rings.ring_counts / len(atoms))
        out["energy"].append(compute_tersoff_metrics(atoms).energy_per_atom)
        out["dmin"].append(min_distance(atoms))
        _, species, pos, cell = initialize_atoms(atoms)
        with torch.no_grad():
            for name, model in DIFFRACTION.items():
                out[name].append(model(pos, species, cell).numpy())
            for name, model in SURROGATES.items():
                out[name].append(surrogate_spectrum(model, species, pos, cell))
    out["r"] = pdf.r
    out["angles"] = np.degrees(adf.angles)
    return out


def cn_histogram(cn_list):
    cn = np.concatenate(cn_list)
    return np.bincount(np.clip(cn, 0, MAX_CN), minlength=MAX_CN + 1) / len(cn)


def summarize(data, ref):
    g, g_ref = np.mean(data["g_r"], 0), np.mean(ref["g_r"], 0)
    a, a_ref = np.mean(data["adf"], 0), np.mean(ref["adf"], 0)
    h, h_ref = cn_histogram(data["cn"]), cn_histogram(ref["cn"])
    rings = np.mean(data["rings"], 0)
    energy = np.array(data["energy"])

    def rel_err(key):
        a, b = np.mean(data[key], 0), np.mean(ref[key], 0)
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    return OrderedDict(
        n_structures=len(data["g_r"]),
        pdf_rmse=float(np.sqrt(np.mean((g - g_ref) ** 2))),
        adf_rmse=float(np.sqrt(np.mean((a - a_ref) ** 2))),
        # Earth mover's distance between integer histograms
        coordination_emd=float(np.abs(np.cumsum(h - h_ref)).sum()),
        # relative L2 error of the mean diffraction intensity
        xrd_error=rel_err("xrd"),
        nd_error=rel_err("nd"),
        **{f"{name}_error": rel_err(name) for name in SURROGATES},
        mean_coordination=float(np.mean(np.concatenate(data["cn"]))),
        fraction_cn={str(k): float(h[k]) for k in range(2, 7)},
        adf_peak_deg=float(data["angles"][np.argmax(a)]),
        rings_per_atom={str(k): float(rings[k]) for k in range(3, MAX_RING + 1)},
        tersoff_energy_mean=float(energy.mean()),
        tersoff_energy_std=float(energy.std()),
        min_distance=float(np.min(data["dmin"])),
    )


def plot(sets, ref, cutoff, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    styles = {"reference": dict(color="k", lw=2.5, alpha=0.35)}
    for name, data in [("reference", ref)] + list(sets.items()):
        kw = styles.get(name, dict(lw=1.4))
        axes[0].plot(data["r"], np.mean(data["g_r"], 0), label=name, **kw)
        axes[1].plot(data["angles"], np.mean(data["adf"], 0), label=name, **kw)
        axes[2].plot(np.arange(MAX_CN + 1), cn_histogram(data["cn"]), "o-", label=name, **kw)
    axes[0].set(xlabel="r (Å)", ylabel="g(r)", xlim=(1.5, PDF_CUTOFF))
    axes[1].set(xlabel="bond angle (deg)", ylabel="ADF", xlim=(40, 180))
    axes[2].set(xlabel=f"coordination number (r < {cutoff:.2f} Å)", ylabel="fraction",
                xlim=(1.5, 7.5))
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, help="Directory with reference *.xyz")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=GLOB",
                        help="Named glob of generated structures (repeatable)")
    parser.add_argument("--output", default="summary.json")
    parser.add_argument("--figure", default=None)
    parser.add_argument("--surrogate", action="append", default=[], metavar="NAME=CKPT",
                        help="EXAFS/XANES surrogate used to compare averaged spectra")
    args = parser.parse_args()
    load_surrogates(args.surrogate)

    ref_files = sorted(glob.glob(f"{args.reference}/*.xyz"))
    minima = [compute_pdf(read(f)).first_minima_position for f in ref_files]
    cutoff = float(np.mean([m for m in minima if m is not None]))
    print(f"Bond cutoff from the reference g(r): {cutoff:.3f} Å")

    ref = analyze_set(ref_files, cutoff)
    summary = OrderedDict(bond_cutoff=cutoff, reference=summarize(ref, ref))
    sets = OrderedDict()
    for spec in args.set:
        name, pattern = spec.split("=", 1)
        files = sorted(glob.glob(pattern))
        if not files:
            raise SystemExit(f"No files match {pattern}")
        sets[name] = analyze_set(files, cutoff)
        summary[name] = summarize(sets[name], ref)

    with open(args.output, "w") as fh:
        json.dump(summary, fh, indent=2)

    keys = ["pdf_rmse", "adf_rmse", "coordination_emd", "xrd_error", "nd_error",
            "mean_coordination", "adf_peak_deg", "tersoff_energy_mean", "min_distance"]
    labels = ["PDF", "ADF", "CN EMD", "XRD", "ND", "<CN>", "ADF peak", "E (eV)", "d_min"]
    print(f"{'set':<16}" + "".join(f"{k:>10}" for k in labels) + f"{'CN=4':>8}")
    for name, s in summary.items():
        if name == "bond_cutoff":
            continue
        print(f"{name:<16}" + "".join(f"{s[k]:>10.3f}" for k in keys)
              + f"{s['fraction_cn']['4']:>8.3f}")

    for name in SURROGATES:
        print(f"{name} error: " + ", ".join(
            f"{k} {s[name + '_error']:.4f}" for k, s in summary.items() if k != "bond_cutoff"))

    if args.figure:
        plot(sets, ref, cutoff, args.figure)


if __name__ == "__main__":
    main()
