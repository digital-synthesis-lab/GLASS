"""Compare an EXAFS or XANES surrogate with the FEFF spectra it was trained on.

The data folder must have the layout used by `glass train --model-type spec`:

    DATA/structures/<name>.xyz        structures
    DATA/<exafs|xanes>/train/<name>.txt   per-atom spectra, one row per atom

For every structure, the surrogate predicts one spectrum per atom from the
clean (noise-free) positions. The script reports the per-atom error and the
error of the spectrum averaged over all atoms, which is what an experiment
measures, grouped by the prefix of the file name (e.g. amorph_Si_2.5).

Usage:
    python evaluate_surrogate.py Si_exafs.ckpt --data DATA --figure exafs.png
"""

import argparse
import json
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from glass.lit.modules import LitSpecNet
from glass.nn import periodic_radius_graph
from glass.utils.atoms import initialize_atoms

CUTOFF = 5.0  # graph cutoff used to train the surrogates


def predict(model, atoms, device):
    _, species, pos, cell = initialize_atoms(atoms)
    species, pos, cell = species.to(device), pos.to(device), cell.to(device)
    edge_index, edge_vec = periodic_radius_graph(pos, CUTOFF, cell)
    edge_attr = torch.hstack([edge_vec, edge_vec.norm(dim=-1, keepdim=True)])
    with torch.no_grad():
        return model(species, edge_index, edge_attr).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None, help="JSON file with the errors")
    parser.add_argument("--figure", default=None, help="Averaged spectra of a few structures")
    args = parser.parse_args()

    net = LitSpecNet.load_from_checkpoint(args.checkpoint, map_location=args.device)
    model = net.ema_model.eval()
    out_dim = net.hparams.out_dim
    kind = {400: "exafs", 100: "xanes"}[out_dim]

    data = Path(args.data)
    groups = defaultdict(lambda: defaultdict(list))
    examples = {}
    for xyz in sorted((data / "structures").glob("*.xyz")):
        y = np.loadtxt(data / kind / "train" / f"{xyz.stem}.txt")
        pred = predict(model, read(xyz), args.device)
        group = re.sub(r"_\d+$", "", xyz.stem)
        g = groups[group]
        g["atom_mse"].append(np.mean((pred - y) ** 2))
        g["atom_var"].append(np.var(y))
        mean_pred, mean_ref = pred.mean(0), y.mean(0)
        g["mean_rel"].append(np.linalg.norm(mean_pred - mean_ref) / np.linalg.norm(mean_ref))
        examples.setdefault(group, (mean_ref, mean_pred))

    summary = OrderedDict(checkpoint=str(args.checkpoint), spectrum=kind)
    print(f"{'group':<18}{'n':>4}{'atom RMSE':>12}{'atom R2':>10}{'mean rel. err':>15}")
    for name, g in sorted(groups.items()):
        mse, var = np.mean(g["atom_mse"]), np.mean(g["atom_var"])
        row = OrderedDict(n=len(g["atom_mse"]), atom_rmse=float(np.sqrt(mse)),
                          atom_r2=float(1 - mse / var),
                          mean_rel_error=float(np.mean(g["mean_rel"])))
        summary[name] = row
        print(f"{name:<18}{row['n']:>4}{row['atom_rmse']:>12.4f}{row['atom_r2']:>10.3f}"
              f"{row['mean_rel_error']:>15.4f}")

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(summary, fh, indent=2)

    if args.figure:
        import matplotlib.pyplot as plt

        shown = [k for k in sorted(examples) if k.startswith("amorph")]
        fig, axes = plt.subplots(1, len(shown), figsize=(3.2 * len(shown), 2.8), sharey=True)
        for ax, name in zip(np.atleast_1d(axes), shown):
            ref, pred = examples[name]
            ax.plot(ref, color="k", lw=2.2, alpha=0.4, label="FEFF")
            ax.plot(pred, color="#d0605b", lw=1.2, label="surrogate")
            ax.set_title(name.replace("amorph_", "a-").replace("_", " ") + " g/cm³", fontsize=10)
            ax.set_xlabel("grid point")
        np.atleast_1d(axes)[0].set_ylabel(f"{kind.upper()} (atom average)")
        np.atleast_1d(axes)[0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.figure, dpi=150)


if __name__ == "__main__":
    main()
