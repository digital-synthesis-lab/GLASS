"""Render a denoising trajectory as a GIF.

Each frame shows the structure with its bonds (left) and its g(r) against the
reference ensemble (right). The trajectory is the ``NN_traj.xyz`` file written
by ``glass generate --save-traj``; the ``params.json`` next to it tells where
the unconditional pass ends and the guided pass begins. By default only the
guided pass is shown.

Bonds are drawn between atoms closer than the first minimum after the first
peak of the reference g(r), unless --bond-cutoff is given.

Usage:
    python make_gif.py results/pdf/init_00/.../00_traj.xyz \
        --reference data/density_2.5 --output figures/denoising.gif
"""

import argparse
import glob
import io
import json
import os

import numpy as np
from ase.io import read
from PIL import Image
from scipy.spatial.transform import Rotation

from glass.metrics import compute_pdf

PDF_CUTOFF = 8.0
ATOM_COLOR = "#5b8bd0"
BOND_COLOR = "#7a7a7a"
# Viewing direction: rotate the cell by 10 degrees about x, then -20 about y
VIEW = Rotation.from_euler("xy", [10, -20], degrees=True).as_matrix().T


def first_minimum_after_peak(r, g):
    """Position of the first minimum of g(r) after its first maximum."""
    peak = int(np.argmax(g > 0.5 * g.max()))  # start of the first peak
    peak += int(np.argmax(np.diff(g[peak:]) < 0))  # top of the first peak
    return float(r[peak + int(np.argmax(np.diff(g[peak:]) > 0))])


def half_bonds(atoms, cutoff):
    """Segments from each atom to the midpoint of each of its bonds.

    Using the minimum-image vector, bonds inside the cell appear whole and
    bonds that cross the cell boundary appear as stubs on both sides.
    """
    pos, cell = atoms.positions, atoms.cell.array
    vec = pos[None, :, :] - pos[:, None, :]
    vec -= np.round(vec @ np.linalg.inv(cell)) @ cell
    dist = np.linalg.norm(vec, axis=-1)
    i, j = np.nonzero((dist < cutoff) & (dist > 0))
    return i, pos[i], pos[i] + 0.5 * vec[i, j]


def draw_structure(ax, atoms, cutoff, radius=0.4):
    from matplotlib.patches import Circle

    proj = lambda x: x @ VIEW
    pos = proj(atoms.positions)
    i, start, end = half_bonds(atoms, cutoff)
    start, end = proj(start), proj(end)

    # Painter's algorithm: atoms and their half bonds share the depth of the atom
    for k in range(len(i)):
        z = pos[i[k], 2]
        ax.plot([start[k, 0], end[k, 0]], [start[k, 1], end[k, 1]],
                color=BOND_COLOR, lw=1.6, solid_capstyle="round", zorder=z - 1e-3)
    for a in range(len(pos)):
        ax.add_patch(Circle(pos[a, :2], radius, facecolor=ATOM_COLOR, edgecolor="k",
                            lw=0.6, zorder=pos[a, 2]))

    # Cell edges, drawn behind everything
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    corners = proj(corners @ atoms.cell.array)
    for a in range(8):
        for b in range(a + 1, 8):
            if bin(a ^ b).count("1") == 1:
                ax.plot(*corners[[a, b], :2].T, color="k", lw=0.6, ls=":", zorder=-1e6)

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_axis_off()


def render(atoms, cutoff, r, g, r_ref, g_ref, label):
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.0, 3.6),
                                   gridspec_kw=dict(width_ratios=[1, 1.25]))
    draw_structure(ax0, atoms, cutoff)
    ax1.plot(r_ref, g_ref, color="k", lw=2.5, alpha=0.35, label="reference")
    ax1.plot(r, g, color="#d0605b", lw=1.6, label="generated")
    ax1.axvline(cutoff, color="k", lw=0.8, ls="--", alpha=0.5)
    ax1.set(xlabel="r (Å)", ylabel="g(r)", xlim=(1.0, PDF_CUTOFF), ylim=(0, 3.2))
    ax1.legend(frameon=False, loc="upper right")
    ax1.set_title(label, fontsize=11)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trajectory")
    parser.add_argument("--reference", required=True, help="Directory with reference *.xyz")
    parser.add_argument("--output", default="denoising.gif")
    parser.add_argument("--stage", choices=["guided", "all"], default="guided",
                        help="Show only the guided pass, or the unconditional pass too")
    parser.add_argument("--bond-cutoff", type=float, default=None,
                        help="Bond length cutoff in Å (default: from the reference g(r))")
    parser.add_argument("--frames", type=int, default=60, help="Frames per pass")
    parser.add_argument("--duration", type=int, default=80, help="Milliseconds per frame")
    args = parser.parse_args()

    traj = read(args.trajectory, ":")
    with open(os.path.join(os.path.dirname(args.trajectory), "params.json")) as fh:
        params = json.load(fh)

    # Frame 0 is the init; the unconditional prepass (if any) comes next.
    guided = params["mode"] == "cond"
    n_uncond = params["uncond_tstep"] - 1 if (not guided or params["uncond_prepass"]) else 0
    passes = []
    if args.stage == "all" or not guided:
        passes.append(("unconditional", 0, n_uncond))
    if guided:
        passes.append((f"{params['guidance_type'].upper()}-guided", n_uncond, len(traj) - 1))

    refs = [read(f) for f in sorted(glob.glob(f"{args.reference}/*.xyz"))]
    ref_pdfs = [compute_pdf(a, cutoff=PDF_CUTOFF) for a in refs]
    r_ref, g_ref = ref_pdfs[0].r, np.mean([p.g_r for p in ref_pdfs], 0)
    cutoff = args.bond_cutoff or first_minimum_after_peak(r_ref, g_ref)
    print(f"Bond cutoff: {cutoff:.2f} Å")

    images = []
    for label, start, end in passes:
        if end <= start:
            continue
        # Denser sampling near the end of each pass, where most structure forms.
        idx = np.unique(np.round(start + (end - start) * np.linspace(0, 1, args.frames) ** 0.6))
        for i in idx.astype(int):
            atoms = traj[i]
            pdf = compute_pdf(atoms, cutoff=PDF_CUTOFF)
            images.append(render(atoms, cutoff, pdf.r, pdf.g_r, r_ref, g_ref, label))
    # Hold the final frame.
    images += [images[-1]] * 15
    images[0].save(args.output, save_all=True, append_images=images[1:],
                   duration=args.duration, loop=0, optimize=True)
    print(f"Wrote {len(images)} frames to {args.output}")


if __name__ == "__main__":
    main()
