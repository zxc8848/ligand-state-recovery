#!/usr/bin/env python3
r"""Figure 6: where the clean-pose parity between BondNet and Knodle ends.

On the clean 488-pose benchmark the two systems are tied on the primary
resonance-aware endpoint (0.801 vs 0.797, McNemar p = 0.905). That tie is measured
in the condition most favourable to the external comparator: undisturbed
bound-ligand poses, the distribution Knodle's SVM was developed on. This figure
plots the same method-neutral endpoint as the input moves away from that condition,
along two axes measured in scripts/probe_noise_ccd.py:

  left  -- Gaussian coordinate noise on the same 488 poses (sigma 0.00-0.15 A);
  right -- the CCD ideal heavy-only control (n = 1000), clean and at sigma 0.10 A.

Both methods receive bit-identical coordinates in every condition, and a molecule a
method fails to return is scored as an incorrect prediction.

Usage:
  python scripts/plot_robustness.py \
      --summary results/paper2_noise_ccd_probe/summary.json \
      --output paper2/figures/fig6_robustness.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Okabe-Ito blue / vermillion: validated for deuteranopia, protanopia and
# tritanopia separation (worst adjacent CVD dE 21.9) and for greyscale print.
BONDNET = "#0072B2"
KNODLE = "#D55E00"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", default="results/paper2_noise_ccd_probe/summary.json")
    p.add_argument("--output", default="paper2/figures/fig6_robustness.pdf")
    p.add_argument("--metric", default="resonance",
                   choices=["strict", "resonance", "species"])
    return p.parse_args()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args()
    conditions = json.loads(Path(args.summary).read_text(encoding="utf-8"))["conditions"]

    def series(prefix, method):
        rows = sorted(
            ((entry["sigma"], entry[method][args.metric])
             for key, entry in conditions.items() if key.startswith(prefix)),
            key=lambda r: r[0])
        return [r[0] for r in rows], [r[1] for r in rows]

    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 200,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True,
                             gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.08})

    panels = [
        (axes[0], "pdbbind", "(a) PDBBind bound poses ($n=488$)", "Coordinate noise $\\sigma$ (\u00c5)"),
        (axes[1], "ccd", "(b) CCD ideal control ($n=1000$)", "Coordinate noise $\\sigma$ (\u00c5)"),
    ]
    for ax, prefix, title, xlabel in panels:
        for method, colour, label, marker in (
                ("bondnet", BONDNET, "BondNet", "o"),
                ("knodle", KNODLE, "Knodle", "s")):
            x, y = series(prefix, method)
            ax.plot(x, y, color=colour, linewidth=2.0, marker=marker,
                    markersize=5.5, markerfacecolor=colour,
                    markeredgecolor="white", markeredgewidth=1.0,
                    label=label, clip_on=False, zorder=3)
        ax.set_title(title, loc="left", pad=8)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("Resonance-aware exact recovery")
    axes[0].set_xticks([0.00, 0.05, 0.10, 0.15])
    axes[1].set_xticks([0.00, 0.10])
    for ax in axes:
        ax.xaxis.set_major_formatter(lambda v, _pos: f"{v:.2f}")
    axes[1].set_xlim(-0.012, 0.112)

    # The one point the figure exists to qualify: parity holds only at sigma = 0
    # on real poses. Annotate it rather than leaving the reader to infer it.
    bx, by = series("pdbbind", "bondnet")
    axes[0].annotate("parity only here\n($p=0.905$)",
                     xy=(bx[0], by[0]), xytext=(0.012, 0.945),
                     color=MUTED, fontsize=8, ha="left", va="top",
                     arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8,
                                     shrinkA=0, shrinkB=3))
    axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0.02, 0.02))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
    print(f"wrote {output} and {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
