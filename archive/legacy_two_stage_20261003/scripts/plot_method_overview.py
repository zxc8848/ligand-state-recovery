#!/usr/bin/env python3
"""Draw the manuscript overview of heavy-atom BondNet perception.

The figure deliberately distinguishes information available at inference from
training-only labels.  It also shows that the global charge is an optional
supplied constraint, rather than an atom-wise charge label.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="paper2/figures/fig1_overview.pdf")
    return p.parse_args()


def box(ax, xy, width, height, title, body, color):
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.018,rounding_size=0.025",
                           linewidth=1.1, edgecolor="#30445c", facecolor=color)
    ax.add_patch(patch)
    x, y = xy
    ax.text(x + width / 2, y + height * .68, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#152335")
    ax.text(x + width / 2, y + height * .35, body, ha="center", va="center",
            fontsize=8.4, color="#263849", linespacing=1.35)


def arrow(ax, src, dst, label=None, dashed=False):
    props = dict(arrowstyle="-|>", mutation_scale=12, linewidth=1.2,
                 color="#455a70", linestyle="--" if dashed else "-")
    ax.add_patch(FancyArrowPatch(src, dst, **props))
    if label:
        x, y = (src[0] + dst[0]) / 2, (src[1] + dst[1]) / 2
        ax.text(x, y + .025, label, ha="center", va="bottom", fontsize=7.4, color="#455a70")


def main():
    args = parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Inference path, laid out for a single-column manuscript width.
    box(ax, (.07, .73), .37, .18, "Inference input", "heavy elements $Z$  |  pose coordinates $R$\noptional molecular net charge $Q$", "#eaf2fb")
    box(ax, (.56, .73), .37, .18, "Candidate graph", "pairs within 2.5 Å  |  degree cap = 15\nno template and no H coordinates", "#eef6ef")
    box(ax, (.07, .43), .37, .18, "Stage 1 (frozen)", "PaiNN connectivity encoder\nmessage scaffold; edge logit + probability", "#f4effb")
    box(ax, (.56, .43), .37, .18, "Stage 2 refiner", "four-layer PaiNN; six edge classes\nreadout over all candidate pairs", "#fff3e8")
    box(ax, (.07, .13), .37, .18, "Constrained decode", "MILP: valence + global-charge conservation\nKekulé / sanitization checks", "#fdecef")
    box(ax, (.56, .13), .37, .18, "Output", "top graph or ranked feasible candidates\nbonds, latent states, accept / reject", "#edf5f7")

    arrow(ax, (.44, .82), (.56, .82))
    arrow(ax, (.745, .73), (.255, .61))
    arrow(ax, (.44, .52), (.56, .52), "$p_{\mathrm{retain}}\geq0.15$")
    arrow(ax, (.745, .43), (.255, .31))
    arrow(ax, (.44, .22), (.56, .22))

    ax.text(.5, .975, "Template-free heavy-atom bond perception", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#152335")
    ax.text(.5, .945, "Arrows: inference.  Training labels construct the catalogue and supervise both stages.",
            ha="center", va="center", fontsize=7.5, color="#536577")
    ax.text(.5, .055, "Training only: Diverse-80k labelled graphs supervise Stage 1/2 and define the valence-state catalogue.",
            ha="center", va="center", fontsize=7.4, color="#536577")
    fig.tight_layout(pad=.25)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
