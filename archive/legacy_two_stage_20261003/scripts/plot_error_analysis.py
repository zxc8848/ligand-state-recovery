#!/usr/bin/env python3
r"""Figure 4: residual error composition and the valence-fix effect.

Left panel: residual per-edge errors grouped by chemical environment (terminal-oxy
groups dominate at 52%). Right panel: strict-exact rate before vs after the
neutral-hypervalent-P/S valence-catalogue fix, by motif stratum, showing the
hypervalent-sulfur stratum jumping from worst to best.

Left-panel counts come from the per-edge summary; the small fixed set of strata
values on the right are the measured before/after rates (Progress doc section D).

Usage:
  python scripts/plot_error_analysis.py \
      --summary results/per_edge_to50_valencefix_summary.json \
      --output paper2/figures/fig4_error_analysis.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", default="results/per_edge_to50_valencefix_summary.json",
                   help="Output of analyze_per_edge.py (for context; the environment "
                        "counts are taken from the measured breakdown below).")
    p.add_argument("--output", default="paper2/figures/fig4_error_analysis.pdf")
    return p.parse_args()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    args = parse_args()

    # Measured residual-error environment split (Progress doc section E; 545
    # error edges after the valence fix).
    env_labels = ["terminal-oxy\n(H-decided)", "other O-bond", "C--N", "C--C"]
    env_counts = [285, 124, 77, 50]

    # Measured strict-exact by stratum, before vs after the valence fix
    # (Progress doc section D).
    strata = ["all", "hypervalent S", "aromatic"]
    before = [0.5635, 0.5250, 0.6000]
    after = [0.5840, 0.6875, 0.6241]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))

    # Left: error environment bar (share).
    total = sum(env_counts)
    shares = [c / total * 100 for c in env_counts]
    colours = ["#C44E52", "#DD8452", "#4C72B0", "#55A868"]
    ax1.barh(range(len(env_labels)), shares, color=colours)
    ax1.set_yticks(range(len(env_labels)))
    ax1.set_yticklabels(env_labels, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("share of residual error edges (%)", fontsize=9)
    ax1.set_title("Where the residual errors are", fontsize=9)
    for i, (s, c) in enumerate(zip(shares, env_counts)):
        ax1.text(s + 1, i, f"{s:.0f}% ({c})", va="center", fontsize=8)
    ax1.set_xlim(0, max(shares) + 12)
    ax1.tick_params(labelsize=8)

    # Right: before/after by stratum grouped bars.
    x = np.arange(len(strata))
    w = 0.38
    ax2.bar(x - w / 2, before, w, label="before fix", color="#8C8C8C")
    ax2.bar(x + w / 2, after, w, label="after valence fix", color="#4C72B0")
    ax2.set_xticks(x)
    ax2.set_xticklabels(strata, fontsize=8)
    ax2.set_ylabel("strict graph exact", fontsize=9)
    ax2.set_title("Neutral hypervalent P/S valence fix", fontsize=9)
    ax2.set_ylim(0, 0.8)
    ax2.legend(fontsize=8, frameon=False)
    for xi, (b, a) in enumerate(zip(before, after)):
        ax2.text(xi + w / 2, a + 0.01, f"+{a - b:.03f}", ha="center", fontsize=7)
    ax2.tick_params(labelsize=8)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
