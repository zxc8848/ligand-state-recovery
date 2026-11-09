#!/usr/bin/env python3
r"""Figure 3: terminal X--O bond-length distributions, ideal vs real pose.

Visualises the Cohen's d collapse from Table 1: in CCD ideal coordinates the
single- and double-bond length populations are cleanly separated (d ~ 9-11),
while in real PDBBind poses they overlap almost completely (d ~ 0). This is the
information-theoretic reason the single/double assignment of delocalized
terminal-oxy groups cannot be recovered from (Z, R, Q).

Re-measures lengths directly from the SDFs (same procedure as
measure_terminal_oxy_separability.py) so the figure and Table 1 cannot drift.

Usage:
  python scripts/plot_terminal_oxy.py \
      --ideal_sdf data/ccd_heavy_leakagefree.sdf \
      --pose_sdf  data/pdbbind_ccd_pose500.sdf \
      --output    paper2/figures/fig3_terminal_oxy.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ideal_sdf", required=True)
    p.add_argument("--pose_sdf", required=True)
    p.add_argument("--output", default="paper2/figures/fig3_terminal_oxy.pdf")
    p.add_argument("--allowed_atomic_nums", type=int, nargs="+",
                   default=[6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53, 83])
    return p.parse_args()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from scripts.measure_terminal_oxy_separability import collect, cohens_d

    args = parse_args()
    ideal, _ = collect(args.ideal_sdf, args.allowed_atomic_nums, None)
    pose, _ = collect(args.pose_sdf, args.allowed_atomic_nums, None)

    # Two rows (carboxyl C--O, phosphate P--O) x two columns (ideal, pose).
    groups = [("carboxyl_CO", "Carboxyl C--O"), ("phosphate_PO", "Phosphate P--O")]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.2), sharex="row")
    colours = {"single": "#4C72B0", "double": "#C44E52"}

    for row, (key, title) in enumerate(groups):
        for col, (dataset, dlabel) in enumerate([(ideal, "CCD ideal"), (pose, "PDBBind pose")]):
            ax = axes[row][col]
            s = np.asarray(dataset[(key, 1)])
            d = np.asarray(dataset[(key, 2)])
            lo = min(s.min(), d.min()) if len(s) and len(d) else 1.1
            hi = max(s.max(), d.max()) if len(s) and len(d) else 1.7
            bins = np.linspace(lo - 0.02, hi + 0.02, 40)
            if len(s):
                ax.hist(s, bins=bins, alpha=0.6, color=colours["single"],
                        label="single", density=True)
            if len(d):
                ax.hist(d, bins=bins, alpha=0.6, color=colours["double"],
                        label="double", density=True)
            dd = cohens_d(d, s)
            ax.set_title(f"{title} -- {dlabel}\nCohen's $d$ = {dd:.2f}" if dd is not None
                         else f"{title} -- {dlabel}", fontsize=9)
            if row == 1:
                ax.set_xlabel("bond length (Å)", fontsize=9)
            if col == 0:
                ax.set_ylabel("density", fontsize=9)
            ax.tick_params(labelsize=8)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
