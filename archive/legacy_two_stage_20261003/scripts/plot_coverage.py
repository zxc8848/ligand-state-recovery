#!/usr/bin/env python3
r"""Figure 5: coverage@N of the ranked candidate-set output (real 488 poses).

Plots two coverage@N curves from an evaluate_candidate_sets.py run:
  * equivalence-aware species coverage (``coverage_equivalence``, aka the retained
    ``coverage_species``) -- the reference species appears among the top N decoded
    feasible graphs under the declared resonance equivalence rules, with a
    classify() equivalence fallback when a candidate cannot be sanitized. This is
    NOT the strict canonical-SMILES species endpoint of Table 4 (which counts a
    build failure as a miss); it is deliberately more permissive.
  * strict-labeling coverage -- the exact reference bond-order labeling appears
    among the top N.
The gap between them is the notational degeneracy (delocalized terminal-oxy,
equivalent-oxygen permutation) that species coverage absorbs and strict labeling
does not. Both curves are the full 488-molecule figures used in the text.

Usage:
  python scripts/plot_coverage.py \
      --candidate_sets results/cov_two_stage_n0.00.json \
      --output paper2/figures/fig5_coverage.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate_sets", default="results/candidate_sets_to50.json",
                   help="evaluate_candidate_sets.py output on the FULL 488 with the "
                        "MAIN paper refiner (diverse80k_..._to50). This is the "
                        "paper-consistent source for Fig 5. Falls back to the "
                        "ablation-checkpoint sweep or the smoke only if absent.")
    p.add_argument("--output", default="paper2/figures/fig5_coverage.pdf")
    return p.parse_args()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args()
    path = Path(args.candidate_sets)
    if not path.exists():
        for fallback in ("results/cov_two_stage_n0.00.json",
                         "results/cov_single_stage_n0.00.json",
                         "results/candidate_sets_smoke.json"):
            if Path(fallback).exists():
                print(f"note: {path} absent, using {fallback}")
                path = Path(fallback)
                break
        else:
            raise SystemExit(f"no candidate-set result found at {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    n_mol = data.get("n_molecules")
    species = data.get("coverage_species") or data.get("coverage_equivalence")
    labeling = data.get("coverage_labeling")
    if species is None and labeling is None:
        raise SystemExit("file has neither coverage_species nor coverage_labeling")

    ns = sorted(int(k) for k in (species or labeling))

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    if species is not None:
        ys = [species[str(n)] for n in ns]
        ax.plot(ns, ys, "-o", color="#4C72B0",
                label=f"equivalence-aware species coverage@N ({ys[0]:.2f}$\\to${ys[-1]:.2f})")
        for x, y in zip(ns, ys):
            if x in (1, 2, ns[-1]):
                ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=7, color="#4C72B0")
    if labeling is not None:
        yl = [labeling[str(n)] for n in ns]
        ax.plot(ns, yl, "-s", color="#DD8452",
                label=f"strict-label coverage@N ({yl[0]:.2f}$\\to${yl[-1]:.2f})")
        for x, y in zip(ns, yl):
            if x in (1, ns[-1]):
                ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                            xytext=(0, -12), ha="center", fontsize=7, color="#DD8452")

    ax.set_xlabel("N (number of ranked candidates)", fontsize=9)
    ax.set_ylabel("coverage@N", fontsize=9)
    title = "Ranked candidate-set coverage"
    if n_mol:
        title += f" ($n={n_mol}$ poses)"
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0.45, 0.95)
    ax.set_xticks(ns)
    ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.5)
    ax.legend(fontsize=8, frameon=False, loc="center right")
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.png')} (source: {path.name}, n={n_mol})")


if __name__ == "__main__":
    main()
