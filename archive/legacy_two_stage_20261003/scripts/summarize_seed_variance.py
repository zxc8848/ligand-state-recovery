#!/usr/bin/env python3
r"""Summarise multi-seed training variance for the main pose488 result.

Reads the per-seed score_knodle_vs_bondnet.py outputs (each carries the BondNet
strict / resonance-aware(C/N/P/S) / canonical-species numbers for one seed on the
identical 488 poses) and reports mean, sample standard deviation, and range across
seeds, plus a paper-ready "mean +/- sample SD" string for each headline metric.

Usage:
  python scripts/summarize_seed_variance.py \
      --runs results/seedvar_knodle_s42.json results/seedvar_knodle_s43.json \
             results/seedvar_knodle_s44.json \
      --output results/seed_variance_summary.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

METRICS = ("strict", "resonance", "species")
LABELS = {"strict": "Strict exact",
          "resonance": "Resonance-aware exact (C/N/P/S)",
          "species": "Canonical species"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True,
                   help="per-seed score_knodle_vs_bondnet.py JSON outputs")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    per_seed = {}
    for path in args.runs:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        bn = d.get("bondnet") or d
        per_seed[Path(path).stem] = {m: float(bn[m]) for m in METRICS if m in bn}

    if len(per_seed) < 2:
        raise SystemExit("need at least two seed runs to report variance")

    report = {"n_seeds": len(per_seed), "per_seed": per_seed, "summary": {}}
    print(f"multi-seed variance over {len(per_seed)} runs: {', '.join(per_seed)}\n")
    print(f"{'metric':<34}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}   paper-ready")
    for m in METRICS:
        vals = [s[m] for s in per_seed.values() if m in s]
        if len(vals) < 2:
            continue
        mean = st.mean(vals)
        sd = st.stdev(vals)
        lo, hi = min(vals), max(vals)
        half_range = (hi - lo) / 2.0
        paper = f"{mean:.3f} $\\pm$ {sd:.3f}"
        report["summary"][m] = {"label": LABELS[m], "mean": mean, "std": sd,
                                "min": lo, "max": hi, "half_range": half_range,
                                "values": vals, "paper_ready": paper}
        print(f"{LABELS[m]:<34}{mean:>8.3f}{sd:>8.3f}{lo:>8.3f}{hi:>8.3f}   {mean:.3f} +/- {sd:.3f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")
    print("\nReport as mean +/- sample standard deviation across seeds 42-44 in Table 3 / the text.")


if __name__ == "__main__":
    main()
