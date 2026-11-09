#!/usr/bin/env python3
r"""Summarise the single- vs two-stage candidate-set coverage sweep.

Reads the per-noise coverage JSONs from both arms and prints a table of
coverage@N (labeling) at each noise level, with the single - two gap. The
hypothesis under test: the gap in favour of single-stage should be ~0 on clean
poses and grow with noise, because two-stage under-represents connectivity
multi-modality (its readout is conditioned on one Stage-1 topology).

Usage:
  python scripts/summarize_coverage_comparison.py \
      --two results/cov_two_stage_n0.00.json ... \
      --single results/cov_single_stage_n0.00.json ... \
      --output results/coverage_single_vs_two_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--two", nargs="+", required=True, help="two-stage coverage JSONs")
    p.add_argument("--single", nargs="+", required=True, help="single-stage coverage JSONs")
    p.add_argument("--report_n", type=int, nargs="+", default=[1, 2, 4, 8],
                   help="which coverage@N columns to print")
    p.add_argument("--output", default=None)
    return p.parse_args()


def load(paths):
    """{noise: {N: coverage}} keyed by the noise recorded in each file.

    Missing files are skipped with a warning rather than crashing, so a partial
    sweep (e.g. one arm failed) still summarises what completed.
    """
    out = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"warning: missing {path} -- skipped")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        noise = float(d.get("noise", 0.0))
        cov = d.get("coverage_labeling") or {}
        out[noise] = {int(k): v for k, v in cov.items()}
        out[noise]["_n_molecules"] = d.get("n_molecules")
        out[noise]["_single_stage"] = d.get("single_stage")
    return out


def main():
    args = parse_args()
    two = load(args.two)
    single = load(args.single)
    noises = sorted(set(two) & set(single))

    rows = []
    print(f"{'noise':>6}  " + "  ".join(f"two@{n:<2} single@{n:<2}  gap@{n:<2}"
                                        for n in args.report_n))
    for noise in noises:
        cells = []
        row = {"noise": noise}
        for n in args.report_n:
            t = two[noise].get(n)
            s = single[noise].get(n)
            gap = (s - t) if (t is not None and s is not None) else None
            row[f"two@{n}"] = t
            row[f"single@{n}"] = s
            row[f"gap@{n}"] = gap
            cells.append(f"{t:>6.3f} {s:>8.3f}  {gap:>+6.3f}"
                         if gap is not None else f"{'n/a':>6} {'n/a':>8}  {'n/a':>6}")
        rows.append(row)
        print(f"{noise:>6.2f}  " + "  ".join(cells))

    print("\ninterpretation: gap ~ 0 on clean poses is expected (connectivity is "
          "near-deterministic); a gap that GROWS with noise supports single-stage "
          "for candidate-set coverage under connectivity ambiguity.")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({"report_n": args.report_n, "rows": rows},
                                                indent=1), encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
