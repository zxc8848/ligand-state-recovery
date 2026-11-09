#!/usr/bin/env python3
r"""Quantify the PDB-deposition-era breakdown of the benchmark poses.

Supports the Knodle-comparison caveat: a substantial fraction of the test poses
predate Knodle's release and could lie within its (unavailable) training data.
The year bucket is read from the PDBBind source path recorded in the pose manifest
(e.g. .../P-L/2001-2010/...). Scoring-only; no model or GPU.

Usage:
  python scripts/knodle_overlap_years.py \
      --manifest data/pdbbind_ccd_pose500_manifest.csv \
      --output results/knodle_overlap_years.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="data/pdbbind_ccd_pose500_manifest.csv")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    buckets = Counter()
    n = 0
    for row in csv.DictReader(open(args.manifest, encoding="utf-8")):
        if row.get("status") != "written":
            continue
        n += 1
        m = re.search(r"(20\d\d-20\d\d)", row.get("source", ""))
        buckets[m.group(1) if m else "unknown"] += 1

    pre2011 = buckets.get("2001-2010", 0)
    report = {
        "n_written": n,
        "buckets": dict(buckets),
        "pre_2011_count": pre2011,
        "pre_2011_fraction": round(pre2011 / n, 4) if n else None,
    }
    print(f"written poses: {n}")
    for k in sorted(buckets):
        print(f"  {k}: {buckets[k]}  ({100*buckets[k]/n:.1f}%)")
    print(f"\n>= {100*pre2011/n:.1f}% are from 2001-2010, predating Knodle's release; "
          f"overlap with its training data cannot be excluded (see paper Sec. Comparison with Knodle).")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
