"""Paired significance tests between two heavy-only benchmark runs.

Aggregate exact-match rates cannot tell you whether a difference is real.  On
n=488 the standard error of a rate near 0.57 is about 0.022, so the observed
15-epoch vs 50-epoch gap (0.566 vs 0.564) is far inside noise.  Because both
runs score the *same molecules*, the unpaired standard error is also needlessly
conservative -- an exact McNemar test on the discordant pairs is much more
sensitive and is the correct test here.

Inputs are the ``--per_molecule_out`` JSON files produced by
``benchmark_templatefree_motifs.py`` (or the ``per_molecule`` block of
``benchmark_shared_harness_rules.py`` output).

Usage
-----
  python scripts/paired_significance.py \
      --run 15epoch=results/per_mol_e15.json \
      --run 50epoch=results/per_mol_e50.json \
      --metric strict --sigma 0.0 --method bondnet \
      --output results/paired_significance.json
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from math import comb, sqrt
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="LABEL=PATH",
                   help="Repeatable. Per-molecule outcome JSON, labelled for the report.")
    p.add_argument("--metric", default="strict", choices=["strict", "resonance"])
    p.add_argument("--sigma", default="0.0", help="Noise key inside the per-molecule block.")
    p.add_argument("--method", default="bondnet", help="Method key inside the per-molecule block.")
    p.add_argument("--output", default=None)
    return p.parse_args()


def load_run(spec, sigma, method, metric):
    """Load one run.  ``LABEL=PATH`` or ``LABEL=PATH#METHOD`` to override --method."""
    if "=" not in spec:
        raise ValueError(f"--run expects LABEL=PATH[#METHOD], got {spec!r}")
    label, path = spec.split("=", 1)
    if "#" in path:
        path, method = path.rsplit("#", 1)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    block = payload["per_molecule"]
    # Shared-harness output is flat {method: [...]}; the motif benchmark nests
    # an extra noise level {sigma: {method: [...]}}.
    if sigma in block:
        block = block[sigma]
    if method in block:
        entries = block[method]
    elif len(block) == 1:
        # Unambiguous single-method file: use it rather than forcing --method.
        entries = next(iter(block.values()))
    else:
        raise KeyError(f"method {method!r} not in {path} (available: {sorted(block)})")
    outcomes = {}
    for entry in entries:
        if entry["name"] in outcomes:
            raise ValueError(f"duplicate molecule name {entry['name']!r} in {path}; "
                             "paired tests need a unique key per molecule")
        outcomes[entry["name"]] = int(entry[metric])
    return label.strip(), path, outcomes


def wilson_interval(successes, n, z=1.96):
    if n == 0:
        return (None, None)
    phat = successes / n
    denominator = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denominator
    margin = z * sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def exact_mcnemar(a, b, names):
    """Two-sided exact binomial test on discordant pairs."""
    b_only = sum(1 for name in names if a[name] == 1 and b[name] == 0)
    c_only = sum(1 for name in names if a[name] == 0 and b[name] == 1)
    n = b_only + c_only
    if n == 0:
        return {"a_only": b_only, "b_only": c_only, "discordant": 0, "p_value": 1.0}
    tail = sum(comb(n, k) for k in range(0, min(b_only, c_only) + 1)) / (2 ** n)
    return {"a_only": b_only, "b_only": c_only, "discordant": n,
            "p_value": min(1.0, 2 * tail)}


def paired_difference_ci(a, b, names, z=1.96):
    """Wald CI on the paired difference in rates (Agresti-Min style)."""
    n = len(names)
    b_only = sum(1 for name in names if a[name] == 1 and b[name] == 0)
    c_only = sum(1 for name in names if a[name] == 0 and b[name] == 1)
    diff = (b_only - c_only) / n
    variance = (b_only + c_only - (b_only - c_only) ** 2 / n) / (n * n)
    margin = z * sqrt(max(variance, 0.0))
    return diff, (diff - margin, diff + margin)


def main():
    args = parse_args()
    runs = [load_run(spec, args.sigma, args.method, args.metric) for spec in args.run]

    shared = set.intersection(*(set(outcomes) for _, _, outcomes in runs))
    if not shared:
        raise ValueError("no molecule names shared across runs; cannot pair")
    names = sorted(shared)

    report = {
        "metric": args.metric,
        "sigma": args.sigma,
        "method": args.method,
        "n_paired": len(names),
        "runs": {},
        "pairwise": {},
    }
    for label, path, outcomes in runs:
        successes = sum(outcomes[name] for name in names)
        low, high = wilson_interval(successes, len(names))
        report["runs"][label] = {
            "path": path,
            "n_total_in_file": len(outcomes),
            "rate_on_paired_subset": successes / len(names),
            "wilson_ci95": [low, high],
        }

    for (label_a, _, a), (label_b, _, b) in combinations(runs, 2):
        test = exact_mcnemar(a, b, names)
        diff, (low, high) = paired_difference_ci(a, b, names)
        key = f"{label_a} vs {label_b}"
        report["pairwise"][key] = {
            **test,
            "rate_difference": diff,
            "difference_ci95": [low, high],
            "interpretation": (
                "no detectable difference" if test["p_value"] > 0.05
                else f"{label_a} better" if test["a_only"] > test["b_only"]
                else f"{label_b} better"),
        }

    text = json.dumps(report, indent=1)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    print(f"paired on {len(names)} molecules, metric={args.metric}")
    for label, entry in report["runs"].items():
        low, high = entry["wilson_ci95"]
        print(f"  {label:16s} {entry['rate_on_paired_subset']:.4f}  95% CI [{low:.3f}, {high:.3f}]")
    for key, entry in report["pairwise"].items():
        low, high = entry["difference_ci95"]
        print(f"  {key}: diff={entry['rate_difference']:+.4f} "
              f"CI [{low:+.3f}, {high:+.3f}] discordant={entry['discordant']} "
              f"p={entry['p_value']:.3f} -> {entry['interpretation']}")


if __name__ == "__main__":
    main()
