#!/usr/bin/env python3
r"""Localise error modes from the per-edge dump.

This exists because two rounds of inferring edge-level mechanisms from
molecule-level statistics were wrong.  Everything reported here is counted
directly on the edges themselves.

Reports, for every true->predicted transition that matters:
  * where the edge sits (element pair, ring membership)
  * whether the reference context satisfies the class-5 training condition
    (C/N centre, exactly two terminal oxygens, bond-order sum 3)
  * how confident the model was, and what its runner-up class was

The last one decides whether a decode-side constraint can help at all.  If the
model puts most of its mass on the wrong class, forbidding that class just moves
the error somewhere else -- which is exactly what the element gate did.

Usage
-----
  python scripts/analyze_per_edge.py --per_edge results/per_edge_to50.jsonl \
      --output results/per_edge_to50_summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

CLASS = {0: "nobond", 1: "single", 2: "double", 3: "triple", 4: "aromatic", 5: "deloc"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per_edge", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--top", type=int, default=6)
    return p.parse_args()


def symbol(z):
    return {6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P", 16: "S",
            17: "Cl", 33: "As", 35: "Br", 53: "I", 83: "Bi"}.get(z, str(z))


def main():
    args = parse_args()
    rows = [json.loads(line) for line in
            Path(args.per_edge).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("sigma", 0.0) == 0.0]
    print(f"loaded {len(rows)} edge rows")

    errors = [r for r in rows if not r["correct"]]
    transitions = Counter(f"{CLASS.get(r['true'])}->{CLASS.get(r['pred'])}" for r in errors)
    print(f"\ntotal edges {len(rows)}, errors {len(errors)} "
          f"({100 * len(errors) / max(len(rows), 1):.2f}%)\n")
    print("top error transitions:")
    for name, count in transitions.most_common(args.top):
        print(f"  {name:22s} {count:5d}")

    report = {"n_edges": len(rows), "n_errors": len(errors),
              "transitions": dict(transitions), "breakdowns": {}}

    for target in ("single->deloc", "double->deloc", "double->single", "single->double"):
        subset = [r for r in errors
                  if f"{CLASS.get(r['true'])}->{CLASS.get(r['pred'])}" == target]
        if not subset:
            continue
        print(f"\n=== {target}  (n={len(subset)}) ===")

        pairs = Counter("".join(sorted((symbol(r["z_i"]), symbol(r["z_j"])))) for r in subset)
        print("  element pair:      " + ", ".join(f"{k}:{v}" for k, v in pairs.most_common(6)))

        termo = Counter(str(r.get("n_terminal_o")) for r in subset)
        print("  terminal O on centre: " + ", ".join(f"{k}:{v}" for k, v in
                                                     sorted(termo.items(), key=lambda x: -x[1])))

        trained = Counter(str(r.get("matches_class5_training_pattern")) for r in subset)
        print("  matches class-5 training pattern: "
              + ", ".join(f"{k}:{v}" for k, v in trained.items()))

        ring = Counter(str(r.get("in_ring")) for r in subset)
        print("  in ring:           " + ", ".join(f"{k}:{v}" for k, v in ring.items()))

        # Confidence: was the correct class even a close runner-up?
        margins, correct_rank, recoverable = [], Counter(), 0
        for r in subset:
            probs = r.get("probs") or []
            if len(probs) < 6:
                continue
            order = sorted(range(len(probs)), key=lambda c: -probs[c])
            rank = order.index(r["true"]) if r["true"] < len(probs) else -1
            correct_rank[rank] += 1
            margins.append(probs[r["pred"]] - probs[r["true"]])
            if rank == 1:
                recoverable += 1
        if margins:
            margins.sort()
            mid = margins[len(margins) // 2]
            print(f"  median p(pred) - p(true): {mid:+.3f}")
            print("  rank of the CORRECT class in the model's own ranking: "
                  + ", ".join(f"#{k + 1}:{v}" for k, v in sorted(correct_rank.items())))
            print(f"  -> correct class was runner-up in {recoverable}/{len(subset)} "
                  f"({100 * recoverable / len(subset):.0f}%). A decode constraint can only "
                  "help where this is high.")

        report["breakdowns"][target] = {
            "n": len(subset),
            "element_pairs": dict(pairs),
            "n_terminal_o": dict(termo),
            "matches_class5_training_pattern": dict(trained),
            "in_ring": dict(ring),
            "correct_class_rank": {str(k): v for k, v in correct_rank.items()},
            "median_prob_margin": (margins[len(margins) // 2] if margins else None),
            "correct_was_runner_up": recoverable,
        }

    # How many molecules would become exact if a given transition were fixed?
    by_molecule = defaultdict(list)
    for r in rows:
        by_molecule[r["name"]].append(r)
    headroom = {}
    for target in report["breakdowns"]:
        fixed = 0
        for name, edges in by_molecule.items():
            bad = [e for e in edges if not e["correct"]]
            if bad and all(f"{CLASS.get(e['true'])}->{CLASS.get(e['pred'])}" == target
                           for e in bad):
                fixed += 1
        headroom[target] = fixed
    total_mols = len(by_molecule)
    exact_now = sum(1 for edges in by_molecule.values() if all(e["correct"] for e in edges))
    print(f"\n=== molecule-level headroom (n={total_mols}, currently exact {exact_now}) ===")
    for target, fixed in sorted(headroom.items(), key=lambda x: -x[1]):
        print(f"  fixing ONLY {target:16s} -> +{fixed:3d} molecules "
              f"({(exact_now + fixed) / total_mols:.4f} strict)")
    report["molecule_headroom"] = {"n_molecules": total_mols, "exact_now": exact_now,
                                   "if_transition_fixed": headroom}

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
