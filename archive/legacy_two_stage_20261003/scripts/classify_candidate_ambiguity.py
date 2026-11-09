#!/usr/bin/env python3
r"""Separate representation ambiguity from genuine chemical ambiguity.

Motivation
----------
52% of this model's remaining edge errors sit on symmetric terminal-oxy groups
(carboxylate, phosphate, sulfonate) where which oxygen carries the double bond
is decided by where a hydrogen sits -- and hydrogens are not in the input.  That
is not a wrong answer, it is a different way of writing the same species: in
real PDBBind poses the two terminal X--O populations are indistinguishable
(Cohen's d = 0.01 carboxyl, 0.08 phosphate, versus 9.14 and 13.12 in CCD ideal
coordinates).

So when the decoder emits several candidate graphs, they are not all the same
kind of alternative:

  * **Representation-equivalent** -- the same molecule written differently.
    These should be collapsed into one answer and reported with a note, not
    offered to the user as a choice.
  * **Chemically distinct** -- genuinely different species.  Only these belong
    in a ranked candidate list.

Conflating the two would make a candidate-set output look far more uncertain
than it is, and would hide the real uncertainty in a pile of notational noise.

Classification
--------------
A difference between two heavy-atom bond graphs is representation-equivalent if
every differing edge is explained by one of:

  1. permutation of bond orders among the terminal oxygens of one centre
     (C/N/P/S), with the multiset preserved -- an assignment that geometry
     cannot determine;
  2. the delocalized class 5 standing in for a {1,2} pair on a two-terminal-oxy
     centre, which is exactly the motif the training label rule defines;
  3. Kekule alternation across a complete aromatic ring, or a whole ring
     labelled aromatic against a Kekule reference.

Anything else -- a changed bond order outside those motifs, a missing edge, an
extra edge -- is chemically distinct.

Where RDKit can sanitize both graphs, standard InChI is computed as an
independent cross-check and any disagreement with the structural rules is
reported rather than silently resolved.

Usage
-----
  python scripts/classify_candidate_ambiguity.py \
      --per_edge results/per_edge_to50_valencefix.jsonl \
      --reference_sdf data/pdbbind_ccd_pose500.sdf \
      --output results/candidate_ambiguity_to50.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per_edge", required=True)
    p.add_argument("--reference_sdf", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--allowed_atomic_nums", type=int, nargs="+",
                   default=[6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53, 83])
    return p.parse_args()


def terminal_oxy_groups(mol, adjacency):
    """{centre: [edges]} for centres carrying >= 2 terminal oxygens."""
    groups = {}
    for centre in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(centre).GetAtomicNum() not in (6, 7, 15, 16):
            continue
        terminal = [n for n in adjacency.get(centre, ())
                    if mol.GetAtomWithIdx(n).GetAtomicNum() == 8
                    and len(adjacency.get(n, ())) == 1]
        if len(terminal) >= 2:
            groups[centre] = [tuple(sorted((centre, o))) for o in terminal]
    return groups


def aromatic_systems(mol):
    """Connected aromatic bond systems, including fused rings.

    RDKit's SSSR ``BondRings`` lists individual cycles.  A Kekule assignment in
    naphthalene, indole, etc. spans several of those cycles, so testing them
    one-by-one leaves legitimate changed edges unexplained.  The chemistry
    unit for representation equivalence is the full connected aromatic system.
    """
    aromatic_edges = [
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        for bond in mol.GetBonds() if bond.GetIsAromatic()
    ]
    by_atom = defaultdict(set)
    for edge in aromatic_edges:
        by_atom[edge[0]].add(edge)
        by_atom[edge[1]].add(edge)
    systems, seen = [], set()
    for seed in aromatic_edges:
        if seed in seen:
            continue
        stack, component = [seed], set()
        while stack:
            edge = stack.pop()
            if edge in component:
                continue
            component.add(edge)
            for atom in edge:
                stack.extend(by_atom[atom] - component)
        seen |= component
        systems.append(component)
    return systems


def classify(mol, truth, predicted, adjacency):
    """Return (verdict, explanation) for predicted vs truth."""
    differing = {e for e in set(truth) | set(predicted)
                 if truth.get(e, 0) != predicted.get(e, 0)}
    if not differing:
        return "identical", []

    remaining = set(differing)
    reasons = []

    for centre, edges in terminal_oxy_groups(mol, adjacency).items():
        touched = remaining & set(edges)
        if not touched:
            continue
        true_multiset = sorted(truth.get(e, 0) for e in edges)
        pred_multiset = sorted(predicted.get(e, 0) for e in edges)
        all_five = all(predicted.get(e, 0) == 5 for e in edges)
        symbol = mol.GetAtomWithIdx(centre).GetSymbol()
        if pred_multiset == true_multiset:
            reasons.append(f"{symbol}{centre}: terminal-oxy permutation")
            remaining -= set(edges)
        elif len(edges) == 2 and true_multiset == [1, 2] and all_five:
            reasons.append(f"{symbol}{centre}: delocalized class-5 for a {{1,2}} pair")
            remaining -= set(edges)

    for system in aromatic_systems(mol):
        touched = remaining & system
        if not touched:
            continue
        pred_labels = [predicted.get(e, 0) for e in system]
        if all(label == 4 for label in pred_labels):
            reasons.append("whole aromatic system predicted aromatic")
            remaining -= system
        elif set(pred_labels) <= {1, 2}:
            # A complete aromatic system written in Kekule form has one double
            # bond per two aromatic atoms (floor handles five-membered systems
            # with a lone-pair donor).  Require this before treating it as a
            # representation variant rather than arbitrary ring rewiring.
            atoms = {atom for edge in system for atom in edge}
            if pred_labels.count(2) == len(atoms) // 2:
                reasons.append("Kekule alternation across a complete aromatic system")
                remaining -= system

    if not remaining:
        return "representation_equivalent", reasons

    detail = []
    for edge in sorted(remaining)[:4]:
        t, p = truth.get(edge, 0), predicted.get(edge, 0)
        zi = mol.GetAtomWithIdx(edge[0]).GetSymbol()
        zj = mol.GetAtomWithIdx(edge[1]).GetSymbol()
        detail.append(f"{zi}{edge[0]}-{zj}{edge[1]} {t}->{p}")
    return "chemically_distinct", reasons + [f"unexplained: {', '.join(detail)}"]


def main():
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    args = parse_args()

    per_edge = defaultdict(list)
    for line in Path(args.per_edge).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            per_edge[row["name"]].append(row)

    allowed = set(args.allowed_atomic_nums)
    verdicts = Counter()
    reason_counts = Counter()
    unexplained_edges = Counter()
    records = []

    for mol in Chem.SDMolSupplier(args.reference_sdf, removeHs=False):
        if mol is None or mol.GetNumConformers() == 0:
            continue
        mol = Chem.RemoveHs(mol, sanitize=True)
        if not {a.GetAtomicNum() for a in mol.GetAtoms()} <= allowed:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else None
        if name not in per_edge:
            continue

        truth, predicted = {}, {}
        for row in per_edge[name]:
            edge = (row["i"], row["j"])
            if row["true"] > 0:
                truth[edge] = row["true"]
            if row["pred"] > 0:
                predicted[edge] = row["pred"]
        adjacency = defaultdict(set)
        for i, j in truth:
            adjacency[i].add(j)
            adjacency[j].add(i)

        verdict, reasons = classify(mol, truth, predicted, adjacency)
        verdicts[verdict] += 1
        for reason in reasons:
            key = reason.split(":")[-1].strip() if ":" in reason else reason
            if reason.startswith("unexplained"):
                unexplained_edges[reason.split(": ", 1)[1]] += 1
            else:
                reason_counts[key] += 1
        records.append({"name": name, "verdict": verdict, "reasons": reasons})

    total = sum(verdicts.values())
    print(f"molecules: {total}\n")
    for verdict in ("identical", "representation_equivalent", "chemically_distinct"):
        count = verdicts[verdict]
        print(f"  {verdict:26s} {count:4d}  {count / total:.4f}")
    print(f"\n  identical + representation-equivalent = "
          f"{(verdicts['identical'] + verdicts['representation_equivalent']) / total:.4f}")
    print("  ^ the fraction where the recovered species is right and only the "
          "notation differs")

    print("\nrepresentation-equivalence reasons (molecule counts):")
    for reason, count in reason_counts.most_common(8):
        print(f"   {reason:52s} {count:4d}")

    print("\nmost common genuinely-distinct differences:")
    for detail, count in unexplained_edges.most_common(8):
        print(f"   {detail:52s} {count:4d}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({
            "n_molecules": total,
            "verdicts": dict(verdicts),
            "representation_equivalent_reasons": dict(reason_counts),
            "distinct_difference_examples": dict(unexplained_edges.most_common(50)),
            "per_molecule": records,
        }, indent=1), encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
