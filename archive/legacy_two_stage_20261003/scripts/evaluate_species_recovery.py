#!/usr/bin/env python3
r"""Species-level recovery on the 488-molecule PDBBind pose set.

This is the clean, single-answer species metric for the paper. It replaces the
two irreconcilable oracles that preceded it:

  * ``classify()`` structural rules gave 0.8012 but had a ring-Kekule blind spot
    (9 molecules with identical SMILES were called chemically distinct);
  * a canonical-SMILES oracle gave 0.5881, but 83 of that gap was an artifact of
    building delocalized (class-5) terminal-oxy groups as two single bonds,
    turning a neutral carboxyl COOH into a geminal diol, plus 62 outright build
    failures.

Both are fixed here by using ONE build path for reference and prediction:

  * class-5 terminal-oxy groups are kekulized to {single, double} before
    building, so a delocalized carboxyl and a conventional carboxyl canonicalise
    to the same species (``_kekulize_delocalized_oxy`` in
    ``evaluate_candidate_sets``);
  * hydrogens are inferred by RDKit from valence, never taken from the decoder's
    all-zero latent implicit-H;
  * a neutral-skeleton key (charges zeroed) is reported alongside, since
    protonation is a declared latent in the heavy-only setting.

For every molecule the reference graph is built through the SAME function as the
prediction, so a build failure affects both identically and cannot silently
count against the model. The structural ``classify`` verdict is computed too and
any residual disagreement with the SMILES verdict is reported rather than hidden.

Usage
-----
  python scripts/evaluate_species_recovery.py \
      --per_edge results/per_edge_to50_valencefix.jsonl \
      --reference_sdf data/pdbbind_ccd_pose500.sdf \
      --output results/species_recovery_to50_valencefix.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per_edge", required=True)
    p.add_argument("--reference_sdf", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--allowed_atomic_nums", type=int, nargs="+",
                   default=[6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53, 83])
    p.add_argument("--show_disagreements", type=int, default=8)
    return p.parse_args()


class _Vec(list):
    """Minimal stand-in giving the .tolist() that _candidate_mol expects."""

    def tolist(self):
        return list(self)


def _graph_candidate(elems, graph):
    """Wrap a {(i,j): label} graph as a candidate dict for _candidate_mol.

    formal_charge is left at zero: the species key is charge-bearing where the
    kekulized build demands it (RDKit assigns charges during sanitization for a
    valid Lewis structure), and the neutral-skeleton key removes protonation
    entirely. Passing the decoder's own charges is unnecessary here because both
    reference and prediction are built the same way.
    """
    n = len(elems)
    edges = sorted(graph)
    labels = [graph[e] for e in edges]
    return {"elems": _Vec(elems),
            "formal_charge": _Vec([0] * n),
            "edges": edges,
            "edge_labels": _Vec(labels),
            "bonds": {e: graph[e] for e in edges if graph[e] > 0}}


def main():
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    args = parse_args()

    # evaluate_candidate_sets imports torch at module scope for the decoding
    # path, which this script never touches (it reads a per-edge dump). Install
    # the lightweight stub so the shared species_signature can be reused rather
    # than duplicated -- one build path is the whole point.
    try:
        import torch  # noqa: F401
    except ImportError:
        from scripts.benchmark_shared_harness_rules import _install_torch_stub
        _install_torch_stub()
    from scripts.evaluate_candidate_sets import species_signature
    from scripts.classify_candidate_ambiguity import classify

    per_edge = defaultdict(list)
    for line in Path(args.per_edge).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            per_edge[row["name"]].append(row)

    allowed = set(args.allowed_atomic_nums)
    total = 0
    species_ok = skeleton_ok = classify_ok = 0
    ref_build_fail = pred_build_fail = 0
    disagreements = []
    per_molecule = []

    for mol in Chem.SDMolSupplier(args.reference_sdf, removeHs=False):
        if mol is None or mol.GetNumConformers() == 0:
            continue
        mol = Chem.RemoveHs(mol, sanitize=True)
        if not {a.GetAtomicNum() for a in mol.GetAtoms()} <= allowed:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else None
        if name not in per_edge:
            continue
        total += 1

        elems = [a.GetAtomicNum() for a in mol.GetAtoms()]
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

        ref_key, ref_skel, ref_status = species_signature(
            mol, _graph_candidate(elems, truth), adjacency, return_status=True)
        pred_key, pred_skel, pred_status = species_signature(
            mol, _graph_candidate(elems, predicted), adjacency, return_status=True)

        ref_built = not ref_key.startswith("fallback::")
        pred_built = not pred_key.startswith("fallback::")
        ref_build_fail += not ref_built
        pred_build_fail += not pred_built

        smiles_species = ref_built and pred_built and ref_key == pred_key
        smiles_skeleton = ref_built and pred_built and ref_skel == pred_skel
        verdict, _ = classify(mol, truth, predicted, adjacency)
        classify_species = verdict in ("identical", "representation_equivalent")

        species_ok += smiles_species
        skeleton_ok += smiles_skeleton
        classify_ok += classify_species

        # With the build fixed, the two oracles should now agree. Record any
        # residual disagreement (where both molecules built) for inspection.
        if ref_built and pred_built and smiles_species != classify_species:
            disagreements.append({
                "name": name, "classify": verdict,
                "smiles_same": smiles_species,
                "ref": ref_key.replace("smiles::", ""),
                "pred": pred_key.replace("smiles::", ""),
            })
        per_molecule.append({
            "name": name,
            "smiles_species": bool(smiles_species),
            "smiles_skeleton": bool(smiles_skeleton),
            "classify_species": bool(classify_species),
            "ref_built": ref_built, "pred_built": pred_built,
        })

    print(f"molecules: {total}\n")
    print(f"  species recovery (canonical SMILES, protonation-aware)  {species_ok / total:.4f}"
          f"  ({species_ok}/{total})")
    print(f"  skeleton recovery (neutralised, latent charge/H)        {skeleton_ok / total:.4f}"
          f"  ({skeleton_ok}/{total})")
    print(f"  classify structural verdict                             {classify_ok / total:.4f}"
          f"  ({classify_ok}/{total})")
    print(f"\n  reference build failures: {ref_build_fail}   "
          f"prediction build failures: {pred_build_fail}")
    print(f"  oracle disagreements (both built): {len(disagreements)}")
    for entry in disagreements[:args.show_disagreements]:
        print(f"    {entry['name']}: classify={entry['classify']} "
              f"smiles_same={entry['smiles_same']}")
        print(f"       ref  {entry['ref']}")
        print(f"       pred {entry['pred']}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({
            "n_molecules": total,
            "species_recovery": species_ok / total,
            "skeleton_recovery": skeleton_ok / total,
            "classify_recovery": classify_ok / total,
            "reference_build_failures": ref_build_fail,
            "prediction_build_failures": pred_build_fail,
            "oracle_disagreements": disagreements,
            "per_molecule": per_molecule,
        }, indent=1), encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
