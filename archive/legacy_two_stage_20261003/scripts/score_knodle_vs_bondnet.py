#!/usr/bin/env python3
r"""Table 4: score Knodle and BondNet on the identical 488 poses, one metric set.

Both methods are scored with the SAME method-neutral functions:
  strict     -- exact heavy-atom bond-order graph match
  resonance  -- resonance_align_terminal_oxy(elements=(6,7,15,16)); accepts any
                delocalized terminal-oxy encoding (BondNet class-5, Knodle
                aromatic, CCD {1,2} permutation) uniformly for both methods
  species    -- STRICT canonical-SMILES species key
                (evaluate_candidate_sets.species_signature). A graph that cannot
                be sanitized counts as NOT recovered (no classify() fallback), so
                this column matches evaluate_species_recovery.py exactly and is a
                single, well-defined oracle rather than a canonical/classify hybrid.

BondNet predictions are read from its per-edge dump; Knodle predictions from the
normalized JSON produced by p0d_knodle_compare.py adapt.

Usage:
  python scripts/score_knodle_vs_bondnet.py \
      --knodle_pred results/p0c/preds_knodle_pdbbind488.json \
      --reference_sdf data/pdbbind_ccd_pose500.sdf \
      --bondnet_per_edge results/per_edge_to50_valencefix.jsonl \
      --output results/knodle_vs_bondnet_488.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_ORDER = {"ar": 4, 1: 1, 2: 2, 3: 3, "1": 1, "2": 2, "3": 3}


class _Vec(list):
    def tolist(self):
        return list(self)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--knodle_pred", required=True)
    p.add_argument("--reference_sdf", required=True)
    p.add_argument("--bondnet_per_edge", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--count_missing_as_failure", action="store_true",
                   help="Score a molecule for which Knodle produced no output as an "
                        "incorrect prediction on every metric, instead of aborting on "
                        "the molecule-set equality check. This implements the "
                        "manuscript's declared accounting -- a tool's own failure to "
                        "return a sanitized molecule counts against it -- and matters "
                        "under coordinate noise, where a comparator may fail outright.")
    p.add_argument("--allowed_atomic_nums", type=int, nargs="+",
                   default=[6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53, 83])
    return p.parse_args()


def _cand(zs, graph):
    edges = sorted(graph)
    return {"elems": _Vec(zs), "formal_charge": _Vec([0] * len(zs)),
            "edges": edges, "edge_labels": _Vec([graph[e] for e in edges]),
            "bonds": {e: graph[e] for e in edges if graph[e] > 0}}


def _score_one(mol, truth, pred, heavy_graph, resonance_align_terminal_oxy,
               species_signature):
    zs = [a.GetAtomicNum() for a in mol.GetAtoms()]
    adj = defaultdict(set)
    for i, j in truth:
        adj[i].add(j)
        adj[j].add(i)
    strict = all(pred.get(p, 0) == truth.get(p, 0) for p in set(truth) | set(pred))
    aligned = resonance_align_terminal_oxy(pred, truth, mol, elements=(6, 7, 15, 16))
    resonance = all(aligned.get(p, 0) == truth.get(p, 0) for p in set(truth) | set(aligned))
    rk = species_signature(mol, _cand(zs, truth), adj, return_status=True)
    pk = species_signature(mol, _cand(zs, pred), adj, return_status=True)
    # STRICT canonical-SMILES species: both graphs must sanitize AND their
    # canonical species keys must match. An unbuildable graph is a miss (no
    # classify() fallback), so this equals evaluate_species_recovery.py.
    if not rk[0].startswith("fallback") and not pk[0].startswith("fallback"):
        species = rk[0] == pk[0]
    else:
        species = False
    return int(strict), int(resonance), int(species)


def _wilson_interval(successes, n, z=1.959963984540054):
    """Two-sided Wilson interval for a binomial proportion."""
    if n == 0:
        return [None, None]
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return [centre - half, centre + half]


def _exact_mcnemar(a_only, b_only):
    """Two-sided exact McNemar p-value without a SciPy dependency."""
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(discordant, i) for i in range(k + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def main():
    args = parse_args()
    try:
        import torch  # noqa: F401
    except ImportError:
        from scripts.benchmark_shared_harness_rules import _install_torch_stub
        _install_torch_stub()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from scripts.benchmark_templatefree_motifs import heavy_graph, resonance_align_terminal_oxy
    from scripts.evaluate_candidate_sets import species_signature
    from scripts.p0c_yuelbond_compare import _read_valid_molecules

    allowed = set(args.allowed_atomic_nums)

    # ---- Knodle ----
    preds = json.loads(Path(args.knodle_pred).read_text(encoding="utf-8"))
    preds.pop("__meta__", None)
    mols = _read_valid_molecules(args.reference_sdf)
    kn = {"n": 0, "strict": 0, "resonance": 0, "species": 0}
    kn_per_molecule = {}
    for idx_str, bonds in preds.items():
        mol = Chem.RemoveHs(mols[int(idx_str)], sanitize=True)
        if not {a.GetAtomicNum() for a in mol.GetAtoms()} <= allowed:
            continue
        truth = heavy_graph(mol)
        pred = {tuple(sorted((int(i), int(j)))): _ORDER.get(o, 1) for i, j, o in bonds}
        s, r, sp = _score_one(mol, truth, pred, heavy_graph,
                              resonance_align_terminal_oxy, species_signature)
        kn["n"] += 1
        kn["strict"] += s
        kn["resonance"] += r
        kn["species"] += sp
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else str(idx_str)
        kn_per_molecule[name] = {"strict": s, "resonance": r, "species": sp}

    # ---- BondNet (from per-edge dump) ----
    rows = [json.loads(l) for l in Path(args.bondnet_per_edge).read_text(encoding="utf-8").splitlines() if l.strip()]
    by = defaultdict(dict)
    for row in rows:
        by[row["name"]][(row["i"], row["j"])] = row
    bn = {"n": 0, "strict": 0, "resonance": 0, "species": 0}
    bn_per_molecule = {}
    for mol in Chem.SDMolSupplier(args.reference_sdf, removeHs=False):
        if mol is None or mol.GetNumConformers() == 0:
            continue
        mol = Chem.RemoveHs(mol, sanitize=True)
        if not {a.GetAtomicNum() for a in mol.GetAtoms()} <= allowed:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else None
        if name not in by:
            continue
        truth = heavy_graph(mol)
        pred = {k: v["pred"] for k, v in by[name].items() if v["pred"] > 0}
        s, r, sp = _score_one(mol, truth, pred, heavy_graph,
                              resonance_align_terminal_oxy, species_signature)
        bn["n"] += 1
        bn["strict"] += s
        bn["resonance"] += r
        bn["species"] += sp
        bn_per_molecule[name] = {"strict": s, "resonance": r, "species": sp}

    def rates(d):
        n = d["n"]
        out = {k: (d[k] / n if n else None) for k in ("strict", "resonance", "species")}
        out |= {"n": n}
        out["wilson_ci95"] = {k: _wilson_interval(d[k], n) for k in ("strict", "resonance", "species")}
        return out

    missing_knodle = sorted(set(bn_per_molecule) - set(kn_per_molecule))
    if args.count_missing_as_failure:
        for name in missing_knodle:
            kn_per_molecule[name] = {"strict": 0, "resonance": 0, "species": 0}
            kn["n"] += 1

    common_names = sorted(set(kn_per_molecule) & set(bn_per_molecule))
    if len(common_names) != kn["n"] or len(common_names) != bn["n"]:
        raise RuntimeError(
            "Knodle and BondNet predictions do not cover the same molecule set: "
            f"Knodle={kn['n']}, BondNet={bn['n']}, shared={len(common_names)}"
        )
    paired_tests = {}
    for metric in ("strict", "resonance", "species"):
        bn_only = sum(
            bn_per_molecule[name][metric] and not kn_per_molecule[name][metric]
            for name in common_names
        )
        kn_only = sum(
            kn_per_molecule[name][metric] and not bn_per_molecule[name][metric]
            for name in common_names
        )
        paired_tests[metric] = {
            "n": len(common_names),
            "bondnet_only": int(bn_only),
            "knodle_only": int(kn_only),
            "mcnemar_exact_p": _exact_mcnemar(int(bn_only), int(kn_only)),
        }

    payload = {
        "note": "Knodle vs BondNet on identical leakage-free PDBBind poses, "
                "method-neutral scoring (resonance accepts class-5, aromatic, and "
                "{1,2}-permutation encodings of delocalized terminal-oxy groups).",
        "knodle_input": "(Z,R) only; same-distribution to PDBBindCN training",
        "bondnet_input": "(Z,R,Q); cross-domain from Diverse-80k",
        "bondnet": rates(bn),
        "knodle": rates(kn),
        "knodle_no_output": {
            "n": len(missing_knodle),
            "counted_as_failure": bool(args.count_missing_as_failure),
            "names": missing_knodle[:50],
        },
        "paired_tests": paired_tests,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=1), encoding="utf-8")

    for name, d in (("BondNet", payload["bondnet"]), ("Knodle", payload["knodle"])):
        print(f"{name:<9} n={d['n']}  strict={d['strict']:.4f}  "
              f"resonance={d['resonance']:.4f}  species={d['species']:.4f}")
    if args.output:
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
