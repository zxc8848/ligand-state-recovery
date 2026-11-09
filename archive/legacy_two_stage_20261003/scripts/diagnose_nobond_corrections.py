#!/usr/bin/env python3
r"""Quantify what the Stage-2 ``nobond`` class actually does to Stage-1's topology,
on the 488 real PDBBind poses (same data/decoder as Table 3).

The two-stage claim is that Stage-2 -- because its readout covers *every* candidate
heavy--heavy pair and includes a ``nobond`` class -- can delete Stage-1 false
positives and restore Stage-1 false negatives, rather than being locked to the
Stage-1 message graph. This measures that correction directly.

Per molecule, three undirected edge sets over the candidate graph:
  S1 = Stage-1 message edges (conn_prob >= retain_threshold)  -- "Stage-1 says bond"
  S2 = Stage-2 final bonds   (decoded label != nobond)        -- "Stage-2 says bond"
  T  = reference heavy--heavy bonds                           -- ground truth
Reported per molecule and in aggregate:
  deleted        = S1 \ S2   (Stage-2 turned a Stage-1 bond into nobond)
    deleted_correct = deleted \ T   (fixed a Stage-1 FALSE POSITIVE)   <- nobond win
    deleted_wrong   = deleted & T   (removed a TRUE bond)              <- nobond cost
  added          = S2 \ S1   (Stage-2 added a bond Stage-1 had dropped)
    added_correct   = added & T     (restored a Stage-1 FALSE NEGATIVE)
    added_wrong     = added \ T     (invented a bond)
  stage1_fp/fn = S1 vs T ; final_fp/fn = S2 vs T
Headline: fraction of Stage-1 false positives the nobond class removes, and how it
grows with pose noise. Run across noise levels.

Requires torch + the checkpoints (GPU). Uses the same molcraft inference path and
joint_decode as scripts/benchmark_templatefree_motifs.py, so it is apples-to-apples
with Table 3 (heavy-only, latent charge/H).

Usage:
  python scripts/diagnose_nobond_corrections.py \
      --reference_sdf data/pdbbind_ccd_pose500.sdf \
      --stage1_ckpt checkpoints/diverse80k_heavy_stage1/best.pt \
      --refiner_ckpt checkpoints/diverse80k_heavy_struct6_vector_hardneg_to50/best.pt \
      --allowed_atomic_nums 6 7 8 9 14 15 16 17 33 35 53 83 \
      --noise 0.0 0.05 0.10 --output results/nobond_corrections.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference_sdf", required=True)
    p.add_argument("--stage1_ckpt", required=True)
    p.add_argument("--refiner_ckpt", required=True)
    p.add_argument("--noise", type=float, nargs="+", default=(0.0, 0.05, 0.10))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--conn_threshold", type=float, default=0.5)
    p.add_argument("--retain_threshold", type=float, default=0.15)
    p.add_argument("--candidate_radius", type=float, default=2.5)
    p.add_argument("--candidate_knn", type=int, default=0)
    p.add_argument("--max_neighbors", type=int, default=15)
    p.add_argument("--h_candidate_cutoff", type=float, default=1.4)
    p.add_argument("--restrict_delocalized", action="store_true")
    p.add_argument("--aromatic_constraints", action="store_true")
    p.add_argument("--aromatic_bridge_repair", action="store_true")
    p.add_argument("--allowed_atomic_nums", type=int, nargs="+",
                   default=[6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53, 83])
    p.add_argument("--max_mols", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None)
    return p.parse_args()


def _undirected(rows):
    out = set()
    for i, j in rows:
        a, b = int(i), int(j)
        out.add((a, b) if a < b else (b, a))
    return out


_KEYS = ("candidate", "s1", "s2", "true", "deleted", "deleted_correct",
         "deleted_wrong", "added", "added_correct", "added_wrong",
         "stage1_fp", "stage1_fn", "final_fp", "final_fn")


def _summary(agg, n_mol, n_unsolved):
    n = max(n_mol, 1)
    per = {k: round(agg[k] / n, 4) for k in _KEYS}
    s1fp = agg["stage1_fp"] or 1
    s1fn = agg["stage1_fn"] or 1
    return {
        "n_molecules": n_mol,
        "n_unsolved_decode": n_unsolved,
        "per_molecule": per,
        "totals": {k: agg[k] for k in _KEYS},
        "nobond_removed_stage1_fp_fraction": round(agg["deleted_correct"] / s1fp, 4),
        "restored_stage1_fn_fraction": round(agg["added_correct"] / s1fn, 4),
        "true_bonds_wrongly_deleted_per_mol": round(agg["deleted_wrong"] / n, 4),
    }


def main():
    import torch
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device)

    from scripts.benchmark_templatefree_motifs import coordinate_view, heavy_graph, load_bondnet
    from scripts.run_molcraft_bondnet import undirected_labels
    from bondnet.integration.molcraft import (
        all_atom_candidate_edges, build_refiner_inference_batch,
        forward_coordinate_refiner, make_coordinate_batch)
    from bondnet.inference.chemical_repair import joint_decode

    args.input_mode = "heavy_only"
    args.methods = ["bondnet"]
    args.bondnet_decode_mode = "full_graph"
    args.heavy_latent_charge_h = True
    stage1, (refiner, train_args) = load_bondnet(args, device)
    train_args.conn_threshold = args.conn_threshold
    train_args.retain_threshold = args.retain_threshold

    allowed = set(args.allowed_atomic_nums)
    mols = []
    for mol in Chem.SDMolSupplier(args.reference_sdf, removeHs=False):
        if mol is None or mol.GetNumConformers() == 0:
            continue
        mol = Chem.RemoveHs(mol, sanitize=True)
        if not {a.GetAtomicNum() for a in mol.GetAtoms()} <= allowed:
            continue
        mols.append(mol)
        if args.max_mols is not None and len(mols) >= args.max_mols:
            break

    report = {"refiner": str(Path(args.refiner_ckpt).resolve()),
              "n_molecules": len(mols), "noise": {}}

    for sigma in args.noise:
        agg = {k: 0 for k in _KEYS}
        n_mol = n_unsolved = 0
        for idx, mol in enumerate(mols):
            elems, coord = coordinate_view(mol, sigma, args.seed + idx)
            total_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
            candidates = all_atom_candidate_edges(
                elems, coord, args.candidate_radius, args.candidate_knn,
                args.max_neighbors, args.h_candidate_cutoff)
            batch = make_coordinate_batch(
                elems.to(device), coord.to(device), total_charge,
                args.candidate_radius, args.candidate_knn, args.max_neighbors,
                candidate_edge_index=candidates.to(device))
            with torch.no_grad():
                built = build_refiner_inference_batch(
                    stage1, batch, train_args, refiner_heavy_only=False)
                logits, aux = forward_coordinate_refiner(
                    refiner, batch, built, return_aux=True)
            readout_ei = built["readout_ei"]
            decoded = joint_decode(
                logits, readout_ei, batch["elems"], total_charge,
                charge_logits=None, implicit_h_logits=None, explicit_h=False,
                aromatic_constraints=args.aromatic_constraints,
                aromatic_bridge_repair=args.aromatic_bridge_repair,
                restrict_delocalized=args.restrict_delocalized)
            n_mol += 1
            if not decoded.solved:
                n_unsolved += 1
                continue
            pairs, labels = undirected_labels(readout_ei, decoded.labels)
            S1 = _undirected(built["message_ei"].detach().cpu().tolist())
            S2 = {p for p, lab in zip(pairs, labels) if lab > 0}
            truth = heavy_graph(mol)
            T = _undirected(list(truth.keys()) if isinstance(truth, dict) else list(truth))
            cand = _undirected(readout_ei.detach().cpu().tolist())

            deleted, added = S1 - S2, S2 - S1
            agg["candidate"] += len(cand)
            agg["s1"] += len(S1)
            agg["s2"] += len(S2)
            agg["true"] += len(T)
            agg["deleted"] += len(deleted)
            agg["deleted_correct"] += len(deleted - T)
            agg["deleted_wrong"] += len(deleted & T)
            agg["added"] += len(added)
            agg["added_correct"] += len(added & T)
            agg["added_wrong"] += len(added - T)
            agg["stage1_fp"] += len(S1 - T)
            agg["stage1_fn"] += len(T - S1)
            agg["final_fp"] += len(S2 - T)
            agg["final_fn"] += len(T - S2)

        report["noise"][str(sigma)] = _summary(agg, n_mol, n_unsolved)
        s = report["noise"][str(sigma)]
        print(f"noise={sigma}: Stage-1 FP/mol={s['per_molecule']['stage1_fp']}, "
              f"nobond removed {100*s['nobond_removed_stage1_fp_fraction']:.1f}% of them; "
              f"final FP/mol={s['per_molecule']['final_fp']}; "
              f"restored {100*s['restored_stage1_fn_fraction']:.1f}% of Stage-1 FN; "
              f"true bonds wrongly deleted/mol={s['true_bonds_wrongly_deleted_per_mol']}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
