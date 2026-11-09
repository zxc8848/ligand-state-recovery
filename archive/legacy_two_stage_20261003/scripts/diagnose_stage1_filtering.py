#!/usr/bin/env python3
r"""Quantify how much Stage-1 actually filters the message (topology) graph.

The refiner uses TWO graphs:
  * message_ei : the message-passing / topology graph, gated by Stage-1's
    connectivity probability (conn_prob >= conn_threshold, widened by
    retain_threshold). This is what the "two-stage" claim rests on.
  * readout_ei : every candidate heavy-heavy edge is typed (including a nobond
    class), so the refiner can delete Stage-1 false positives and restore false
    negatives inside the candidate graph. Being over all candidates is by design.

This script reports, per molecule and in aggregate:
  |candidate HH edges|  -- the distance-cutoff candidate graph
  |message @0.5|        -- Stage-1 "predicts bonded"
  |message @retain|     -- the actually-used message graph (retain_threshold)
  |true bonds|          -- reference heavy-heavy bonds

The key ratios:
  message@retain / candidate  -> if ~1.0, Stage-1 is NOT filtering (the concern);
                                  if well below 1.0, Stage-1 prunes the topology.
  message@retain / true        -> how much wider than the true graph the topology is.

Run on a machine with torch + the checkpoints.

Usage:
  python scripts/diagnose_stage1_filtering.py \
      --reference_sdf data/pdbbind_ccd_pose500.sdf \
      --stage1_ckpt checkpoints/diverse80k_heavy_stage1/best.pt \
      --refiner_ckpt checkpoints/diverse80k_heavy_struct6_vector_hardneg_to50/best.pt \
      --retain_threshold 0.15 --conn_threshold 0.5 \
      --output results/stage1_filtering_diagnosis.json
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
    p.add_argument("--conn_threshold", type=float, default=0.5)
    p.add_argument("--retain_threshold", type=float, default=0.15)
    p.add_argument("--candidate_radius", type=float, default=2.5)
    p.add_argument("--candidate_knn", type=int, default=0)
    p.add_argument("--max_neighbors", type=int, default=15)
    p.add_argument("--allowed_atomic_nums", type=int, nargs="+",
                   default=[6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53, 83])
    p.add_argument("--max_mols", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    import numpy as np
    import torch
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    args = parse_args()

    from scripts.benchmark_templatefree_motifs import coordinate_view, heavy_graph, load_bondnet
    from bondnet.integration.molcraft import (
        build_refiner_inference_batch, make_coordinate_batch)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else args.device)
    args.input_mode = "heavy_only"
    args.methods = ["bondnet"]
    args.h_candidate_cutoff = 1.4
    stage1, (refiner, train_args) = load_bondnet(args, device)
    # Force the thresholds under test onto the bridge config.
    train_args.conn_threshold = args.conn_threshold
    train_args.retain_threshold = args.retain_threshold

    allowed = set(args.allowed_atomic_nums)
    rows = []
    for mol in Chem.SDMolSupplier(args.reference_sdf, removeHs=False):
        if mol is None or mol.GetNumConformers() == 0:
            continue
        mol = Chem.RemoveHs(mol, sanitize=True)
        if not {a.GetAtomicNum() for a in mol.GetAtoms()} <= allowed:
            continue
        if args.max_mols is not None and len(rows) >= args.max_mols:
            break

        elems, coord = coordinate_view(mol, 0.0, 0)
        total_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
        from bondnet.integration.molcraft import all_atom_candidate_edges
        candidates = all_atom_candidate_edges(elems, coord, args.candidate_radius,
                                              args.candidate_knn, args.max_neighbors,
                                              args.h_candidate_cutoff)
        batch = make_coordinate_batch(elems.to(device), coord.to(device), total_charge,
                                      args.candidate_radius, args.candidate_knn,
                                      args.max_neighbors,
                                      candidate_edge_index=candidates.to(device))

        with torch.no_grad():
            _, _, edge_feat = stage1.backbone(
                batch["elems"], batch["coord"], batch["edge_index"],
                batch["edge_diff"], batch["edge_dist"])
            conn_prob = torch.sigmoid(stage1.stage1(edge_feat))
        ei = batch["edge_index"]
        e = batch["elems"]
        hh = (e[ei[:, 0]] != 1) & (e[ei[:, 1]] != 1)
        tm = batch.get("train_edge_mask")
        tm = tm.bool() if tm is not None else torch.ones(ei.shape[0], dtype=torch.bool, device=ei.device)
        base = tm & hh
        # undirected edge counts
        def undirected(mask):
            sel = ei[mask].tolist()
            return len({tuple(sorted(map(int, p))) for p in sel})
        cand = undirected(base)
        m05 = undirected(base & (conn_prob >= args.conn_threshold))
        mret = undirected(base & (conn_prob >= args.retain_threshold))
        truth = len(heavy_graph(mol))
        rows.append({"name": mol.GetProp("_Name") if mol.HasProp("_Name") else str(len(rows)),
                     "candidate": cand, "message_at_0.5": m05,
                     "message_at_retain": mret, "true_bonds": truth})

    def agg(key):
        return float(np.mean([r[key] for r in rows]))

    summary = {
        "n_molecules": len(rows),
        "conn_threshold": args.conn_threshold,
        "retain_threshold": args.retain_threshold,
        "mean_candidate_edges": agg("candidate"),
        "mean_message_at_0.5": agg("message_at_0.5"),
        "mean_message_at_retain": agg("message_at_retain"),
        "mean_true_bonds": agg("true_bonds"),
        "message_retain_over_candidate": agg("message_at_retain") / agg("candidate"),
        "message_retain_over_true": agg("message_at_retain") / agg("true_bonds"),
        "message_0.5_over_true": agg("message_at_0.5") / agg("true_bonds"),
        "per_molecule": rows,
    }
    print(f"n = {summary['n_molecules']}")
    print(f"  mean candidate HH edges      {summary['mean_candidate_edges']:.1f}")
    print(f"  mean message @0.5            {summary['mean_message_at_0.5']:.1f}")
    print(f"  mean message @retain={args.retain_threshold}   {summary['mean_message_at_retain']:.1f}")
    print(f"  mean true bonds              {summary['mean_true_bonds']:.1f}")
    print(f"  message@retain / candidate   {summary['message_retain_over_candidate']:.3f}"
          "   (near 1.0 => Stage-1 barely filters)")
    print(f"  message@retain / true        {summary['message_retain_over_true']:.3f}")
    print(f"  message@0.5   / true         {summary['message_0.5_over_true']:.3f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
