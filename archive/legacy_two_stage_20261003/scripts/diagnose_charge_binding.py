#!/usr/bin/env python3
"""Diagnose when a supplied net charge changes BondNet's decoded graph.

The heavy-only refiner conditions its neural representation on a molecular net
charge, while the MILP enforces that the latent per-atom formal charges sum to
that same value.  A whole-graph benchmark score alone cannot reveal whether a
different supplied charge changes the heavy-atom graph, only the latent atom
state assignment, or neither.  This script records that distinction without
using reference bonds or local reference charges during inference.

Example
-------
python scripts/diagnose_charge_binding.py \
  --reference_sdf data/pdbbind_ccd_pose500.sdf \
  --stage1_ckpt checkpoints/diverse80k_heavy_stage1/best.pt \
  --refiner_ckpt checkpoints/diverse80k_heavy_struct6_vector_hardneg_to50/best.pt \
  --state_catalogue_json results/training_state_catalogue.json \
  --output results/pdbbind_80k_charge_binding.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference_sdf", required=True)
    p.add_argument("--stage1_ckpt", required=True)
    p.add_argument("--refiner_ckpt", required=True)
    p.add_argument("--state_catalogue_json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allowed_atomic_nums", type=int, nargs="+",
                   default=[6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53, 83])
    p.add_argument("--device", default="auto")
    p.add_argument("--max_mols", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _signature(candidate):
    """Make model outputs comparable and JSON-safe; no reference is used here."""
    if not candidate:
        return None
    bonds = tuple(sorted((int(i), int(j), int(label))
                         for (i, j), label in candidate["bonds"].items()))
    charges = tuple(int(x) for x in candidate["formal_charge"].tolist())
    hydrogens = tuple(int(x) for x in candidate["implicit_h"].tolist())
    return {"bonds": bonds, "formal_charge": charges, "implicit_h": hydrogens,
            "objective": float(candidate["objective"])}


def _stratum(q):
    return "charged" if int(q) else "neutral"


def _counter_dict():
    return Counter({"n": 0, "decode_failed": 0, "same_graph_same_state": 0,
                    "same_graph_changed_state": 0, "changed_graph": 0})


def main():
    from rdkit import Chem, RDLogger
    from scripts.benchmark_templatefree_motifs import (
        bondnet_predict, coordinate_view, load_bondnet, load_state_catalogue,
    )

    cli = parse_args()
    RDLogger.DisableLog("rdApp.*")
    device = torch.device("cuda" if cli.device == "auto" and torch.cuda.is_available() else cli.device)
    inference_args = SimpleNamespace(
        stage1_ckpt=cli.stage1_ckpt, refiner_ckpt=cli.refiner_ckpt,
        candidate_radius=2.5, candidate_knn=0, max_neighbors=15,
        h_candidate_cutoff=1.4, bondnet_decode_mode="full_graph",
        heavy_latent_charge_h=True, use_heavy_atom_state_logits=False,
        aromatic_constraints=False, aromatic_bridge_repair=False,
        restrict_delocalized=False, state_catalogue=load_state_catalogue(cli.state_catalogue_json),
        require_kekulizable_candidates=False, single_stage=False,
    )
    stage1, loaded = load_bondnet(inference_args, device)
    refiner, train_args = loaded
    allowed = set(cli.allowed_atomic_nums)
    transitions = defaultdict(_counter_dict)
    rows, skipped = [], 0

    for source_index, mol in enumerate(Chem.SDMolSupplier(cli.reference_sdf, removeHs=False)):
        if mol is None or mol.GetNumConformers() == 0:
            continue
        mol = Chem.RemoveHs(mol, sanitize=True)
        if not {atom.GetAtomicNum() for atom in mol.GetAtoms()} <= allowed:
            skipped += 1
            continue
        q = int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
        elems, coord = coordinate_view(mol, 0.0, cli.seed + source_index * 1009)
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else str(source_index)
        supplied = {"reference_q": q, "forced_q0": 0,
                    "q_minus_1": q - 1, "q_plus_1": q + 1}
        outputs, elapsed = {}, {}
        for mode, supplied_q in supplied.items():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            candidates = bondnet_predict(elems, coord, supplied_q, stage1, refiner,
                                         train_args, inference_args, device, n_solutions=1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed[mode] = time.perf_counter() - started
            outputs[mode] = _signature(candidates[0] if candidates else None)

        oracle = outputs["reference_q"]
        comparisons = {}
        for mode, candidate in outputs.items():
            if mode == "reference_q":
                continue
            if oracle is None or candidate is None:
                outcome = "decode_failed"
            elif candidate["bonds"] != oracle["bonds"]:
                outcome = "changed_graph"
            elif (candidate["formal_charge"], candidate["implicit_h"]) != (
                    oracle["formal_charge"], oracle["implicit_h"]):
                outcome = "same_graph_changed_state"
            else:
                outcome = "same_graph_same_state"
            comparisons[mode] = outcome
            for key in ("all", _stratum(q)):
                transitions[f"{mode}:{key}"]["n"] += 1
                transitions[f"{mode}:{key}"][outcome] += 1

        rows.append({"name": name, "reference_q": q, "supplied_q": supplied,
                     "comparison_to_reference_q": comparisons,
                     "decode_seconds": elapsed,
                     "outputs": outputs})
        if len(rows) % 50 == 0:
            print(f"diagnosed {len(rows)} molecules", flush=True)
        if cli.max_mols is not None and len(rows) >= cli.max_mols:
            break

    def normalize(counter):
        n = counter["n"]
        return {key: int(value) if key in {"n", "decode_failed", "same_graph_same_state",
                                            "same_graph_changed_state", "changed_graph"} else value
                for key, value in counter.items()} | {
                    "graph_change_rate": counter["changed_graph"] / n if n else None,
                    "state_only_change_rate": counter["same_graph_changed_state"] / n if n else None,
                }

    report = {
        "task": "effect of supplied molecular net charge on heavy-only BondNet decoding",
        "inference_information": "Only (Z, R, supplied Q) enters inference; reference Q is used only to define perturbations.",
        "n_molecules": len(rows), "skipped_unsupported_elements": skipped,
        "stage1_ckpt": str(Path(cli.stage1_ckpt).resolve()),
        "refiner_ckpt": str(Path(cli.refiner_ckpt).resolve()),
        "state_catalogue_json": str(Path(cli.state_catalogue_json).resolve()),
        "summary": {key: normalize(value) for key, value in transitions.items()},
        "per_molecule": rows,
    }
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
