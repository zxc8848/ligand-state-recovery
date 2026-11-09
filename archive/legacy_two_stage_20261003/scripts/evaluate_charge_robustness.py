#!/usr/bin/env python3
"""Evaluate missing, incorrect, and jointly inferred molecular net charge.

Reference bonds and charges are used only for scoring.  ``infer`` enumerates a
small, declared charge range and selects the highest joint decoder objective;
it is meaningful only for a checkpoint that does not condition its network on
the supplied total charge and that has supervised atom-state heads.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference_sdf", required=True)
    p.add_argument("--stage1_ckpt", required=True)
    p.add_argument("--refiner_ckpt", required=True)
    p.add_argument("--state_catalogue_json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allowed_atomic_nums", type=int, nargs="+",
                   default=[6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53, 83])
    p.add_argument("--charge_candidates", type=int, nargs="+", default=[-2, -1, 0, 1, 2])
    p.add_argument("--infer_charge", action="store_true")
    p.add_argument("--perturb_charge", action="store_true")
    p.add_argument("--use_heavy_atom_state_logits", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def empty_cell():
    return {"n": 0, "strict": 0, "resonance": 0, "true_edges": 0,
            "typed_true_edges": 0, "false_positive_edges": 0}


def update(cell, truth, predicted, resonance_predicted):
    cell["n"] += 1
    cell["strict"] += int(all(predicted.get(edge, 0) == truth.get(edge, 0)
                              for edge in set(truth) | set(predicted)))
    cell["resonance"] += int(all(resonance_predicted.get(edge, 0) == truth.get(edge, 0)
                                 for edge in set(truth) | set(resonance_predicted)))
    cell["true_edges"] += len(truth)
    cell["typed_true_edges"] += sum(predicted.get(edge, 0) == label
                                    for edge, label in truth.items())
    cell["false_positive_edges"] += sum(edge not in truth for edge in predicted)


def summarize(cell):
    n = cell["n"]
    return {
        "n": n,
        "strict_graph_exact_rate": cell["strict"] / n if n else None,
        "resonance_aware_exact_rate": cell["resonance"] / n if n else None,
        "true_edge_typed_recall": (cell["typed_true_edges"] / cell["true_edges"]
                                   if cell["true_edges"] else None),
        "mean_false_positive_edges": cell["false_positive_edges"] / n if n else None,
    }


def main():
    from rdkit import Chem, RDLogger
    from scripts.benchmark_templatefree_motifs import (
        bondnet_predict, coordinate_view, heavy_graph, load_bondnet,
        load_state_catalogue, resonance_align_terminal_oxy,
    )

    RDLogger.DisableLog("rdApp.*")
    cli = parse_args()
    device = torch.device("cuda" if cli.device == "auto" and torch.cuda.is_available()
                          else cli.device)
    inference_args = SimpleNamespace(
        stage1_ckpt=cli.stage1_ckpt,
        refiner_ckpt=cli.refiner_ckpt,
        candidate_radius=2.5,
        candidate_knn=0,
        max_neighbors=15,
        h_candidate_cutoff=1.4,
        bondnet_decode_mode="full_graph",
        heavy_latent_charge_h=False,
        use_heavy_atom_state_logits=cli.use_heavy_atom_state_logits,
        aromatic_constraints=False,
        aromatic_bridge_repair=False,
        restrict_delocalized=False,
        state_catalogue=load_state_catalogue(cli.state_catalogue_json),
        require_kekulizable_candidates=False,
        single_stage=False,
    )
    stage1, loaded = load_bondnet(inference_args, device)
    refiner, train_args = loaded
    train_args.single_stage = bool(getattr(train_args, "single_stage", False))
    network_conditions_on_q = bool(getattr(train_args, "total_charge_condition", False))
    if cli.infer_charge and network_conditions_on_q:
        raise ValueError("cross-charge objectives are not comparable for a Q-conditioned network")
    if cli.infer_charge and not cli.use_heavy_atom_state_logits:
        raise ValueError("charge inference requires --use_heavy_atom_state_logits")

    modes = ["oracle_q", "forced_q0"]
    if cli.perturb_charge:
        modes += ["q_minus_1", "q_plus_1"]
    if cli.infer_charge:
        modes.append("inferred_q")
    metrics = {mode: defaultdict(empty_cell) for mode in modes}
    per_molecule = []
    allowed = set(cli.allowed_atomic_nums)
    accepted = skipped = 0

    for source_index, mol in enumerate(Chem.SDMolSupplier(cli.reference_sdf, removeHs=False)):
        if mol is None or mol.GetNumConformers() == 0:
            continue
        mol = Chem.RemoveHs(mol, sanitize=True)
        if not {atom.GetAtomicNum() for atom in mol.GetAtoms()} <= allowed:
            skipped += 1
            continue
        truth = heavy_graph(mol)
        true_q = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        elems, coord = coordinate_view(mol, 0.0, cli.seed + source_index * 1009)
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else str(source_index)
        strata = ["all", "neutral" if true_q == 0 else "charged", f"q={true_q:+d}"]
        assignments = {"oracle_q": true_q, "forced_q0": 0}
        if cli.perturb_charge:
            assignments["q_minus_1"] = true_q - 1
            assignments["q_plus_1"] = true_q + 1

        row = {"name": name, "true_q": true_q, "modes": {}}
        for mode, supplied_q in assignments.items():
            predicted = bondnet_predict(
                elems, coord, supplied_q, stage1, refiner, train_args,
                inference_args, device,
            )
            resonance = resonance_align_terminal_oxy(
                predicted, truth, mol, elements=(6, 7, 15, 16)
            )
            for stratum in strata:
                update(metrics[mode][stratum], truth, predicted, resonance)
            row["modes"][mode] = {
                "supplied_q": supplied_q,
                "strict": int(all(predicted.get(edge, 0) == truth.get(edge, 0)
                                  for edge in set(truth) | set(predicted))),
                "resonance": int(all(resonance.get(edge, 0) == truth.get(edge, 0)
                                     for edge in set(truth) | set(resonance))),
            }

        if cli.infer_charge:
            proposals = []
            for candidate_q in cli.charge_candidates:
                candidates = bondnet_predict(
                    elems, coord, candidate_q, stage1, refiner, train_args,
                    inference_args, device, n_solutions=1,
                )
                if candidates:
                    proposals.append((float(candidates[0]["objective"]), candidate_q,
                                      candidates[0]["bonds"]))
            if not proposals:
                inferred_q, predicted, objective, margin = None, {}, None, None
            else:
                proposals.sort(key=lambda item: item[0], reverse=True)
                objective, inferred_q, predicted = proposals[0]
                margin = (objective - proposals[1][0]) if len(proposals) > 1 else None
            resonance = resonance_align_terminal_oxy(
                predicted, truth, mol, elements=(6, 7, 15, 16)
            )
            for stratum in strata:
                update(metrics["inferred_q"][stratum], truth, predicted, resonance)
            row["modes"]["inferred_q"] = {
                "predicted_q": inferred_q,
                "q_correct": int(inferred_q == true_q),
                "objective": objective,
                "objective_margin": margin,
                "strict": int(all(predicted.get(edge, 0) == truth.get(edge, 0)
                                  for edge in set(truth) | set(predicted))),
                "resonance": int(all(resonance.get(edge, 0) == truth.get(edge, 0)
                                     for edge in set(truth) | set(resonance))),
            }
        per_molecule.append(row)
        accepted += 1
        if accepted % 100 == 0:
            print(f"evaluated {accepted}", flush=True)

    charge_accuracy = None
    if cli.infer_charge:
        def q_acc(rows):
            values = [r["modes"]["inferred_q"]["q_correct"] for r in rows]
            return {"n": len(values), "accuracy": sum(values) / len(values) if values else None}
        charge_accuracy = {
            "all": q_acc(per_molecule),
            "neutral": q_acc([r for r in per_molecule if r["true_q"] == 0]),
            "charged": q_acc([r for r in per_molecule if r["true_q"] != 0]),
        }

    report = {
        "task": "heavy-atom graph recovery under known, missing, incorrect, or inferred net charge",
        "resonance_metric": "C/N/P/S terminal-oxygen plus aromatic equivalence",
        "reference_sdf": str(Path(cli.reference_sdf).resolve()),
        "n_molecules": accepted,
        "skipped_unsupported_elements": skipped,
        "stage1_ckpt": str(Path(cli.stage1_ckpt).resolve()),
        "refiner_ckpt": str(Path(cli.refiner_ckpt).resolve()),
        "state_catalogue_json": str(Path(cli.state_catalogue_json).resolve()),
        "network_conditions_on_q": network_conditions_on_q,
        "uses_atom_state_logits": bool(cli.use_heavy_atom_state_logits),
        "charge_candidates": cli.charge_candidates if cli.infer_charge else None,
        "selection_rule": ("maximum joint decoder objective over the declared charge candidates"
                           if cli.infer_charge else None),
        "metrics": {mode: {stratum: summarize(cell) for stratum, cell in strata.items()}
                    for mode, strata in metrics.items()},
        "charge_inference_accuracy": charge_accuracy,
        "per_molecule": per_molecule,
    }
    Path(cli.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": report["metrics"],
                      "charge_inference_accuracy": charge_accuracy}, indent=2))


if __name__ == "__main__":
    main()
