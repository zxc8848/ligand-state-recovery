"""Evaluate a six-class structured refiner with and without global decoding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bondnet.data.dataset import CachedBondNetDataset, collate_stage2_fn
from bondnet.data.noise_augment import GaussianNoiseAugment
from bondnet.inference.structured_decoder import decode_delocalized_oxy_groups, decode_valence_constrained
from bondnet.model.bond_type_gnn import BondTypeGNN
from scripts.train_bond_refiner import (
    Metrics, build_refiner_batch, forward_refiner, load_stage1, to_device,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage1_ckpt", required=True)
    p.add_argument("--refiner_ckpt", required=True)
    p.add_argument("--cache_path", required=True)
    p.add_argument("--max_mols", type=int, default=2000)
    p.add_argument("--noise", type=float, nargs="+", default=(0.0, 0.05, 0.10))
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--candidate_policy", choices=("static", "uncertainty"), default=None,
                   help="Override the Stage-2 message-graph policy saved in the checkpoint.")
    p.add_argument("--uncertain_edges_per_atom", type=int, default=1)
    p.add_argument("--group_decode", action="store_true",
                   help="Apply prediction-only center--two-O delocalization decoding.")
    p.add_argument("--group_threshold", type=float, default=0.20)
    p.add_argument("--oracle_atom_state", action="store_true",
                   help="Diagnostic only: give the decoder reference H counts/formal charges.")
    p.add_argument("--output", default=None)
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed for noise injection; enables multi-seed evaluation bands.")
    p.add_argument("--min_heavy", type=int, default=0,
                   help="Keep only molecules with >= this many heavy atoms (size bucketing).")
    p.add_argument("--max_heavy", type=int, default=10**9,
                   help="Keep only molecules with <= this many heavy atoms (size bucketing).")
    return p.parse_args()


def load_refiner(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = checkpoint["args"]
    if not saved.get("structured_six_class", False):
        raise ValueError("--refiner_ckpt must be trained with --structured_six_class")
    model = BondTypeGNN(
        hidden_size=128, num_layers=saved.get("num_layers", 4),
        edge_embedding_size=saved.get("edge_embedding_size", 32),
        bond_cutoff=saved.get("bond_cutoff", 3.0), num_bond_types=6,
        dropout=0.0, vector_norm_limit=5, readout_extra_dim=2,
        readout_feature_mode=saved.get("readout_feature_mode", "none"),
        predict_formal_charge=saved.get("charge_aux_weight", 0.0) > 0,
        predict_delocalized_center=saved.get("delocalized_group_aux_weight", 0.0) > 0,
        predict_implicit_h=saved.get("implicit_h_aux_weight", 0.0) > 0,
        predict_hybridization=saved.get("hybridization_aux_weight", 0.0) > 0,
        total_charge_condition=saved.get("total_charge_condition", False),
        backbone=saved.get("backbone", "painn"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, argparse.Namespace(**saved)


def decode_batch(logits, readout_ei, elems, counts, charge_logits=None,
                 group_decode=False, group_threshold=0.20, implicit_h_logits=None):
    decoded = torch.empty(logits.shape[0], dtype=torch.long, device=logits.device)
    start = 0
    for count in counts.detach().cpu().tolist():
        end = start + int(count)
        mask = (readout_ei[:, 0] >= start) & (readout_ei[:, 0] < end)
        local_ei = readout_ei[mask] - start
        local_charge = charge_logits[start:end] if charge_logits is not None else None
        local_h = implicit_h_logits[start:end] if implicit_h_logits is not None else None
        local_pred = decode_valence_constrained(
            logits[mask], local_ei, elems[start:end], charge_logits=local_charge,
            implicit_h_logits=local_h,
        )
        if group_decode:
            local_pred = decode_delocalized_oxy_groups(
                logits[mask], local_ei, elems[start:end], local_pred, group_threshold
            )
        decoded[mask] = local_pred
        start = end
    return decoded


@torch.no_grad()
def evaluate(model, stage1, loader, train_args, device, sigma, group_decode=False,
             group_threshold=0.20, oracle_atom_state=False):
    local_metrics, global_metrics = Metrics(True), Metrics(True)
    h_correct = h_total = charge_correct = charge_total = 0
    for raw in loader:
        if not raw:
            continue
        batch = to_device(raw, device)
        if sigma:
            batch = GaussianNoiseAugment(sigma=sigma).augment_batch(
                batch, sigma=sigma, per_molecule=True
            )
        built = build_refiner_batch(stage1, batch, train_args, training=False)
        need_aux = (model.predict_formal_charge or model.predict_delocalized_center
                    or model.predict_implicit_h or model.predict_hybridization)
        if need_aux:
            logits, aux = forward_refiner(model, batch, built, return_aux=True)
        else:
            logits, aux = forward_refiner(model, batch, built), {}
        if "implicit_h_logits" in aux and "implicit_h_count" in batch:
            h_pred = aux["implicit_h_logits"].argmax(dim=-1)
            h_true = batch["implicit_h_count"].clamp(0, 4).long()
            h_correct += int((h_pred == h_true).sum())
            h_total += int(h_true.numel())
        if "charge_logits" in aux and "formal_charge" in batch:
            q_pred = aux["charge_logits"].argmax(dim=-1) - 2
            q_true = batch["formal_charge"].clamp(-2, 2).long()
            charge_correct += int((q_pred == q_true).sum())
            charge_total += int(q_true.numel())
        if oracle_atom_state:
            if "formal_charge" not in batch or "implicit_h_count" not in batch:
                raise ValueError("--oracle_atom_state requires formal_charge and implicit_h_count labels")
            aux["charge_logits"] = torch.nn.functional.one_hot(
                (batch["formal_charge"].clamp(-2, 2) + 2).long(), num_classes=5
            ).to(torch.float32) * 30.0
            aux["implicit_h_logits"] = torch.nn.functional.one_hot(
                batch["implicit_h_count"].clamp(0, 4).long(), num_classes=5
            ).to(torch.float32) * 30.0
        local = logits.argmax(-1)
        global_pred = decode_batch(
            logits, built["readout_ei"], batch["elems"], batch["num_atoms_per_mol"],
            aux.get("charge_logits"), group_decode, group_threshold,
            aux.get("implicit_h_logits"),
        )
        common = (built["target"], built["readout_ei"], batch["num_atoms_per_mol"])
        local_metrics.update(local, *common)
        global_metrics.update(global_pred, *common)
    result = {"local": local_metrics.compute(), "valence_global": global_metrics.compute()}
    if h_total or charge_total:
        result["atom_state"] = {}
        if h_total:
            result["atom_state"]["implicit_h_accuracy"] = h_correct / h_total
            result["atom_state"]["n_atoms_h"] = h_total
        if charge_total:
            result["atom_state"]["formal_charge_accuracy"] = charge_correct / charge_total
            result["atom_state"]["n_atoms_charge"] = charge_total
    return result


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    ds = CachedBondNetDataset(args.cache_path)

    def _heavy_count(i):
        e = ds.samples[i]["elems"]
        try:
            return int((e != 1).sum())
        except AttributeError:
            return int(sum(1 for z in e if int(z) != 1))

    keep = [i for i in range(len(ds))
            if args.min_heavy <= _heavy_count(i) <= args.max_heavy]
    keep = keep[: args.max_mols]
    print(f"size bucket [{args.min_heavy},{args.max_heavy}] heavy atoms: {len(keep)} molecules")
    ds = Subset(ds, keep)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_stage2_fn, num_workers=args.num_workers,
                        pin_memory=True)
    stage1 = load_stage1(args.stage1_ckpt, device)
    model, train_args = load_refiner(args.refiner_ckpt, device)
    if args.candidate_policy is not None:
        train_args.candidate_policy = args.candidate_policy
        train_args.uncertain_edges_per_atom = args.uncertain_edges_per_atom
    report = {
        "cache": str(Path(args.cache_path).resolve()), "n_molecules": len(ds),
        "refiner": str(Path(args.refiner_ckpt).resolve()), "noise": {},
    }
    for sigma in args.noise:
        torch.manual_seed(args.seed + int(round(1000 * sigma)))
        report["noise"][str(sigma)] = evaluate(
            model, stage1, loader, train_args, device, sigma,
            args.group_decode, args.group_threshold,
            args.oracle_atom_state,
        )
    output = Path(args.output) if args.output else Path(args.refiner_ckpt).with_suffix(".eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
