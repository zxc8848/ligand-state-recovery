"""Pilot BondNet 2.0 refiner: jointly correct connectivity and bond type.

The frozen Stage 1 supplies a noisy message graph and a connectivity confidence
for every candidate edge.  The refiner reads all heavy-heavy candidate edges
and predicts five classes: none, single, double, triple, aromatic (optionally
delocalized).  Unlike the
current hard two-stage pipeline, it can therefore delete Stage-1 false positive
edges and restore false negatives that remain inside the candidate graph.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bondnet.data.dataset import CachedBondNetDataset, collate_stage2_fn
from bondnet.data.noise_augment import GaussianNoiseAugment
from bondnet.loss.focal_loss import FocalLoss
from bondnet.model.bond_type_gnn import BondTypeGNN, compute_bonded_geometry
from bondnet.model.bondnet import BondNet, _symmetrize_logits


CLASS_NAMES = ("single", "double", "triple", "aromatic")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage1_ckpt", required=True)
    p.add_argument(
        "--init_refiner_ckpt", default=None,
        help="Optional compatible structured-refiner checkpoint to fine-tune instead of random initialization.",
    )
    p.add_argument("--cache_path", required=True)
    p.add_argument("--weighted_sampler", action="store_true",
                   help="Draw training batches with WeightedRandomSampler using the cache's "
                        "diversity_weights. Required for mixed-domain caches, where the domain "
                        "ratio is encoded as sampling weight rather than duplicated samples.")
    p.add_argument("--split_key", default=None,
                   help="Sample field to group by before the train/val split (e.g. group_id). "
                        "Mandatory for mixed caches with upsampled molecules, otherwise copies "
                        "of one molecule land on both sides of the split.")
    p.add_argument(
        "--cache_max_samples", type=int, default=None,
        help="Limit the pilot to the first N molecules per fixed split (without rewriting the cache).",
    )
    p.add_argument("--output_dir", default="checkpoints/bond_refiner")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--edge_embedding_size", type=int, default=32)
    p.add_argument("--bond_cutoff", type=float, default=3.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument(
        "--class_weights", type=float, nargs="+", default=None,
        help="Optional weights. Use 5 values normally, or 6 with --structured_six_class.",
    )
    p.add_argument(
        "--structured_six_class", action="store_true",
        help="Use {nobond,single,double,triple,aromatic,delocalized}; delocalized labels are"
             " assigned to symmetric neutral nitro/carboxyl X-O pairs.",
    )
    p.add_argument(
        "--readout_feature_mode", choices=("none", "vector", "angle", "vector_geometry"), default="none",
        help="Extra Stage-2 readout: vector invariants, local geometry, or both.",
    )
    p.add_argument("--charge_aux_weight", type=float, default=0.0,
                   help="Weight of formal-charge auxiliary CE; requires formal_charge in cache.")
    p.add_argument("--implicit_h_aux_weight", type=float, default=0.0,
                   help="Weight of implicit attached-H-count auxiliary CE (heavy-only pilot).")
    p.add_argument("--hybridization_aux_weight", type=float, default=0.0,
                   help="Weight of SP/SP2/SP3/aromatic auxiliary CE; enables soft bond compatibility.")
    p.add_argument("--single_double_margin_weight", type=float, default=0.0,
                   help="Weight of a targeted single-vs-double logit margin loss (six-class only).")
    p.add_argument("--single_double_margin", type=float, default=0.50,
                   help="Required logit gap between the true single/double class and its counterpart.")
    p.add_argument("--hard_double_pair_weight", type=float, default=1.0,
                   help="Extra focal-loss multiplier for true C=N, P=O, and S=O double bonds.")
    p.add_argument("--total_charge_condition", action="store_true",
                   help="Broadcast the generator-supplied molecular net charge to every atom.")
    p.add_argument("--delocalized_group_aux_weight", type=float, default=0.0,
                   help="Weight of the atom-level delocalized-center auxiliary BCE.")
    p.add_argument("--hard_negative_weight", type=float, default=1.0,
                   help="Extra loss weight for close/high-confidence non-bond candidates.")
    p.add_argument("--hard_negative_distance", type=float, default=1.90,
                   help="Heavy-pair distance threshold (A) for a hard non-bond.")
    p.add_argument("--hard_negative_confidence", type=float, default=0.20,
                   help="Stage-1 connectivity probability threshold for a hard non-bond.")
    p.add_argument("--candidate_policy", choices=("static", "uncertainty"), default="static",
                   help="Stage-2 message graph: fixed threshold or uncertainty-adaptive edge budget.")
    p.add_argument("--uncertain_edges_per_atom", type=int, default=1,
                   help="For uncertainty policy, retain this many uncertain non-bond pairs per atom.")
    p.add_argument("--single_stage", action="store_true",
                   help="Ablation: no Stage-1. Message passing runs over the full candidate "
                        "graph and the six-class readout (incl. nobond) does connectivity and "
                        "typing jointly. --stage1_ckpt is still read for the backbone hidden "
                        "size but its connectivity predictions are unused (conn feature zeroed).")
    p.add_argument("--conn_threshold", type=float, default=0.5)
    p.add_argument(
        "--retain_threshold", type=float, default=None,
        help="Also retain Stage-1 candidate edges with p(connect)>=this value in the Stage-2"
             " message graph, letting the nobond class explicitly delete false positives.",
    )
    p.add_argument("--topology_fp_rate", type=float, default=0.02)
    p.add_argument("--topology_fn_rate", type=float, default=0.02)
    p.add_argument("--teacher_message_prob", type=float, default=0.20)
    p.add_argument("--noise_min", type=float, default=0.0)
    p.add_argument("--noise_max", type=float, default=0.05)
    p.add_argument("--noise_clean_prob", type=float, default=0.30)
    p.add_argument("--test_noise", type=float, nargs="+", default=(0.0, 0.05, 0.10))
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def load_stage1(path: str, device: torch.device) -> BondNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {})
    model = BondNet(
        num_interactions=saved.get("num_interactions", 4),
        hidden_size=saved.get("hidden_size", 128),
        cutoff=saved.get("cutoff", 2.5),
        edge_embedding_size=saved.get("edge_embedding_size", 32),
        vector_norm_limit=saved.get("vector_norm_limit", 3.0),
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device)


def make_loaders(args):
    # split_key groups samples before splitting. Required for any cache built by
    # build_mixed_pose_cache.py: upsampled copies of one molecule share a
    # group_id, and without grouping a random per-sample split would put copies
    # of the same molecule in both train and validation, so the validation score
    # would measure memorisation rather than generalisation.
    train_ds, val_ds = CachedBondNetDataset.split(
        args.cache_path, val_frac=0.1, seed=args.seed, split_key=args.split_key
    )
    full_ds = CachedBondNetDataset(args.cache_path)
    test_indices = [i for i, sample in enumerate(full_ds.samples) if sample.get("split") == "test"]
    if args.cache_max_samples is not None:
        # Keep the fixed split semantics while making small pilot runs possible.
        # The cache is still read once, but only a bounded number of examples
        # enter each DataLoader.
        n = max(1, int(args.cache_max_samples))
        n_train = max(1, int(round(n * 0.8)))
        n_val = max(1, int(round(n * 0.1)))
        n_test = max(1, n - n_train - n_val)
        train_ds = Subset(train_ds, range(min(len(train_ds), n_train)))
        val_ds = Subset(val_ds, range(min(len(val_ds), n_val)))
        test_indices = test_indices[:min(len(test_indices), n_test)]
    test_ds = Subset(full_ds, test_indices)
    common = dict(
        batch_size=args.batch_size,
        collate_fn=collate_stage2_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    train_sampler = None
    if args.weighted_sampler:
        weights = getattr(train_ds, "_diversity_weights", None)
        if weights is None:
            raise ValueError("--weighted_sampler needs diversity_weights in the cache")
        from torch.utils.data import WeightedRandomSampler
        train_sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(train_ds), replacement=True)
        domains = [s.get("domain") for s in getattr(train_ds, "samples", [])]
        if any(d is not None for d in domains):
            total = float(weights.sum())
            pose_mass = float(sum(w for w, d in zip(weights.tolist(), domains) if d == "pose"))
            print(f"weighted sampler: pose expected share {pose_mass / total:.1%} of draws")
    return (
        DataLoader(train_ds, shuffle=(train_sampler is None), sampler=train_sampler,
                   drop_last=False, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
        (len(train_ds), len(val_ds), len(test_ds)),
    )


def augment(batch, sigma_min, sigma_max, clean_prob):
    if sigma_max <= 0:
        return batch
    return GaussianNoiseAugment(
        sigma=sigma_max,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        cutoff=3.0,
    ).augment_batch(batch, per_molecule=True, clean_prob=clean_prob)


def pair_random_mask(edge_index: torch.Tensor, probability: float) -> torch.Tensor:
    if probability <= 0 or edge_index.numel() == 0:
        return torch.zeros(edge_index.shape[0], dtype=torch.bool, device=edge_index.device)
    i, j = edge_index[:, 0], edge_index[:, 1]
    n = int(max(i.max(), j.max()).item()) + 1
    lo, hi = torch.minimum(i, j), torch.maximum(i, j)
    _, inverse = torch.unique(lo * n + hi, return_inverse=True)
    decisions = torch.rand(int(inverse.max().item()) + 1, device=edge_index.device) < probability
    return decisions[inverse]


@torch.no_grad()
def stage1_predictions(stage1: BondNet, batch):
    _, _, edge_feat = stage1.backbone(
        batch["elems"], batch["coord"], batch["edge_index"],
        batch["edge_diff"], batch["edge_dist"],
    )
    logits = stage1.stage1(edge_feat)
    return logits, torch.sigmoid(logits)


def mark_symmetric_delocalized(full_target, edge_index, true_mask, elems):
    """Map neutral nitro/carboxyl single+double X-O pairs to class 5.

    The transform intentionally only covers unambiguous, symmetric two-terminal-O
    motifs.  Charged motifs and other resonance systems remain their conventional
    discrete labels, which keeps this pilot's new supervision conservative.
    """
    src, dst = edge_index[:, 0], edge_index[:, 1]
    heavy_true = true_mask & (elems[src] != 1) & (elems[dst] != 1)
    # Each heavy neighbour occurs once in the outgoing directed graph.
    heavy_degree = torch.zeros(elems.shape[0], device=elems.device, dtype=full_target.dtype)
    heavy_degree.index_add_(0, src, heavy_true.to(full_target.dtype))
    center_to_oxygen = (
        heavy_true & ((elems[src] == 6) | (elems[src] == 7)) & (elems[dst] == 8)
        & (heavy_degree[dst] == 1)
    )
    terminal_count = torch.zeros_like(heavy_degree)
    terminal_order = torch.zeros_like(heavy_degree)
    terminal_count.index_add_(0, src, center_to_oxygen.to(full_target.dtype))
    terminal_order.index_add_(0, src, full_target * center_to_oxygen.to(full_target.dtype))
    # ``count=2, total=3`` is exactly a single+double X-O pair.  The reverse
    # directed edges are selected below from the same valid center mask.
    valid_center = (terminal_count == 2) & (terminal_order == 3)
    mark = heavy_true & (
        (valid_center[src] & (elems[dst] == 8) & (heavy_degree[dst] == 1))
        | (valid_center[dst] & (elems[src] == 8) & (heavy_degree[src] == 1))
    )
    full_target[mark] = 5
    return full_target


def delocalized_center_target(full_target, edge_index, true_mask, elems):
    """Return a conservative group-level label for nitro/carboxyl centers."""
    src, dst = edge_index[:, 0], edge_index[:, 1]
    heavy_true = true_mask & (elems[src] != 1) & (elems[dst] != 1)
    degree = torch.zeros(elems.shape[0], device=elems.device, dtype=full_target.dtype)
    degree.index_add_(0, src, heavy_true.to(full_target.dtype))
    c_to_o = (heavy_true & ((elems[src] == 6) | (elems[src] == 7))
              & (elems[dst] == 8) & (degree[dst] == 1))
    count = torch.zeros_like(degree)
    order = torch.zeros_like(degree)
    count.index_add_(0, src, c_to_o.to(full_target.dtype))
    order.index_add_(0, src, full_target * c_to_o.to(full_target.dtype))
    return ((count == 2) & (order == 3)).to(torch.float32)


def uncertainty_candidate_mask(edge_index, conn_prob, base_mask, per_atom: int):
    """Retain a symmetric, bounded budget of uncertain Stage-1 pairs per atom.

    This makes topology expansion local: atoms with ambiguous neighbourhoods get
    extra Stage-2 context, while confidently non-bonded regions remain sparse.
    """
    if per_atom <= 0 or edge_index.numel() == 0:
        return base_mask
    n_atoms = int(edge_index.max().item()) + 1
    lo, hi = torch.minimum(edge_index[:, 0], edge_index[:, 1]), torch.maximum(edge_index[:, 0], edge_index[:, 1])
    key = lo * n_atoms + hi
    pairs, inverse = torch.unique(key, return_inverse=True)
    n_pairs = pairs.numel()
    pair_prob = torch.zeros(n_pairs, device=conn_prob.device, dtype=conn_prob.dtype)
    pair_count = torch.zeros_like(pair_prob)
    pair_prob.index_add_(0, inverse, conn_prob)
    pair_count.index_add_(0, inverse, torch.ones_like(conn_prob))
    pair_prob = pair_prob / pair_count.clamp_min(1)
    pair_base = torch.zeros(n_pairs, device=conn_prob.device, dtype=torch.bool)
    pair_base.index_put_((inverse,), base_mask, accumulate=True)
    # Bernoulli entropy is high near p=0.5; only non-base pairs compete for
    # the adaptive context budget.
    p = pair_prob.clamp(1e-6, 1.0 - 1e-6)
    entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
    pair_i, pair_j = pairs // n_atoms, pairs % n_atoms
    nodes = torch.cat((pair_i, pair_j))
    pair_ids = torch.cat((torch.arange(n_pairs, device=conn_prob.device),) * 2)
    scores = torch.cat((entropy, entropy))
    eligible = ~pair_base[pair_ids]
    # Stable two-key sort: descending entropy within each receiving atom.
    order = torch.argsort(scores, descending=True, stable=True)
    order = order[torch.argsort(nodes[order], stable=True)]
    sorted_nodes = nodes[order]
    rank_pos = torch.arange(order.numel(), device=conn_prob.device)
    starts = torch.where(
        torch.cat((torch.ones(1, dtype=torch.bool, device=conn_prob.device), sorted_nodes[1:] != sorted_nodes[:-1])),
        rank_pos, torch.zeros_like(rank_pos),
    )
    starts = torch.cummax(starts, dim=0).values
    take = eligible[order] & ((rank_pos - starts) < per_atom)
    pair_keep = pair_base.clone()
    pair_keep[pair_ids[order[take]]] = True
    return pair_keep[inverse]


def build_refiner_batch(stage1, batch, args, training: bool):
    edge_index = batch["edge_index"]
    train_mask = batch.get(
        "train_edge_mask",
        torch.ones(edge_index.shape[0], dtype=torch.bool, device=edge_index.device),
    ).bool()
    true_mask = batch["bond_mask"].bool() & train_mask
    elems = batch["elems"]
    hh_mask = (elems[edge_index[:, 0]] != 1) & (elems[edge_index[:, 1]] != 1)
    single_stage = getattr(args, "single_stage", False)

    if single_stage:
        # No Stage-1. The network performs connectivity and bond typing jointly:
        # message passing runs over the FULL distance-cutoff candidate graph and
        # the six-class readout (with its nobond class) decides every candidate.
        # No Stage-1 connectivity feature is supplied. This is a matched-capacity
        # ablation of the two-stage design -- same backbone, same readout, only
        # the message graph and the conn feature differ.
        conn_logits = torch.zeros(edge_index.shape[0], device=edge_index.device)
        conn_prob = torch.zeros(edge_index.shape[0], device=edge_index.device)
        message_mask = (train_mask & hh_mask).clone()
    else:
        conn_logits, conn_prob = stage1_predictions(stage1, batch)
        predicted_mask = (conn_prob >= args.conn_threshold) & train_mask
        if args.retain_threshold is not None:
            # Deliberately keep uncertain Stage-1 non-bonds as context.  Stage 2 is
            # trained to remove them through its explicit ``nobond`` class.
            predicted_mask = predicted_mask | ((conn_prob >= args.retain_threshold) & train_mask)
        if getattr(args, "candidate_policy", "static") == "uncertainty":
            predicted_mask = uncertainty_candidate_mask(
                edge_index, conn_prob, predicted_mask,
                getattr(args, "uncertain_edges_per_atom", 1),
            ) & train_mask

        if training and torch.rand((), device=edge_index.device) < args.teacher_message_prob:
            message_mask = true_mask.clone()
        else:
            message_mask = predicted_mask.clone()
        if training:
            add = pair_random_mask(edge_index, args.topology_fp_rate) & ~true_mask & train_mask
            drop = pair_random_mask(edge_index, args.topology_fn_rate) & true_mask
            message_mask = (message_mask | add) & ~drop

    message_ei = edge_index[message_mask]
    message_diff, message_dist = compute_bonded_geometry(batch["coord"], message_ei)

    readout_mask = train_mask & hh_mask
    readout_ei = edge_index[readout_mask]
    readout_diff, readout_dist = compute_bonded_geometry(batch["coord"], readout_ei)

    full_target = torch.zeros(edge_index.shape[0], dtype=torch.long, device=edge_index.device)
    bonded_labels = batch["bond_type_full"].long() + 1
    if int(true_mask.sum()) != bonded_labels.shape[0]:
        raise ValueError(
            f"bond mask/label mismatch: {int(true_mask.sum())} vs {bonded_labels.shape[0]}"
        )
    full_target[true_mask] = bonded_labels
    center_target = None
    if args.structured_six_class:
        center_target = delocalized_center_target(full_target, edge_index, true_mask, elems)
        full_target = mark_symmetric_delocalized(full_target, edge_index, true_mask, elems)
    target = full_target[readout_mask]
    extra = torch.stack((conn_logits[readout_mask].clamp(-8, 8) / 8.0, conn_prob[readout_mask]), dim=-1)
    return {
        "message_ei": message_ei,
        "message_diff": message_diff,
        "message_dist": message_dist,
        "readout_ei": readout_ei,
        "readout_diff": readout_diff,
        "readout_dist": readout_dist,
        "extra": extra,
        "target": target,
        "conn_prob": conn_prob,
        "readout_conn_prob": conn_prob[readout_mask],
        "delocalized_center_target": center_target,
    }


def forward_refiner(model, batch, built, return_aux=False):
    atom_mol_index = torch.repeat_interleave(
        torch.arange(batch["num_atoms_per_mol"].numel(), device=batch["elems"].device),
        batch["num_atoms_per_mol"],
    )
    output = model(
        batch["elems"],
        built["message_ei"], built["message_diff"], built["message_dist"],
        built["readout_ei"], built["readout_diff"], built["readout_dist"],
        readout_extra=built["extra"], coord=batch["coord"], return_aux=return_aux,
        molecular_total_charge=batch.get("molecular_total_charge"),
        atom_mol_index=atom_mol_index if model.total_charge_condition else None,
    )
    if not return_aux:
        return _symmetrize_logits(output, built["readout_ei"])
    logits, aux = output
    return _symmetrize_logits(logits, built["readout_ei"]), aux


class Metrics:
    def __init__(self, structured_six_class=False):
        self.class_names = CLASS_NAMES + (("delocalized",) if structured_six_class else ())
        self.num_bond_classes = len(self.class_names)
        self.tp = torch.zeros(self.num_bond_classes, dtype=torch.long)
        self.fp = torch.zeros(self.num_bond_classes, dtype=torch.long)
        self.fn = torch.zeros(self.num_bond_classes, dtype=torch.long)
        self.conn_tp = self.conn_fp = self.conn_fn = 0
        self.strict = self.true_exact = self.molecules = 0

    def update(self, pred, target, readout_ei, atoms_per_mol):
        pred, target = pred.detach().cpu(), target.detach().cpu()
        edges = readout_ei.detach().cpu()
        for cls in range(1, self.num_bond_classes + 1):
            self.tp[cls - 1] += ((pred == cls) & (target == cls)).sum()
            self.fp[cls - 1] += ((pred == cls) & (target != cls)).sum()
            self.fn[cls - 1] += ((pred != cls) & (target == cls)).sum()
        pred_conn, true_conn = pred > 0, target > 0
        self.conn_tp += int((pred_conn & true_conn).sum())
        self.conn_fp += int((pred_conn & ~true_conn).sum())
        self.conn_fn += int((~pred_conn & true_conn).sum())

        start = 0
        for count in atoms_per_mol.detach().cpu().tolist():
            end = start + int(count)
            mask = (edges[:, 0] >= start) & (edges[:, 0] < end)
            p, t = pred[mask], target[mask]
            self.strict += int(bool(p.numel()) and bool(torch.equal(p, t)))
            true_edges = t > 0
            self.true_exact += int(bool(true_edges.any()) and bool(torch.equal(p[true_edges], t[true_edges])))
            self.molecules += 1
            start = end

    @staticmethod
    def _f1(tp, fp, fn):
        denom = 2 * tp + fp + fn
        return (2 * tp / denom) if denom else 0.0

    def compute(self):
        result = {}
        f1s = []
        active_f1s = []
        supported_f1s = []
        for idx, name in enumerate(self.class_names):
            tp, fp, fn = int(self.tp[idx]), int(self.fp[idx]), int(self.fn[idx])
            f1 = self._f1(tp, fp, fn)
            result[f"f1_{name}"] = f1
            f1s.append(f1)
            if 2 * tp + fp + fn:
                active_f1s.append(f1)
            if tp + fn:
                supported_f1s.append(f1)
        result["f1_macro"] = sum(f1s) / len(f1s)
        result["f1_macro_present"] = sum(active_f1s) / max(len(active_f1s), 1)
        result["n_active_bond_classes"] = len(active_f1s)
        result["f1_macro_true_support"] = sum(supported_f1s) / max(len(supported_f1s), 1)
        result["n_true_bond_classes"] = len(supported_f1s)
        result["conn_f1"] = self._f1(self.conn_tp, self.conn_fp, self.conn_fn)
        result["strict_graph_exact"] = self.strict / max(self.molecules, 1)
        result["true_bond_exact"] = self.true_exact / max(self.molecules, 1)
        result["n_molecules"] = self.molecules
        return result


def structured_loss(logits, aux, batch, built, criterion, args):
    """Edge focal loss plus optional atom-state and hard-negative objectives."""
    per_edge = criterion(logits, built["target"], reduction="none")
    if args.hard_negative_weight > 1.0:
        hard_nonbond = ((built["target"] == 0)
                        & ((built["readout_dist"] <= args.hard_negative_distance)
                           | (built["readout_conn_prob"] >= args.hard_negative_confidence)))
        per_edge = per_edge * torch.where(
            hard_nonbond, torch.full_like(per_edge, args.hard_negative_weight), torch.ones_like(per_edge)
        )
    if args.hard_double_pair_weight > 1.0:
        # PDBBind failure attribution shows that actual double-to-single
        # undercalls concentrate in C=N and hypervalent P=O/S=O.  Reweight
        # only those reference double edges, rather than all double bonds:
        # conjugated C=O is already recognised and mostly differs only by the
        # valid delocalized-vs-Kekule representation.
        pair_elems = batch["elems"][built["readout_ei"]]
        lo, hi = pair_elems.min(dim=1).values, pair_elems.max(dim=1).values
        difficult_pair = ((lo == 6) & (hi == 7)) | ((lo == 8) & ((hi == 15) | (hi == 16)))
        targeted_double = (built["target"] == 2) & difficult_pair
        per_edge = per_edge * torch.where(
            targeted_double,
            torch.full_like(per_edge, args.hard_double_pair_weight),
            torch.ones_like(per_edge),
        )
    loss = per_edge.mean()

    if args.charge_aux_weight > 0:
        if "formal_charge" not in batch or "charge_logits" not in aux:
            raise ValueError("charge auxiliary loss requires a cache with formal_charge labels")
        charge_target = batch["formal_charge"].clamp(-2, 2) + 2
        counts = torch.bincount(charge_target.long(), minlength=5).to(torch.float32)
        charge_weight = torch.sqrt(counts.sum() / (5.0 * counts.clamp_min(1.0)))
        charge_weight = charge_weight / charge_weight.mean()
        loss = loss + args.charge_aux_weight * torch.nn.functional.cross_entropy(
            aux["charge_logits"], charge_target.long(), weight=charge_weight
        )
    if args.implicit_h_aux_weight > 0:
        if "implicit_h_count" not in batch or "implicit_h_logits" not in aux:
            raise ValueError("implicit-H auxiliary loss requires a cache with implicit_h_count labels")
        h_target = batch["implicit_h_count"].clamp(0, 4).long()
        counts = torch.bincount(h_target, minlength=5).to(torch.float32)
        h_weight = torch.sqrt(counts.sum() / (5.0 * counts.clamp_min(1.0)))
        h_weight = h_weight / h_weight.mean()
        loss = loss + args.implicit_h_aux_weight * torch.nn.functional.cross_entropy(
            aux["implicit_h_logits"], h_target, weight=h_weight
        )
    if args.hybridization_aux_weight > 0:
        if "hybridization" not in batch or "hybridization_logits" not in aux:
            raise ValueError(
                "hybridization auxiliary loss requires a cache rebuilt with hybridization labels"
            )
        hybrid_target = batch["hybridization"].long()
        hybrid_mask = hybrid_target >= 0
        if hybrid_mask.any():
            y = hybrid_target[hybrid_mask]
            counts = torch.bincount(y, minlength=4).to(torch.float32)
            class_weight = torch.sqrt(counts.sum() / (4.0 * counts.clamp_min(1.0)))
            class_weight = class_weight / class_weight.mean()
            loss = loss + args.hybridization_aux_weight * torch.nn.functional.cross_entropy(
                aux["hybridization_logits"][hybrid_mask], y, weight=class_weight
            )
    if args.single_double_margin_weight > 0:
        if logits.shape[-1] < 3:
            raise ValueError("single/double margin requires structured bond classes")
        target = built["target"]
        sd_mask = (target == 1) | (target == 2)
        if sd_mask.any():
            # Only resolves the observed single<->double competition.  It does
            # not force a bond order through the decoder and does not conflate
            # the separate delocalized/Kekule representation issue.
            signed_gap = torch.where(
                target[sd_mask] == 2,
                logits[sd_mask, 2] - logits[sd_mask, 1],
                logits[sd_mask, 1] - logits[sd_mask, 2],
            )
            loss = loss + args.single_double_margin_weight * torch.relu(
                args.single_double_margin - signed_gap
            ).mean()
    if args.delocalized_group_aux_weight > 0:
        target = built["delocalized_center_target"]
        if target is None or "delocalized_center_logits" not in aux:
            raise ValueError("delocalized group auxiliary loss requires --structured_six_class")
        center_mask = (batch["elems"] == 6) | (batch["elems"] == 7)
        y = target[center_mask]
        x = aux["delocalized_center_logits"][center_mask]
        if y.numel():
            pos_weight = ((1.0 - y.mean()) / y.mean().clamp_min(1.0 / max(y.numel(), 1))).clamp(max=50.0)
            loss = loss + args.delocalized_group_aux_weight * torch.nn.functional.binary_cross_entropy_with_logits(
                x, y, pos_weight=pos_weight
            )
    return loss


def run_epoch(model, stage1, loader, criterion, device, args, optimizer=None):
    training = optimizer is not None
    model.train(training)
    metrics = Metrics(args.structured_six_class)
    loss_sum = 0.0
    batches = 0
    for raw_batch in loader:
        if not raw_batch:
            continue
        batch = to_device(raw_batch, device)
        if training:
            batch = augment(batch, args.noise_min, args.noise_max, args.noise_clean_prob)
        built = build_refiner_batch(stage1, batch, args, training=training)
        need_aux = (args.charge_aux_weight > 0 or args.delocalized_group_aux_weight > 0
                    or args.implicit_h_aux_weight > 0 or args.hybridization_aux_weight > 0)
        if need_aux:
            logits, aux = forward_refiner(model, batch, built, return_aux=True)
        else:
            logits, aux = forward_refiner(model, batch, built), {}
        loss = structured_loss(logits, aux, batch, built, criterion, args)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        loss_sum += float(loss.detach())
        batches += 1
        metrics.update(logits.argmax(-1), built["target"], built["readout_ei"], batch["num_atoms_per_mol"])
    return loss_sum / max(batches, 1), metrics.compute()


@torch.no_grad()
def evaluate_noise(model, stage1, loader, criterion, device, args, sigma):
    torch.manual_seed(args.seed + int(round(float(sigma) * 1000)))
    model.eval()
    metrics = Metrics(args.structured_six_class)
    loss_sum = 0.0
    batches = 0
    for raw_batch in loader:
        if not raw_batch:
            continue
        batch = to_device(raw_batch, device)
        if sigma > 0:
            batch = GaussianNoiseAugment(sigma=sigma).augment_batch(batch, sigma=sigma, per_molecule=True)
        built = build_refiner_batch(stage1, batch, args, training=False)
        need_aux = (args.charge_aux_weight > 0 or args.delocalized_group_aux_weight > 0
                    or args.implicit_h_aux_weight > 0 or args.hybridization_aux_weight > 0)
        if need_aux:
            logits, aux = forward_refiner(model, batch, built, return_aux=True)
        else:
            logits, aux = forward_refiner(model, batch, built), {}
        loss_sum += float(structured_loss(logits, aux, batch, built, criterion, args).detach())
        batches += 1
        metrics.update(logits.argmax(-1), built["target"], built["readout_ei"], batch["num_atoms_per_mol"])
    result = metrics.compute()
    result["loss"] = loss_sum / max(batches, 1)
    result["sigma"] = float(sigma)
    return result


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, sizes = make_loaders(args)
    print(f"device={device} train={sizes[0]} val={sizes[1]} test={sizes[2]}")
    stage1 = load_stage1(args.stage1_ckpt, device)
    model = BondTypeGNN(
        hidden_size=stage1.backbone.hidden_state_size,
        num_layers=args.num_layers,
        edge_embedding_size=args.edge_embedding_size,
        bond_cutoff=args.bond_cutoff,
        num_bond_types=6 if args.structured_six_class else 5,
        dropout=args.dropout,
        vector_norm_limit=5,
        readout_extra_dim=2,
        readout_feature_mode=args.readout_feature_mode,
        predict_formal_charge=args.charge_aux_weight > 0,
        predict_delocalized_center=args.delocalized_group_aux_weight > 0,
        predict_implicit_h=args.implicit_h_aux_weight > 0,
        predict_hybridization=args.hybridization_aux_weight > 0,
        total_charge_condition=args.total_charge_condition,
        backbone=getattr(args, "backbone", "painn"),
    ).to(device)
    if args.init_refiner_ckpt:
        init = torch.load(args.init_refiner_ckpt, map_location=device, weights_only=False)
        saved_state = init["model_state_dict"]
        current_state = model.state_dict()
        compatible = {
            key: value for key, value in saved_state.items()
            if key in current_state and current_state[key].shape == value.shape
        }
        skipped = sorted(set(saved_state) - set(compatible))
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        allowed_missing = {key for key in missing if key.startswith(("cls_head.0", "hybridization_", "hybrid_bond_compat"))}
        if unexpected or set(missing) != allowed_missing:
            raise ValueError(
                "Incompatible --init_refiner_ckpt architecture; "
                f"missing={missing}, unexpected={unexpected}. "
                "Match the original six-class/readout/auxiliary-head configuration."
            )
        print(f"fine-tuning refiner from {args.init_refiner_ckpt}; reinitialized={skipped}")
    default_weights = ((0.20, 1.0, 2.0, 4.0, 2.0, 2.0)
                       if args.structured_six_class else (0.20, 1.0, 2.0, 4.0, 2.0))
    class_weights = tuple(args.class_weights) if args.class_weights is not None else default_weights
    expected_classes = 6 if args.structured_six_class else 5
    if len(class_weights) != expected_classes:
        raise ValueError(f"Expected {expected_classes} --class_weights values, got {len(class_weights)}")
    weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = FocalLoss(gamma=args.focal_gamma, weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.01)

    best_score = -1.0
    best_path = output / "best.pt"
    for epoch in range(args.epochs):
        started = time.time()
        train_loss, train_metrics = run_epoch(model, stage1, train_loader, criterion, device, args, optimizer)
        with torch.no_grad():
            val_loss, val_metrics = run_epoch(model, stage1, val_loader, criterion, device, args)
        scheduler.step()
        # Before global decoding, a retained non-bond candidate can make the
        # strict whole-graph metric zero even when all true bonds are correct.
        # Macro bond-class F1 is therefore the stable early-stopping signal for
        # the structured (six-class) pilot; final graph exactness is measured
        # after the constrained decoder.
        score = val_metrics["f1_macro"] if args.structured_six_class else val_metrics["strict_graph_exact"]
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "val_metrics": val_metrics,
                },
                best_path,
            )
        print(
            f"epoch={epoch + 1:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_conn={val_metrics['conn_f1']:.4f} val_macro={val_metrics['f1_macro']:.4f} "
            f"val_true_exact={val_metrics['true_bond_exact']:.3f} "
            f"val_strict={val_metrics['strict_graph_exact']:.3f} time={time.time() - started:.1f}s"
        )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_results = [
        evaluate_noise(model, stage1, test_loader, criterion, device, args, sigma)
        for sigma in args.test_noise
    ]
    report = {
        "dataset": str(Path(args.cache_path).resolve()),
        "stage1": str(Path(args.stage1_ckpt).resolve()),
        "best_epoch": int(checkpoint["epoch"]) + 1,
        "best_val": checkpoint["val_metrics"],
        "test": test_results,
        "args": vars(args),
    }
    (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": report["best_epoch"], "test": test_results}, indent=2))


if __name__ == "__main__":
    main()
