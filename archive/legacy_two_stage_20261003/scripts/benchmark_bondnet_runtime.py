#!/usr/bin/env python3
"""Measure end-to-end heavy-only BondNet latency on a fixed ligand set.

The measurement includes candidate construction, frozen Stage-1 inference,
Stage-2 inference, and one charge/valence-constrained MILP decode.  Timings are
synchronized around each call on CUDA so they are per-molecule wall times rather
than asynchronous kernel-launch times.  Reference bonds are never read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
    p.add_argument("--device", default="auto")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--max_mols", type=int, default=None)
    return p.parse_args()


def main():
    from rdkit import Chem, RDLogger
    from scripts.benchmark_templatefree_motifs import (
        bondnet_predict, coordinate_view, load_bondnet, load_state_catalogue,
    )

    cli = parse_args()
    RDLogger.DisableLog("rdApp.*")
    device = torch.device("cuda" if cli.device == "auto" and torch.cuda.is_available() else cli.device)
    args = SimpleNamespace(
        stage1_ckpt=cli.stage1_ckpt, refiner_ckpt=cli.refiner_ckpt,
        candidate_radius=2.5, candidate_knn=0, max_neighbors=15,
        h_candidate_cutoff=1.4, bondnet_decode_mode="full_graph",
        heavy_latent_charge_h=True, use_heavy_atom_state_logits=False,
        aromatic_constraints=False, aromatic_bridge_repair=False,
        restrict_delocalized=False, state_catalogue=load_state_catalogue(cli.state_catalogue_json),
        require_kekulizable_candidates=False, single_stage=False,
    )
    stage1, loaded = load_bondnet(args, device)
    refiner, train_args = loaded
    allowed, timings, heavy_atoms = set(cli.allowed_atomic_nums), [], []
    failures = skipped = seen = 0

    for source_index, mol in enumerate(Chem.SDMolSupplier(cli.reference_sdf, removeHs=False)):
        if mol is None or mol.GetNumConformers() == 0:
            continue
        mol = Chem.RemoveHs(mol, sanitize=True)
        if not {atom.GetAtomicNum() for atom in mol.GetAtoms()} <= allowed:
            skipped += 1
            continue
        charge = int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
        elems, coord = coordinate_view(mol, 0.0, source_index * 1009 + 42)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        predicted = bondnet_predict(elems, coord, charge, stage1, refiner, train_args, args, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        seen += 1
        if seen > cli.warmup:
            timings.append(elapsed)
            heavy_atoms.append(int(elems.numel()))
            # The evaluated pose benchmark contains no bondless ligands; an empty graph
            # is therefore a conservative observable proxy for a failed top decode.
            failures += int(not predicted)
        if seen % 100 == 0:
            print(f"timed {seen} molecules", flush=True)
        if cli.max_mols is not None and seen >= cli.max_mols:
            break

    values = np.asarray(timings, dtype=float)
    report = {
        "task": "end-to-end latency for heavy-only BondNet top-graph decoding",
        "included_stages": ["candidate graph", "frozen Stage-1", "Stage-2", "single MILP decode"],
        "device": str(device), "warmup_discarded": min(cli.warmup, seen),
        "n_timed": int(values.size), "skipped_unsupported_elements": skipped,
        "empty_graph_outputs": failures,
        "seconds": ({"mean": float(values.mean()), "median": float(np.median(values)),
                     "p95": float(np.quantile(values, .95)), "min": float(values.min()),
                     "max": float(values.max())} if values.size else None),
        "heavy_atoms": ({"median": float(np.median(heavy_atoms)), "p95": float(np.quantile(heavy_atoms, .95))}
                        if heavy_atoms else None),
        "stage1_ckpt": str(Path(cli.stage1_ckpt).resolve()),
        "refiner_ckpt": str(Path(cli.refiner_ckpt).resolve()),
    }
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
