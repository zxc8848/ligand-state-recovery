"""Recover and validate BondNet graphs from MolCraftDiffusion XYZ samples.

Example
-------
python scripts/run_molcraft_bondnet.py \
  --xyz_dir molcraft_generated --total_charge 0 \
  --stage1_ckpt checkpoints/allatom_stage1/best.pt \
  --refiner_ckpt checkpoints/allatom_refiner/best.pt \
  --explicit_h \
  --output_dir results/molcraft_bondnet

The script is intentionally independent of MolCraftDiffusion's environment:
MolCraft writes XYZ, while this bridge writes a per-sample report, valid SDF
structures, and optional inpainting requests for rejected samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bondnet.inference.chemical_repair import (
    build_sanitized_mol,
    forcefield_basin_displacement,
    joint_decode,
)
from bondnet.integration.molcraft import (
    build_refiner_inference_batch,
    forward_coordinate_refiner,
    make_coordinate_batch,
    all_atom_candidate_edges,
    read_xyz,
    read_heavy_xyz,
    stage1_hydrogen_logits,
    uncertainty_error_mask,
)
from scripts.evaluate_structured_refiner import load_refiner
from scripts.train_bond_refiner import load_stage1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xyz_dir", required=True, help="Directory of XYZ samples written by MolCraftDiffusion.")
    p.add_argument("--recursive_xyz", action="store_true",
                   help="Recursively collect XYZ files below --xyz_dir (useful for batched in-painting outputs).")
    p.add_argument("--stage1_ckpt", required=True)
    p.add_argument("--refiner_ckpt", required=True)
    p.add_argument("--total_charge", type=int, default=0,
                   help="Fallback molecular charge when --charge_manifest is absent.")
    p.add_argument("--charge_manifest", default=None,
                   help="CSV with filename,total_charge columns; filename may be a basename or stem.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--candidate_radius", type=float, default=3.0)
    p.add_argument("--candidate_knn", type=int, default=12)
    p.add_argument("--max_neighbors", type=int, default=15)
    p.add_argument("--explicit_h", action="store_true",
                   help="Keep generated H atoms and decode H-X edges instead of predicting implicit H.")
    p.add_argument("--allow_residual_implicit_h", action="store_true",
                   help="Explicit-H mode: retain decoded H-X edges but allow RDKit to add missing H on heavy atoms. "
                        "Use for generators whose XYZ atom inventory is only partially hydrogenated.")
    p.add_argument("--h_candidate_cutoff", type=float, default=2.0,
                   help="Explicit-H mode: geometric cutoff for H-heavy candidate edges (A).")
    p.add_argument("--aromatic_constraints", action="store_true",
                   help="Constrain aromatic selections to cycle-capable, locally coherent aromatic subgraphs.")
    p.add_argument("--aromatic_rescue", action="store_true",
                   help="After ordinary decoding fails sanitization, try one additional no-aromatic "
                        "single/double Kekule MILP proposal. Reported separately from Top-K.")
    p.add_argument("--decode_trials", type=int, default=1,
                   help="Number of constrained stochastic decode proposals per XYZ (1 = MAP only).")
    p.add_argument("--decode_jitter", type=float, default=0.0,
                   help="Optional logit Gaussian jitter after no-good cuts; default 0 enumerates exact Top-K MILP graphs.")
    p.add_argument("--decode_seed", type=int, default=42)
    p.add_argument("--rank_by_basin", action="store_true",
                   help="Among sanitized Top-K graph hypotheses, retain the one with the smallest force-field "
                        "heavy-atom displacement; does not modify coordinates.")
    p.add_argument("--basin_weight", type=float, default=None,
                   help="If supplied, maximize decoder objective - basin_weight * basin displacement among "
                        "sanitized Top-K graphs. Overrides pure --rank_by_basin selection.")
    p.add_argument("--max_error_atoms", type=int, default=8)
    p.add_argument("--max_mols", type=int, default=None)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def charge_lookup(path: str | None) -> dict[str, int]:
    if path is None:
        return {}
    values = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if "filename" not in row or "total_charge" not in row:
                raise ValueError("--charge_manifest requires filename,total_charge columns")
            values[row["filename"]] = int(row["total_charge"])
    return values


def total_charge_for(path: Path, charges: dict[str, int], fallback: int) -> int:
    return charges.get(path.name, charges.get(path.stem, int(fallback)))


def undirected_labels(edge_index: torch.Tensor, labels: torch.Tensor):
    """Collapse BondNet's symmetric directed output to one label per pair."""
    found = {}
    for row, (i, j) in enumerate(edge_index.detach().cpu().tolist()):
        found.setdefault(tuple(sorted((int(i), int(j)))), int(labels[row]))
    return list(found), list(found.values())


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    out = Path(args.output_dir)
    valid_dir = out / "valid_sdf"
    out.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    xyz_root = Path(args.xyz_dir)
    xyz_files = sorted(xyz_root.rglob("*.xyz") if args.recursive_xyz else xyz_root.glob("*.xyz"))
    if args.max_mols is not None:
        xyz_files = xyz_files[:int(args.max_mols)]
    if not xyz_files:
        raise ValueError(f"No XYZ files found in {args.xyz_dir}")

    charges = charge_lookup(args.charge_manifest)
    stage1 = load_stage1(args.stage1_ckpt, device)
    refiner, train_args = load_refiner(args.refiner_ckpt, device)
    from rdkit import Chem
    writer = Chem.SDWriter(str(valid_dir / "bondnet_valid.sdf"))
    reports, inpaint_requests = [], []

    for sample_index, xyz_path in enumerate(xyz_files):
        try:
            source_relpath = str(xyz_path.relative_to(xyz_root))
            if args.explicit_h:
                elems, coord = read_xyz(xyz_path, keep_hydrogens=True)
                candidate_edges = all_atom_candidate_edges(
                    elems, coord, args.candidate_radius, args.candidate_knn,
                    args.max_neighbors, args.h_candidate_cutoff,
                )
            else:
                elems, coord = read_heavy_xyz(xyz_path)
                candidate_edges = None
            total_charge = total_charge_for(xyz_path, charges, args.total_charge)
            batch = make_coordinate_batch(
                elems.to(device), coord.to(device), total_charge,
                args.candidate_radius, args.candidate_knn, args.max_neighbors,
                candidate_edge_index=candidate_edges.to(device) if candidate_edges is not None else None,
            )
            built = build_refiner_inference_batch(
                stage1, batch, train_args, refiner_heavy_only=args.explicit_h,
            )
            logits, aux = forward_coordinate_refiner(refiner, batch, built, return_aux=True)
            decode_edge_index = built["readout_ei"]
            if args.explicit_h:
                # Structured refiner checkpoints only learned heavy-heavy
                # labels.  H-X edges use the explicit-H Stage-1 head and can
                # only be {nobond, single}; see stage1_hydrogen_logits.
                all_logits = stage1_hydrogen_logits(built["all_conn_logits"], batch["edge_index"])
                all_logits[built["readout_mask"]] = logits
                logits = all_logits
                decode_edge_index = batch["edge_index"]
            generator = torch.Generator(device=device)
            generator.manual_seed(int(args.decode_seed) + sample_index)
            result = None
            mol = None
            accepted = False
            selected_trial = 0
            selected_basin_displacement = None
            sanitized_candidates = []
            forbidden_labelings = []
            n_decode_attempts = 0
            aromatic_rescue_used = False
            use_basin_ranking = args.rank_by_basin or args.basin_weight is not None
            for trial in range(max(1, int(args.decode_trials))):
                n_decode_attempts = trial + 1
                trial_logits = logits if trial == 0 else logits + float(args.decode_jitter) * torch.randn(
                    logits.shape, dtype=logits.dtype, device=device, generator=generator,
                )
                candidate = joint_decode(
                    trial_logits, decode_edge_index, batch["elems"], total_charge,
                    charge_logits=aux.get("charge_logits"),
                    implicit_h_logits=aux.get("implicit_h_logits"),
                    explicit_h=args.explicit_h,
                    aromatic_constraints=args.aromatic_constraints,
                    forbidden_edge_labelings=forbidden_labelings,
                )
                edges, labels = undirected_labels(decode_edge_index, candidate.labels)
                candidate_mol = build_sanitized_mol(
                    batch["elems"], edges, labels,
                    candidate.formal_charge.detach().cpu().numpy(),
                    None if args.explicit_h else candidate.implicit_h_count.detach().cpu().numpy(), batch["coord"],
                    all_atoms_explicit_h=(args.explicit_h and not args.allow_residual_implicit_h),
                ) if candidate.solved and candidate.kekulizable else None
                result, mol, selected_trial = candidate, candidate_mol, trial
                if candidate_mol is not None:
                    basin_displacement = forcefield_basin_displacement(candidate_mol) if use_basin_ranking else None
                    sanitized_candidates.append((basin_displacement, trial, candidate, candidate_mol))
                    if not use_basin_ranking:
                        accepted = True
                        break
                forbidden_labelings.append(torch.tensor(labels, dtype=torch.long).cpu().numpy())
            if not accepted and args.aromatic_rescue:
                # A repeated no-good cut tends to preserve a high-probability
                # but globally non-Kekulizable aromatic system.  This one
                # explicitly different hypothesis tests whether an alternating
                # single/double graph is chemically acceptable instead.
                n_decode_attempts += 1
                candidate = joint_decode(
                    logits, decode_edge_index, batch["elems"], total_charge,
                    charge_logits=aux.get("charge_logits"),
                    implicit_h_logits=aux.get("implicit_h_logits"),
                    explicit_h=args.explicit_h,
                    aromatic_constraints=args.aromatic_constraints,
                    forbid_aromatic=True,
                )
                edges, labels = undirected_labels(decode_edge_index, candidate.labels)
                candidate_mol = build_sanitized_mol(
                    batch["elems"], edges, labels,
                    candidate.formal_charge.detach().cpu().numpy(),
                    None if args.explicit_h else candidate.implicit_h_count.detach().cpu().numpy(), batch["coord"],
                    all_atoms_explicit_h=(args.explicit_h and not args.allow_residual_implicit_h),
                ) if candidate.solved and candidate.kekulizable else None
                if candidate_mol is not None:
                    result, mol, selected_trial = candidate, candidate_mol, -1
                    accepted = True
                    aromatic_rescue_used = True
            if use_basin_ranking and sanitized_candidates:
                supported = [item for item in sanitized_candidates if item[0] is not None]
                if args.basin_weight is None:
                    selected_basin_displacement, selected_trial, result, mol = min(
                        supported or sanitized_candidates,
                        key=lambda item: (float("inf") if item[0] is None else item[0], item[1]),
                    )
                elif supported:
                    selected_basin_displacement, selected_trial, result, mol = max(
                        supported,
                        key=lambda item: (item[2].objective - args.basin_weight * item[0], -item[1]),
                    )
                else:
                    selected_basin_displacement, selected_trial, result, mol = max(
                        sanitized_candidates, key=lambda item: (item[2].objective, -item[1])
                    )
                accepted = True
            if accepted:
                # Recursive in-painting output directories reuse names such
                # as molecule_0000.xyz, so retain a unique source identity.
                mol.SetProp("_Name", source_relpath.replace("\\", "/"))
                mol.SetIntProp("bondnet_total_charge", total_charge)
                mol.SetProp("bondnet_objective", f"{result.objective:.6f}")
                writer.write(mol)
            error_mask = uncertainty_error_mask(
                logits, decode_edge_index, int(batch["elems"].numel()), args.max_error_atoms
            )
            record = {
                "filename": xyz_path.name,
                "source_relative_path": source_relpath,
                "total_charge": total_charge,
                "n_atoms": int(batch["elems"].numel()),
                "n_heavy_atoms": int((batch["elems"] != 1).sum()),
                "n_explicit_hydrogens": int((batch["elems"] == 1).sum()),
                "explicit_h_mode": bool(args.explicit_h),
                "allow_residual_implicit_h": bool(args.allow_residual_implicit_h),
                "aromatic_constraints": bool(args.aromatic_constraints),
                "aromatic_rescue_used": bool(aromatic_rescue_used),
                "decode_trials": int(args.decode_trials),
                "n_decode_attempts": int(n_decode_attempts),
                "accepted_decode_trial": int(selected_trial) if accepted else None,
                "n_sanitized_graph_hypotheses": len(sanitized_candidates),
                "rank_by_basin": bool(args.rank_by_basin),
                "basin_weight": args.basin_weight,
                "selected_basin_displacement_A": selected_basin_displacement,
                "n_candidate_directed_edges": int(decode_edge_index.shape[0]),
                "joint_solved": bool(result.solved),
                "rdkit_sanitized": bool(accepted),
                "objective": float(result.objective),
                "local_overvalent_atoms": int(result.local_overvalent_atoms),
                "error_mask_atom_indices": error_mask,
            }
            reports.append(record)
            if not accepted:
                # Directly consumable semantic payload for constructing one
                # MolCraft `gen_inpaint` config per rejected XYZ structure.
                inpaint_requests.append({
                    "reference_structure_path": str(xyz_path.resolve()),
                    "condition_component": "xh",
                    "inpaint_cfgs": {"mask_node_index": error_mask, "denoising_strength": 0.8},
                    "reason": "bondnet_reject",
                })
        except Exception as exc:
            reports.append({"filename": xyz_path.name, "source_relative_path": str(xyz_path),
                            "error": str(exc), "rdkit_sanitized": False})
    writer.close()

    (out / "bondnet_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    (out / "molcraft_inpaint_requests.json").write_text(
        json.dumps(inpaint_requests, indent=2), encoding="utf-8"
    )
    accepted = sum(bool(item.get("rdkit_sanitized")) for item in reports)
    print(json.dumps({
        "n_samples": len(reports), "n_accepted": accepted,
        "acceptance_rate": accepted / max(len(reports), 1),
        "report": str((out / "bondnet_report.json").resolve()),
        "valid_sdf": str((valid_dir / "bondnet_valid.sdf").resolve()),
        "inpaint_requests": str((out / "molcraft_inpaint_requests.json").resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
