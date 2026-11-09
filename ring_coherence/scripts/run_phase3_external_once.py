"""One-shot paired evaluation of frozen hybrid v1 and structured v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
import scipy
from scipy.stats import binomtest

from bondbench.graphs import bond_order_agreement_counts, compare_graphs
from bondbench.hybrid_ilp import HybridizationILPModel
from bondbench.io import read_first_sdf
from bondbench.stats import bootstrap_binary_ci
from bondbench.structured_ilp import StructuredHybridILPModel
from scripts.run_rdkit_baseline import METRIC_NAMES, metric_dict


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_input_manifest(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for section in ("fixed_files", "sdf_files"):
        for name, expected in manifest[section].items():
            if sha256(Path(name)) != expected:
                raise RuntimeError(f"frozen input checksum mismatch: {name}")


def verify_evaluator_manifest(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if sha256(Path(name)) != expected:
            raise RuntimeError(f"frozen evaluator checksum mismatch: {name}")


def failed_metrics() -> dict[str, bool]:
    return {metric: False for metric in METRIC_NAMES}


def safe_smiles(molecule: Chem.Mol) -> str:
    try:
        return Chem.MolToSmiles(molecule)
    except Exception:
        return ""


def bootstrap_difference(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(20260822)
    indices = rng.integers(0, len(left), size=(10000, len(left)))
    differences = (left[indices] - right[indices]).mean(axis=1)
    return (
        float((left - right).mean()),
        float(np.quantile(differences, 0.025)),
        float(np.quantile(differences, 0.975)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("data/external/phase3_rcsb_2025"))
    parser.add_argument("--input-manifest", type=Path, default=Path("data/manifests/phase3_external_inputs_frozen.json"))
    parser.add_argument("--evaluator-manifest", type=Path, default=Path("data/manifests/phase3_external_evaluator_frozen.json"))
    parser.add_argument("--training-root", type=Path, default=Path("data/processed/posebusters_paper_data/astex_diverse_set"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"one-shot output directory already exists: {args.output_dir}")
    verify_input_manifest(args.input_manifest)
    verify_evaluator_manifest(args.evaluator_manifest)

    final = pd.read_csv(args.input_root / "final_500.csv").sort_values("selection_rank")
    if len(final) != 500 or final.ccd_id.nunique() != 500:
        raise RuntimeError("external set is not 500 unique CCD records")
    training_files = sorted(args.training_root.glob("*/*_ligand.sdf"))
    if len(training_files) != 85:
        raise RuntimeError(f"expected 85 Astex training references, found {len(training_files)}")
    training = [read_first_sdf(path) for path in training_files]
    models = {
        "hybrid_v1": HybridizationILPModel().fit(training),
        "structured_v2": StructuredHybridILPModel(
            ring_reward=1.0, functional_group_constraints=False
        ).fit(training),
    }

    rows: list[dict] = []
    started = time.time()
    for position, record in enumerate(final.itertuples(index=False), start=1):
        reference = read_first_sdf(Path(record.reference_file))
        atom_cloud = read_first_sdf(Path(record.input_file))
        for method, model in models.items():
            base = {
                "system_id": record.system_id,
                "pdb_id": record.pdb_id,
                "ccd_id": record.ccd_id,
                "selection_rank": int(record.selection_rank),
                "resolution": float(record.resolution),
                "heavy_atoms": reference.GetNumAtoms(),
                "method": method,
            }
            try:
                result = model.predict(atom_cloud, total_charge=None)
                prediction = result.molecule
                correct, total = bond_order_agreement_counts(reference, prediction)
                rows.append({
                    **base,
                    "status": "ok",
                    **metric_dict(compare_graphs(reference, prediction)),
                    "bond_labels_correct": correct,
                    "bond_labels_total": total,
                    "pred_smiles": safe_smiles(prediction),
                    "selected_charge": result.selected_charge,
                    "error": "",
                })
            except Exception as exc:
                rows.append({
                    **base,
                    "status": type(exc).__name__,
                    **failed_metrics(),
                    "bond_labels_correct": 0,
                    "bond_labels_total": reference.GetNumBonds(),
                    "pred_smiles": "",
                    "selected_charge": None,
                    "error": str(exc).replace("\n", " ")[:500],
                })
        if position % 25 == 0:
            print(f"evaluated={position}/500 elapsed={time.time() - started:.1f}s", flush=True)

    frame = pd.DataFrame(rows)
    summaries: list[dict] = []
    for method in models:
        group = frame[frame.method == method].sort_values("selection_rank")
        for metric in METRIC_NAMES:
            accuracy, low, high = bootstrap_binary_ci(group[metric].astype(bool).tolist(), replicates=10000)
            summaries.append({
                "method": method,
                "metric": metric,
                "n": len(group),
                "accuracy": accuracy,
                "ci_low": low,
                "ci_high": high,
                "failures": int((group.status != "ok").sum()),
            })
        summaries.append({
            "method": method,
            "metric": "bond_order_micro_f1",
            "n": len(group),
            "accuracy": float(group.bond_labels_correct.sum() / group.bond_labels_total.sum()),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "failures": int((group.status != "ok").sum()),
        })
    summary = pd.DataFrame(summaries)
    v1 = frame[frame.method == "hybrid_v1"].sort_values("selection_rank").l2_isomorphic.astype(bool).to_numpy()
    v2 = frame[frame.method == "structured_v2"].sort_values("selection_rank").l2_isomorphic.astype(bool).to_numpy()
    only_v2 = int((v2 & ~v1).sum())
    only_v1 = int((v1 & ~v2).sum())
    discordant = only_v2 + only_v1
    delta, delta_low, delta_high = bootstrap_difference(v2.astype(float), v1.astype(float))
    paired = {
        "both": int((v2 & v1).sum()),
        "only_v2": only_v2,
        "only_v1": only_v1,
        "neither": int((~v2 & ~v1).sum()),
        "exact_mcnemar_two_sided_p": float(binomtest(only_v2, discordant, 0.5).pvalue) if discordant else 1.0,
        "paired_difference": delta,
        "difference_ci_low": delta_low,
        "difference_ci_high": delta_high,
    }
    if delta >= 0.01 and delta_low > 0 and paired["exact_mcnemar_two_sided_p"] < 0.05:
        decision = "adopt_v2"
    elif delta > 0 and delta_low >= -0.01:
        decision = "retain_inconclusive"
    else:
        decision = "reject_or_redesign"

    args.output_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.output_dir / "per_molecule.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "paired_l2.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    run = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "rdkit": rdBase.rdkitVersion,
        "scipy": scipy.__version__,
        "seconds": round(time.time() - started, 1),
        "records": 500,
        "bootstrap_replicates": 10000,
        "decision": decision,
        "ring_reward": 1.0,
        "functional_group_constraints": False,
        "phase2_external_predictions_run": False,
        "input_manifest_sha256": sha256(args.input_manifest),
        "evaluator_manifest_sha256": sha256(args.evaluator_manifest),
    }
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps({"paired_l2": paired, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
