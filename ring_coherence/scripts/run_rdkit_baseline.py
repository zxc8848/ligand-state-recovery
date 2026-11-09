from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from rdkit import Chem, rdBase

from bondbench.graphs import (
    GraphMetrics,
    compare_connectivity,
    compare_graphs,
    determine_rdkit,
    determine_rdkit_connectivity,
    heavy_atom_copy,
    heavy_reference,
    successful_charge_candidates,
)
from bondbench.io import discover_sdf_files, read_first_sdf
from bondbench.stats import bootstrap_binary_ci


SYSTEM_ID = re.compile(r"(?i)([0-9][a-z0-9]{3}_[a-z0-9]{3})")
METRIC_NAMES = [
    "l1_connectivity",
    "l2_strict",
    "l2_aromatic",
    "l2_isomorphic",
    "l3_full",
    "l3_isomorphic",
]


def infer_system_id(path: Path) -> str | None:
    match = SYSTEM_ID.search(str(path))
    return match.group(1).upper() if match else None


def choose_reference_files(root: Path, allowed_ids: set[str] | None) -> list[tuple[str, Path]]:
    choices: dict[str, list[Path]] = {}
    for path in discover_sdf_files(root):
        system_id = infer_system_id(path)
        if system_id is None or (allowed_ids is not None and system_id not in allowed_ids):
            continue
        lower = str(path).lower()
        if any(token in lower for token in ("pred", "output", "generated")):
            continue
        choices.setdefault(system_id, []).append(path)

    selected: list[tuple[str, Path]] = []
    for system_id, paths in choices.items():
        ranked = sorted(
            paths,
            key=lambda p: (
                "ligand" not in p.name.lower(),
                "true" not in p.name.lower(),
                len(str(p)),
                str(p),
            ),
        )
        selected.append((system_id, ranked[0]))
    return sorted(selected)


def metric_dict(metrics: GraphMetrics) -> dict[str, bool]:
    return {name: bool(getattr(metrics, name)) for name in METRIC_NAMES}


def failed_metrics() -> dict[str, bool]:
    return {name: False for name in METRIC_NAMES}


def connectivity_metrics(value: bool) -> dict[str, bool | None]:
    return {
        "l1_connectivity": value,
        "l2_strict": None,
        "l2_aromatic": None,
        "l2_isomorphic": None,
        "l3_full": None,
        "l3_isomorphic": None,
    }


def evaluate_one(system_id: str, path: Path, charge_min: int, charge_max: int) -> list[dict]:
    source = read_first_sdf(path)
    reference = heavy_reference(source)
    atom_cloud = heavy_atom_copy(source)
    total_charge = int(Chem.GetFormalCharge(reference))
    base = {
        "system_id": system_id,
        "reference_file": str(path.resolve()),
        "heavy_atoms": reference.GetNumAtoms(),
        "reference_charge": total_charge,
        "rdkit_version": rdBase.rdkitVersion,
    }
    rows: list[dict] = []

    for track, use_vdw in (("connectivity_ctd", False), ("connectivity_vdw", True)):
        row = {**base, "track": track, "supplied_charge": None}
        try:
            prediction = determine_rdkit_connectivity(atom_cloud, use_vdw=use_vdw)
            row.update(connectivity_metrics(compare_connectivity(reference, prediction)))
            row.update(status="ok", error_type="", error_message="")
        except Exception as exc:
            row.update(failed_metrics())
            row.update(
                status="error",
                error_type=type(exc).__name__,
                error_message=str(exc).replace("\n", " ")[:500],
            )
        rows.append(row)

    for track, supplied_charge in (("known", total_charge), ("neutral", 0)):
        row = {**base, "track": track, "supplied_charge": supplied_charge}
        try:
            prediction = determine_rdkit(atom_cloud, supplied_charge)
            row.update(metric_dict(compare_graphs(reference, prediction)))
            row.update(status="ok", error_type="", error_message="")
        except Exception as exc:
            row.update(failed_metrics())
            row.update(
                status="error",
                error_type=type(exc).__name__,
                error_message=str(exc).replace("\n", " ")[:500],
            )
        rows.append(row)

    candidates = successful_charge_candidates(atom_cloud, range(charge_min, charge_max + 1))
    scored = [
        (charge, compare_graphs(reference, prediction))
        for charge, prediction in candidates.items()
    ]
    row = {
        **base,
        "track": "charge_oracle",
        "supplied_charge": None,
        "candidate_count": len(scored),
    }
    if scored:
        charge, best = max(
            scored,
            key=lambda item: (
                item[1].l3_full,
                item[1].l2_aromatic,
                item[1].l2_isomorphic,
                item[1].l2_strict,
                item[1].l1_connectivity,
                -abs(item[0]),
            ),
        )
        row["oracle_selected_charge"] = charge
        row.update(metric_dict(best))
        row.update(status="diagnostic_oracle", error_type="", error_message="")
    else:
        row.update(failed_metrics())
        row.update(
            oracle_selected_charge=None,
            status="error",
            error_type="NoChargeCandidate",
            error_message="no valid RDKit result in predeclared charge range",
        )
    rows.append(row)
    return rows


def load_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip().upper()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--charge-min", type=int, default=-3)
    parser.add_argument("--charge-max", type=int, default=3)
    args = parser.parse_args()

    allowed_ids = load_ids(args.ids_file)
    files = choose_reference_files(args.input_root, allowed_ids)
    if not files:
        raise SystemExit("no reference SDF/MOL files discovered")

    rows: list[dict] = []
    for number, (system_id, path) in enumerate(files, start=1):
        try:
            rows.extend(evaluate_one(system_id, path, args.charge_min, args.charge_max))
        except Exception as exc:
            base = {
                "system_id": system_id,
                "reference_file": str(path.resolve()),
                "heavy_atoms": None,
                "reference_charge": None,
                "rdkit_version": rdBase.rdkitVersion,
            }
            for track in (
                "connectivity_ctd",
                "connectivity_vdw",
                "known",
                "neutral",
                "charge_oracle",
            ):
                rows.append(
                    {
                        **base,
                        "track": track,
                        "supplied_charge": None,
                        **failed_metrics(),
                        "status": "input_error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc).replace("\n", " ")[:500],
                    }
                )
        print(f"[{number}/{len(files)}] {system_id}", flush=True)

    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "per_molecule.csv", index=False)

    summary_rows: list[dict] = []
    for track, group in frame.groupby("track", sort=True):
        summary = {
            "track": track,
            "molecules": int(len(group)),
            "errors": int((group["status"].isin(["error", "input_error"])).sum()),
        }
        for metric in METRIC_NAMES:
            applicable = group.loc[group[metric].notna(), metric].astype(bool).tolist()
            mean, low, high = bootstrap_binary_ci(applicable)
            summary[f"{metric}_applicable"] = len(applicable)
            summary[f"{metric}_accuracy"] = mean
            summary[f"{metric}_ci_low"] = low
            summary[f"{metric}_ci_high"] = high
        summary_rows.append(summary)
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)

    metadata = {
        "input_root": str(args.input_root.resolve()),
        "ids_file": str(args.ids_file.resolve()) if args.ids_file else None,
        "discovered_molecules": len(files),
        "charge_range": [args.charge_min, args.charge_max],
        "rdkit_version": rdBase.rdkitVersion,
        "oracle_is_deployable": False,
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
