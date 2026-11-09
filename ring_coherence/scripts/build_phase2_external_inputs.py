"""Download and validate preregistered RCSB instances; do not run predictors."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from rdkit import Chem, rdBase

from bondbench.graphs import heavy_atom_copy


def as_list(value) -> list[str]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def column(cif: dict, name: str, size: int, default: str = "") -> list[str]:
    values = as_list(cif.get(name))
    if not values:
        return [default] * size
    if len(values) != size:
        raise ValueError(f"mmCIF column length mismatch: {name}={len(values)}, expected {size}")
    return values


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def element_symbol(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("empty_element_symbol")
    return text[0].upper() + text[1:].lower()


def download(session: requests.Session, url: str, path: Path, attempts: int = 5) -> None:
    if path.is_file() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(url, timeout=120)
            response.raise_for_status()
            data = response.content
            if len(data) < 100:
                raise ValueError(f"implausibly short response ({len(data)} bytes)")
            path.write_bytes(data)
            return
        except Exception as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed: {url}") from error


def has_covalent_link(cif: dict, label_asym_id: str) -> bool:
    types = as_list(cif.get("_struct_conn.conn_type_id"))
    first = column(cif, "_struct_conn.ptnr1_label_asym_id", len(types)) if types else []
    second = column(cif, "_struct_conn.ptnr2_label_asym_id", len(types)) if types else []
    return any(
        str(kind).lower().startswith("covale") and label_asym_id in {a, b}
        for kind, a, b in zip(types, first, second)
    )


def instance_coordinates(cif: dict, label_asym_id: str, ccd_id: str) -> dict[str, tuple[str, np.ndarray]]:
    xs = as_list(cif.get("_atom_site.Cartn_x"))
    size = len(xs)
    fields = {
        "y": column(cif, "_atom_site.Cartn_y", size),
        "z": column(cif, "_atom_site.Cartn_z", size),
        "element": column(cif, "_atom_site.type_symbol", size),
        "atom": column(cif, "_atom_site.label_atom_id", size),
        "comp": column(cif, "_atom_site.label_comp_id", size),
        "asym": column(cif, "_atom_site.label_asym_id", size),
        "alt": column(cif, "_atom_site.label_alt_id", size, "."),
        "model": column(cif, "_atom_site.pdbx_PDB_model_num", size, "1"),
    }
    rows: dict[str, tuple[str, np.ndarray]] = {}
    for index in range(size):
        if fields["asym"][index] != label_asym_id or fields["comp"][index].upper() != ccd_id:
            continue
        if fields["model"][index] not in {"1", ".", "?"}:
            continue
        raw_element = fields["element"][index].upper()
        if raw_element in {"H", "D", "T"}:
            continue
        element = element_symbol(fields["element"][index])
        alt = fields["alt"][index]
        if alt not in {".", "?"}:
            raise ValueError("nontrivial_alternate_conformation")
        atom_id = fields["atom"][index]
        if atom_id in rows:
            raise ValueError("duplicate_atom_name")
        rows[atom_id] = (
            element,
            np.asarray([float(xs[index]), float(fields["y"][index]), float(fields["z"][index])]),
        )
    if not rows:
        raise ValueError("instance_not_found")
    return rows


def ccd_reference(cif: dict, coords: dict[str, tuple[str, np.ndarray]]) -> Chem.Mol:
    atom_ids = as_list(cif.get("_chem_comp_atom.atom_id"))
    elements = column(cif, "_chem_comp_atom.type_symbol", len(atom_ids))
    charges = column(cif, "_chem_comp_atom.charge", len(atom_ids), "0")
    raw_element_by_id = {name: element.upper() for name, element in zip(atom_ids, elements)}
    element_by_id = {name: element_symbol(element) for name, element in zip(atom_ids, elements)}
    heavy_ids = [name for name in atom_ids if raw_element_by_id[name] not in {"H", "D", "T"}]
    if set(heavy_ids) != set(coords):
        missing = sorted(set(heavy_ids) - set(coords))
        extra = sorted(set(coords) - set(heavy_ids))
        raise ValueError(f"ccd_coordinate_atom_mismatch:missing={missing};extra={extra}")

    atom_charge = {name: int(float(value)) for name, value in zip(atom_ids, charges)}
    first = as_list(cif.get("_chem_comp_bond.atom_id_1"))
    second = column(cif, "_chem_comp_bond.atom_id_2", len(first)) if first else []
    orders = column(cif, "_chem_comp_bond.value_order", len(first)) if first else []
    aromatic = column(cif, "_chem_comp_bond.pdbx_aromatic_flag", len(first), "N") if first else []
    hydrogen_counts: Counter[str] = Counter()
    heavy_bonds = []
    for a, b, order, is_aromatic in zip(first, second, orders, aromatic):
        ea, eb = raw_element_by_id.get(a), raw_element_by_id.get(b)
        if ea in {"H", "D", "T"} and eb not in {"H", "D", "T", None}:
            hydrogen_counts[b] += 1
        elif eb in {"H", "D", "T"} and ea not in {"H", "D", "T", None}:
            hydrogen_counts[a] += 1
        elif ea not in {"H", "D", "T", None} and eb not in {"H", "D", "T", None}:
            heavy_bonds.append((a, b, order.upper(), is_aromatic.upper() == "Y"))

    rw = Chem.RWMol()
    index: dict[str, int] = {}
    aromatic_atoms: set[str] = set()
    charge_by_id = {name: atom_charge.get(name, 0) for name in heavy_ids}
    for name in heavy_ids:
        element, _ = coords[name]
        if element != element_by_id[name]:
            raise ValueError(f"element_mismatch:{name}:{element}:{element_by_id[name]}")
        atom = Chem.Atom(element)
        atom.SetFormalCharge(charge_by_id[name])
        atom.SetNumExplicitHs(hydrogen_counts[name])
        atom.SetNoImplicit(True)
        index[name] = rw.AddAtom(atom)
    order_types = {
        "SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE,
        "TRIP": Chem.BondType.TRIPLE, "AROM": Chem.BondType.AROMATIC,
    }
    for a, b, order, is_aromatic in heavy_bonds:
        bond_type = Chem.BondType.AROMATIC if is_aromatic else order_types.get(order)
        if bond_type is None:
            raise ValueError(f"unsupported_bond_order:{order}")
        rw.AddBond(index[a], index[b], bond_type)
        if bond_type == Chem.BondType.AROMATIC:
            aromatic_atoms.update((a, b))
    mol = rw.GetMol()
    for name in aromatic_atoms:
        mol.GetAtomWithIdx(index[name]).SetIsAromatic(True)
    conformer = Chem.Conformer(len(heavy_ids))
    conformer.Set3D(True)
    for name in heavy_ids:
        conformer.SetAtomPosition(index[name], coords[name][1])
        mol.GetAtomWithIdx(index[name]).SetProp("ccd_atom_id", name)
    mol.AddConformer(conformer)
    Chem.SanitizeMol(mol)
    return mol


def write_molecule(path: Path, mol: Chem.Mol) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()
    if not path.is_file() or not path.stat().st_size:
        raise RuntimeError(f"failed to write {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", type=int, default=500)
    args = parser.parse_args()
    candidates = pd.read_csv(args.selection_dir / "eligible_ccd_representatives.csv")
    candidates = candidates.sort_values(["sample_hash", "ccd_id"])
    session = requests.Session()
    session.headers["User-Agent"] = "BondBench/0.1 preregistered external-input builder"
    raw_pdb = args.output_dir / "raw" / "pdb"
    raw_ccd = args.output_dir / "raw" / "ccd"
    references = args.output_dir / "references"
    inputs = args.output_dir / "inputs"
    records = []
    accepted = 0
    for row in candidates.itertuples(index=False):
        if accepted >= args.target:
            break
        system_id = f"{row.pdb_id}_{row.ccd_id}_{row.label_asym_id}"
        record = {
            "candidate_rank": len(records) + 1, "system_id": system_id,
            "pdb_id": row.pdb_id, "ccd_id": row.ccd_id,
            "label_asym_id": row.label_asym_id, "auth_asym_id": row.auth_asym_id,
            "resolution": row.resolution, "sample_hash": row.sample_hash,
        }
        pdb_path = raw_pdb / f"{row.pdb_id}.cif"
        ccd_path = raw_ccd / f"{row.ccd_id}.cif"
        try:
            download(session, f"https://files.rcsb.org/download/{row.pdb_id}.cif", pdb_path)
            download(session, f"https://files.rcsb.org/ligands/download/{row.ccd_id}.cif", ccd_path)
            pdb_cif = MMCIF2Dict(str(pdb_path))
            if has_covalent_link(pdb_cif, row.label_asym_id):
                raise ValueError("covalently_linked")
            coords = instance_coordinates(pdb_cif, row.label_asym_id, row.ccd_id)
            reference = ccd_reference(MMCIF2Dict(str(ccd_path)), coords)
            reference.SetProp("_Name", system_id)
            reference.SetProp("pdb_id", row.pdb_id)
            reference.SetProp("ccd_id", row.ccd_id)
            reference.SetProp("label_asym_id", row.label_asym_id)
            cloud = heavy_atom_copy(reference)
            reference_path = references / f"{system_id}.sdf"
            input_path = inputs / f"{system_id}.sdf"
            write_molecule(reference_path, reference)
            write_molecule(input_path, cloud)
            accepted += 1
            record.update(
                status="accepted", selection_rank=accepted, heavy_atoms=reference.GetNumAtoms(),
                net_charge=Chem.GetFormalCharge(reference), reference_file=str(reference_path),
                input_file=str(input_path), reference_sha256=sha256(reference_path),
                input_sha256=sha256(input_path), pdb_cif_sha256=sha256(pdb_path),
                ccd_cif_sha256=sha256(ccd_path), error="",
            )
        except Exception as exc:
            record.update(status="excluded", exclusion_reason=str(exc), error=f"{type(exc).__name__}: {exc}")
        records.append(record)
        if len(records) % 50 == 0:
            print(f"attempted={len(records)} accepted={accepted}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(args.output_dir / "selection_audit.csv", index=False)
    final = frame[frame.status == "accepted"].copy()
    final.to_csv(args.output_dir / "final_500.csv", index=False)
    raw_files = sorted((args.output_dir / "raw").rglob("*.cif"))
    raw_manifest = pd.DataFrame([{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in raw_files])
    raw_manifest.to_csv(args.output_dir / "raw_file_manifest.csv", index=False)
    summary = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(), "rdkit": rdBase.rdkitVersion,
        "attempted": len(frame), "accepted": len(final), "target": args.target,
        "status_counts": frame.status.value_counts().to_dict(),
        "exclusion_counts": frame.get("exclusion_reason", pd.Series(dtype=str)).value_counts().to_dict(),
        "models_run": False,
    }
    (args.output_dir / "build_run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if len(final) != args.target:
        raise RuntimeError(f"accepted {len(final)} of target {args.target}")


if __name__ == "__main__":
    main()
