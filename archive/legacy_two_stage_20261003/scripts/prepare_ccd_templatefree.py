"""Build paired template-free CCD benchmarks from RCSB component CIFs.

Ground-truth bonds are retained only in the output SDF for scoring.  Downstream
readers receive only element types, 3D coordinates, and total formal charge.
The default export removes H for the deployment-relevant heavy-atom task;
``--keep_hydrogens`` preserves CCD ideal H coordinates as an explicitly marked
optimistic upper-bound protocol.
The script downloads one official CCD CIF per component ID, avoiding the very
large monolithic ``components.cif.gz`` download.
"""

from __future__ import annotations

import argparse
import csv
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--component_ids", help="Text/CSV file; first field of each non-comment row is a CCD ID.")
    source.add_argument("--ids", nargs="+", help="CCD IDs supplied directly.")
    p.add_argument("--cache_dir", default="data/ccd_cif_cache")
    p.add_argument("--output_sdf", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--max_components", type=int, default=None)
    p.add_argument("--min_heavy", type=int, default=4)
    p.add_argument("--max_heavy", type=int, default=80)
    p.add_argument("--keep_hydrogens", action="store_true",
                   help="Preserve CCD explicit H atoms and ideal coordinates (upper-bound protocol).")
    p.add_argument("--exclude_sdf", default=None,
                   help="Optional training SDF; exclude exact canonical heavy-atom identities found in it.")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--retries", type=int, default=2)
    return p.parse_args()


def read_ids(args):
    if args.ids:
        return [x.strip().upper() for x in args.ids if x.strip()]
    ids = []
    for line in Path(args.component_ids).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line.replace(",", " ").split()[0].upper())
    return ids


def fetch_cif(component_id: str, cache_dir: Path, timeout: float, retries: int):
    path = cache_dir / f"{component_id}.cif"
    if path.exists() and path.stat().st_size:
        return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://files.rcsb.org/ligands/download/{component_id}.cif"
    last = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "BondNet2-templatefree-benchmark/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            if not payload:
                raise RuntimeError("empty response")
            path.write_bytes(payload)
            return path
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last = exc
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"download failed for {component_id}: {last}")


def values(table, key):
    value = table.get(key)
    if value is None:
        raise KeyError(key)
    return value if isinstance(value, list) else [value]


def number(value):
    if value in {"?", ".", None}:
        raise ValueError("missing coordinate")
    return float(value)


def cif_to_mol(path: Path):
    """Parse one CCD CIF with BioPython, retaining its ideal 3D coordinates."""
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from rdkit import Chem

    table = MMCIF2Dict(str(path))
    atom_id = values(table, "_chem_comp_atom.atom_id")
    symbol = values(table, "_chem_comp_atom.type_symbol")
    charge = table.get("_chem_comp_atom.charge", ["0"] * len(atom_id))
    charge = charge if isinstance(charge, list) else [charge]
    x_key = "_chem_comp_atom.pdbx_model_Cartn_x_ideal"
    y_key = "_chem_comp_atom.pdbx_model_Cartn_y_ideal"
    z_key = "_chem_comp_atom.pdbx_model_Cartn_z_ideal"
    if x_key not in table:
        x_key, y_key, z_key = ("_chem_comp_atom.model_Cartn_x", "_chem_comp_atom.model_Cartn_y",
                               "_chem_comp_atom.model_Cartn_z")
    xs, ys, zs = values(table, x_key), values(table, y_key), values(table, z_key)
    if not (len(atom_id) == len(symbol) == len(xs) == len(ys) == len(zs)):
        raise ValueError("inconsistent atom loop lengths")
    rw = Chem.RWMol()
    index = {}
    for idx, (name, element) in enumerate(zip(atom_id, symbol)):
        atom = Chem.Atom(str(element).capitalize())
        atom.SetFormalCharge(int(float(charge[idx])) if idx < len(charge) and charge[idx] not in {"?", "."} else 0)
        index[name] = rw.AddAtom(atom)
    conf = Chem.Conformer(len(atom_id))
    for idx in range(len(atom_id)):
        conf.SetAtomPosition(idx, (number(xs[idx]), number(ys[idx]), number(zs[idx])))
    rw.AddConformer(conf, assignId=True)

    first = values(table, "_chem_comp_bond.atom_id_1")
    second = values(table, "_chem_comp_bond.atom_id_2")
    orders = values(table, "_chem_comp_bond.value_order")
    aromatic = table.get("_chem_comp_bond.pdbx_aromatic_flag", ["N"] * len(first))
    aromatic = aromatic if isinstance(aromatic, list) else [aromatic]
    order_map = {"SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE,
                 "TRIP": Chem.BondType.TRIPLE, "AROM": Chem.BondType.AROMATIC,
                 "DELO": Chem.BondType.AROMATIC}
    for idx, (a, b, order) in enumerate(zip(first, second, orders)):
        if a not in index or b not in index:
            continue
        is_aromatic = str(order).upper() in {"AROM", "DELO"} or (idx < len(aromatic) and aromatic[idx] == "Y")
        rw.AddBond(index[a], index[b], Chem.BondType.AROMATIC if is_aromatic else order_map.get(str(order).upper(), Chem.BondType.SINGLE))
        bond = rw.GetBondBetweenAtoms(index[a], index[b])
        if is_aromatic:
            bond.SetIsAromatic(True)
            rw.GetAtomWithIdx(index[a]).SetIsAromatic(True)
            rw.GetAtomWithIdx(index[b]).SetIsAromatic(True)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def heavy_identity(mol):
    """Exact chemistry identity used to exclude training-set duplicates.

    Stereochemistry is retained.  This is an exact-identity guardrail, not a
    scaffold-level split claim.
    """
    from rdkit import Chem
    heavy = Chem.RemoveHs(Chem.Mol(mol), sanitize=True)
    return Chem.MolToSmiles(heavy, canonical=True, isomericSmiles=True)


def load_excluded_identities(path: str | None):
    if path is None:
        return set()
    from rdkit import Chem
    identities = set()
    for mol in Chem.SDMolSupplier(str(path), removeHs=False):
        if mol is None:
            continue
        try:
            identities.add(heavy_identity(mol))
        except Exception:
            continue
    return identities


def main():
    from rdkit import Chem

    args = parse_args()
    ids = list(dict.fromkeys(read_ids(args)))
    if args.max_components is not None:
        ids = ids[:args.max_components]
    cache = Path(args.cache_dir)
    output = Path(args.output_sdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output))
    excluded = load_excluded_identities(args.exclude_sdf)
    if args.exclude_sdf:
        print(f"loaded {len(excluded)} exact heavy-atom identities for leakage exclusion", flush=True)
    rows = []
    for number_index, component_id in enumerate(ids, 1):
        row = {"component_id": component_id, "status": "skipped", "reason": ""}
        try:
            source = fetch_cif(component_id, cache, args.timeout, args.retries)
            mol = cif_to_mol(source)
            if heavy_identity(mol) in excluded:
                row["reason"] = "exact_identity_in_exclude_sdf"
                rows.append(row)
                continue
            if not args.keep_hydrogens:
                mol = Chem.RemoveHs(mol, sanitize=True)
            heavy = sum(atom.GetAtomicNum() != 1 for atom in mol.GetAtoms())
            if not args.min_heavy <= heavy <= args.max_heavy:
                row.update(status="skipped", reason=f"heavy_atoms={heavy}")
            else:
                mol.SetProp("_Name", component_id)
                mol.SetProp("source", "RCSB_CCD")
                mol.SetProp("component_id", component_id)
                mol.SetIntProp("total_formal_charge", sum(a.GetFormalCharge() for a in mol.GetAtoms()))
                writer.write(mol)
                row.update(status="written", heavy_atoms=heavy,
                           explicit_hydrogens=sum(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()),
                           total_charge=sum(a.GetFormalCharge() for a in mol.GetAtoms()))
        except Exception as exc:
            row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        if number_index % 50 == 0 or number_index == len(ids):
            print(f"processed {number_index}/{len(ids)}; written={sum(r['status'] == 'written' for r in rows)}", flush=True)
    writer.close()
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=fields)
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    print(f"wrote {sum(r['status'] == 'written' for r in rows)}/{len(rows)} molecules -> {output}")


if __name__ == "__main__":
    main()
