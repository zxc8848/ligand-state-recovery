"""Build a template-free PDBBind pose set with CCD, not PDBBind, graph truth.

The reader later sees only the heavy-atom elements, PDBBind ligand coordinates,
and CCD net charge.  PDBBind SDF/MOL2 bonds are never used as labels.  The
PDBBind index supplies the CCD component ID and MOL2 atom names align the pose
atoms to the component dictionary atom IDs.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.prepare_ccd_templatefree import fetch_cif, heavy_identity, load_excluded_identities


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdbbind_root", required=True)
    p.add_argument("--index", default=None, help="Defaults to PDBBind general PL index.")
    p.add_argument("--ccd_cache", default="data/ccd_cif_cache")
    p.add_argument("--output_sdf", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--max_mols", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_year", type=int, default=None,
                   help="Optional inclusive PDB deposition-year lower bound.")
    p.add_argument("--max_year", type=int, default=None,
                   help="Optional inclusive PDB deposition-year upper bound.")
    p.add_argument("--min_heavy", type=int, default=4)
    p.add_argument("--max_heavy", type=int, default=80)
    p.add_argument("--exclude_sdf", default="data/diverse_subset_80k.sdf")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--retries", type=int, default=2)
    return p.parse_args()


def index_codes(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#") or "//" not in line:
            continue
        pdb_id = line.split()[0].lower()
        match = re.search(r"\(([A-Za-z0-9]{1,3})\)", line.split("//", 1)[1])
        if match:
            out[pdb_id] = match.group(1).upper()
    return out


def index_years(path: Path) -> dict[str, int]:
    """Read deposition years from the same PDBBind general index."""
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 3 and re.fullmatch(r"[A-Za-z0-9]{4}", fields[0]):
            try:
                out[fields[0].lower()] = int(fields[2])
            except ValueError:
                pass
    return out


def mol2_atoms(path: Path):
    rows, reading = [], False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            reading = True
            continue
        if line.startswith("@<TRIPOS>"):
            if reading:
                break
            continue
        if not reading or not line.strip():
            continue
        field = line.split()
        if len(field) < 6:
            continue
        atom_type = field[5].split(".", 1)[0]
        symbol = re.match(r"[A-Za-z]{1,2}", atom_type)
        if not symbol:
            raise ValueError(f"cannot infer element from MOL2 type {field[5]!r}")
        element = symbol.group(0).capitalize()
        rows.append((field[1], element, tuple(map(float, field[2:5]))))
    if not rows:
        raise ValueError("MOL2 atom block missing")
    return rows


def ccd_data(path: Path):
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    table = MMCIF2Dict(str(path))
    def values(key):
        value = table[key]
        return value if isinstance(value, list) else [value]
    names, elems = values("_chem_comp_atom.atom_id"), values("_chem_comp_atom.type_symbol")
    charges = table.get("_chem_comp_atom.charge", ["0"] * len(names))
    charges = charges if isinstance(charges, list) else [charges]
    atoms = {name: (str(element).capitalize(), int(float(charges[i])) if i < len(charges) and charges[i] not in {"?", "."} else 0)
             for i, (name, element) in enumerate(zip(names, elems))}
    first, second, order = values("_chem_comp_bond.atom_id_1"), values("_chem_comp_bond.atom_id_2"), values("_chem_comp_bond.value_order")
    aromatic = table.get("_chem_comp_bond.pdbx_aromatic_flag", ["N"] * len(first))
    aromatic = aromatic if isinstance(aromatic, list) else [aromatic]
    bonds = [(a, b, str(o).upper(), str(aromatic[i]).upper() == "Y")
             for i, (a, b, o) in enumerate(zip(first, second, order))]
    return atoms, bonds


def pose_with_ccd_truth(pose_atoms, atoms, bonds):
    from rdkit import Chem
    heavy = [(name, element, xyz) for name, element, xyz in pose_atoms if element != "H"]
    if len({name for name, _, _ in heavy}) != len(heavy):
        raise ValueError("duplicate heavy atom name in MOL2")
    names = [name for name, _, _ in heavy]
    if set(names) != {name for name, (element, _) in atoms.items() if element != "H"}:
        raise ValueError("pose/CCD heavy atom-name sets differ")
    rw, index = Chem.RWMol(), {}
    conf = Chem.Conformer(len(heavy))
    for i, (name, element, xyz) in enumerate(heavy):
        ccd_element, charge = atoms[name]
        if element != ccd_element:
            raise ValueError(f"element mismatch for atom {name}: pose={element}, CCD={ccd_element}")
        atom = Chem.Atom(element)
        atom.SetFormalCharge(charge)
        index[name] = rw.AddAtom(atom)
        conf.SetAtomPosition(i, xyz)
    rw.AddConformer(conf, assignId=True)
    type_map = {"SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE,
                "TRIP": Chem.BondType.TRIPLE, "AROM": Chem.BondType.AROMATIC,
                "DELO": Chem.BondType.AROMATIC}
    aromatic_atoms = set()
    for left, right, order, aromatic in bonds:
        if left not in index or right not in index:
            continue
        is_aromatic = aromatic or order in {"AROM", "DELO"}
        rw.AddBond(index[left], index[right], Chem.BondType.AROMATIC if is_aromatic else type_map.get(order, Chem.BondType.SINGLE))
        if is_aromatic:
            aromatic_atoms.update((index[left], index[right]))
    mol = rw.GetMol()
    for i in aromatic_atoms:
        mol.GetAtomWithIdx(i).SetIsAromatic(True)
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.AROMATIC:
            bond.SetIsAromatic(True)
    Chem.SanitizeMol(mol)
    return mol


def main():
    from rdkit import Chem
    opt = args()
    root = Path(opt.pdbbind_root)
    index_path = Path(opt.index) if opt.index else root / "index" / "INDEX_general_PL.2020R1.lst"
    codes = index_codes(index_path)
    years = index_years(index_path)
    candidates = {}
    for path in root.joinpath("P-L").rglob("*_ligand.mol2"):
        candidates[path.parent.name.lower()] = path
    items = [(pdb_id, path, codes[pdb_id]) for pdb_id, path in candidates.items()
             if pdb_id in codes
             and (opt.min_year is None or years.get(pdb_id, -1) >= opt.min_year)
             and (opt.max_year is None or years.get(pdb_id, 10**9) <= opt.max_year)]
    random.Random(opt.seed).shuffle(items)
    excluded = load_excluded_identities(opt.exclude_sdf)
    cache, out = Path(opt.ccd_cache), Path(opt.output_sdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer, rows, written = Chem.SDWriter(str(out)), [], 0
    for pdb_id, path, component_id in items:
        row = {"pdb_id": pdb_id, "year": years.get(pdb_id, ""), "component_id": component_id,
               "source": str(path), "status": "skipped", "reason": ""}
        try:
            source = fetch_cif(component_id, cache, opt.timeout, opt.retries)
            atoms, bonds = ccd_data(source)
            mol = pose_with_ccd_truth(mol2_atoms(path), atoms, bonds)
            n_heavy = mol.GetNumAtoms()
            if not opt.min_heavy <= n_heavy <= opt.max_heavy:
                raise ValueError(f"heavy_atoms={n_heavy}")
            if heavy_identity(mol) in excluded:
                raise ValueError("exact_identity_in_training")
            mol.SetProp("_Name", pdb_id)
            mol.SetProp("pdb_id", pdb_id)
            mol.SetProp("component_id", component_id)
            mol.SetProp("source", "PDBBind_pose__CCD_truth")
            mol.SetIntProp("total_formal_charge", sum(a.GetFormalCharge() for a in mol.GetAtoms()))
            writer.write(mol)
            row.update(status="written", heavy_atoms=n_heavy, total_charge=sum(a.GetFormalCharge() for a in mol.GetAtoms()))
            written += 1
        except Exception as exc:
            row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        if len(rows) % 100 == 0:
            print(f"attempted={len(rows)} written={written}", flush=True)
        if opt.max_mols is not None and written >= opt.max_mols:
            break
    writer.close()
    fields = sorted({key for row in rows for key in row})
    with Path(opt.manifest).open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()
        writer_csv = csv.DictWriter(handle, fieldnames=fields)
        writer_csv.writerows(rows)
    print(f"wrote {written} PDBBind poses with CCD truth -> {out}; attempted {len(rows)}")


if __name__ == "__main__":
    main()
