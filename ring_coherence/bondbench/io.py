from __future__ import annotations

from pathlib import Path

from rdkit import Chem


def read_first_sdf(path: Path) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    for mol in supplier:
        if mol is not None:
            return mol
    raise ValueError(f"no valid molecule in {path}")


def discover_sdf_files(root: Path) -> list[Path]:
    return sorted({*root.rglob("*.sdf"), *root.rglob("*.mol")})

