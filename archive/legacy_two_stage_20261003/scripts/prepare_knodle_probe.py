#!/usr/bin/env python3
"""Prepare topology-free Knodle probes on clean or coordinate-noised SDF inputs.

The emitted SDF retains its reference graph only for later scoring.  The PDB files
given to Knodle are exported with no CONECT records, so the comparator receives only
heavy-atom elements and coordinates.  Use this to evaluate Knodle on the CCD control
and at the same declared coordinate-noise levels as BondNet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference_sdf", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.15])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_mols", type=int, default=None)
    p.add_argument("--allowed_atomic_nums", type=int, nargs="+",
                   default=[6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53, 83])
    return p.parse_args()


def _tag(sigma: float) -> str:
    return f"sigma{sigma:.2f}".replace(".", "p")


def main():
    from rdkit import Chem, RDLogger
    from scripts.p0d_knodle_compare import cmd_export

    cli = parse_args()
    RDLogger.DisableLog("rdApp.*")
    allowed = set(cli.allowed_atomic_nums)
    source = []
    for index, mol in enumerate(Chem.SDMolSupplier(cli.reference_sdf, removeHs=False)):
        if mol is None or mol.GetNumConformers() == 0:
            continue
        mol = Chem.RemoveHs(mol, sanitize=True)
        if not {a.GetAtomicNum() for a in mol.GetAtoms()} <= allowed:
            continue
        source.append((index, mol))
        if cli.max_mols is not None and len(source) >= cli.max_mols:
            break

    root = Path(cli.out_root)
    for sigma in cli.noise:
        directory = root / _tag(sigma)
        directory.mkdir(parents=True, exist_ok=True)
        sdf = directory / "reference_for_scoring.sdf"
        writer = Chem.SDWriter(str(sdf))
        for source_index, mol in source:
            copied = Chem.Mol(mol)
            if sigma:
                rng = np.random.default_rng(cli.seed + source_index * 1009 + round(sigma * 1_000_000))
                conformer = copied.GetConformer()
                xyz = np.asarray(conformer.GetPositions(), dtype=float)
                xyz += rng.normal(0.0, sigma, size=xyz.shape)
                for atom_index, point in enumerate(xyz):
                    conformer.SetAtomPosition(atom_index, point)
            writer.write(copied)
        writer.close()
        pdb_dir = directory / "pdb"
        cmd_export(SimpleNamespace(sdf=str(sdf), out_dir=str(pdb_dir)))
        print(f"prepared sigma={sigma:.2f}: {len(source)} molecules -> {directory}")


if __name__ == "__main__":
    main()
