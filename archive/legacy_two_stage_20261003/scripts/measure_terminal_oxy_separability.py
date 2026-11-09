#!/usr/bin/env python3
r"""Table 1 / Figure 3: separability of terminal X--O double vs single bonds.

Measures Cohen's d between the double-bond and single-bond length populations for
symmetric terminal-oxy groups (carboxyl C--O, phosphate P--O) and for aromatic
ring C:C bonds, on CCD ideal coordinates vs real PDBBind poses. In real poses the
two populations coincide (d ~ 0), which is the information-theoretic reason the
single/double assignment is not recoverable from (Z, R, Q).

Usage:
  python scripts/measure_terminal_oxy_separability.py \
      --ideal_sdf data/ccd_heavy_leakagefree.sdf \
      --pose_sdf  data/pdbbind_ccd_pose500.sdf \
      --output    results/terminal_oxy_cohend.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ideal_sdf", required=True)
    p.add_argument("--pose_sdf", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--max_mols", type=int, default=None)
    p.add_argument("--allowed_atomic_nums", type=int, nargs="+",
                   default=[6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53, 83])
    return p.parse_args()


def cohens_d(a, b):
    import numpy as np

    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = ((a.var(ddof=1) + b.var(ddof=1)) / 2.0) ** 0.5
    if pooled == 0:
        return float("inf")
    return abs(a.mean() - b.mean()) / pooled


def collect(path, allowed, max_mols):
    """Return length lists keyed by (group, order)."""
    import numpy as np
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    buckets = {("carboxyl_CO", 1): [], ("carboxyl_CO", 2): [],
               ("phosphate_PO", 1): [], ("phosphate_PO", 2): [],
               ("aromatic_CC", "single"): [], ("aromatic_CC", "double"): []}
    n = 0
    for mol in Chem.SDMolSupplier(path, removeHs=False):
        if mol is None or mol.GetNumConformers() == 0:
            continue
        try:
            mol = Chem.RemoveHs(mol, sanitize=True)
        except Exception:
            continue
        if not {a.GetAtomicNum() for a in mol.GetAtoms()} <= set(allowed):
            continue
        conf = mol.GetConformer()
        pos = {a.GetIdx(): np.asarray(conf.GetAtomPosition(a.GetIdx())) for a in mol.GetAtoms()}
        degree = {a.GetIdx(): a.GetDegree() for a in mol.GetAtoms()}

        # Symmetric terminal-oxy centres (C -> carboxyl, P -> phosphate).
        for atom in mol.GetAtoms():
            z = atom.GetAtomicNum()
            if z not in (6, 15):
                continue
            term = [nb for nb in atom.GetNeighbors()
                    if nb.GetAtomicNum() == 8 and degree[nb.GetIdx()] == 1]
            if len(term) < 2:
                continue
            key = "carboxyl_CO" if z == 6 else "phosphate_PO"
            for nb in term:
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx())
                order = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2}.get(bond.GetBondType())
                if order is None:
                    continue
                length = float(np.linalg.norm(pos[atom.GetIdx()] - pos[nb.GetIdx()]))
                buckets[(key, order)].append(length)

        # Aromatic ring C:C bonds, split by their Kekule single/double assignment.
        try:
            kek = Chem.Mol(mol)
            Chem.Kekulize(kek, clearAromaticFlags=True)
        except Exception:
            kek = None
        for bond in mol.GetBonds():
            if not bond.GetIsAromatic():
                continue
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if mol.GetAtomWithIdx(i).GetAtomicNum() != 6 or mol.GetAtomWithIdx(j).GetAtomicNum() != 6:
                continue
            length = float(np.linalg.norm(pos[i] - pos[j]))
            side = "single"
            if kek is not None:
                kb = kek.GetBondBetweenAtoms(i, j)
                if kb is not None and kb.GetBondTypeAsDouble() == 2:
                    side = "double"
            buckets[("aromatic_CC", side)].append(length)

        n += 1
        if max_mols and n >= max_mols:
            break
    return buckets, n


def auc_shorter_is_double(single, double):
    """AUC of the classifier ``shorter bond -> double''. 1=perfect, 0.5=chance."""
    import numpy as np
    if len(single) < 1 or len(double) < 1:
        return None
    s = np.asarray(single)
    wins = ties = 0
    for x in double:
        wins += int(np.sum(x < s))
        ties += int(np.sum(x == s))
    return (wins + 0.5 * ties) / (len(single) * len(double))


def summarize(buckets):
    import numpy as np

    out = {}
    for group, single_key, double_key in [
        ("carboxyl_CO", ("carboxyl_CO", 1), ("carboxyl_CO", 2)),
        ("phosphate_PO", ("phosphate_PO", 1), ("phosphate_PO", 2)),
        ("aromatic_CC", ("aromatic_CC", "single"), ("aromatic_CC", "double")),
    ]:
        s, d = buckets[single_key], buckets[double_key]
        out[group] = {
            "single_mean": float(np.mean(s)) if s else None,
            "single_sd": float(np.std(s, ddof=1)) if len(s) > 1 else None,
            "single_n": len(s),
            "double_mean": float(np.mean(d)) if d else None,
            "double_sd": float(np.std(d, ddof=1)) if len(d) > 1 else None,
            "double_n": len(d),
            "cohens_d": cohens_d(d, s),
            "abs_cohens_d": abs(cohens_d(d, s)) if cohens_d(d, s) is not None else None,
            "auc_shorter_is_double": auc_shorter_is_double(s, d),
        }
    return out


def main():
    args = parse_args()
    report = {}
    for label, path in (("ideal", args.ideal_sdf), ("pose", args.pose_sdf)):
        buckets, n = collect(path, args.allowed_atomic_nums, args.max_mols)
        report[label] = {"n_molecules": n, "groups": summarize(buckets)}

    print(f"{'group':<14}{'ideal d':>10}{'pose d':>10}")
    for group in ("aromatic_CC", "carboxyl_CO", "phosphate_PO"):
        di = report["ideal"]["groups"][group]["cohens_d"]
        dp = report["pose"]["groups"][group]["cohens_d"]
        di_s = f"{di:.2f}" if di is not None else "n/a"
        dp_s = f"{dp:.2f}" if dp is not None else "n/a"
        print(f"{group:<14}{di_s:>10}{dp_s:>10}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
