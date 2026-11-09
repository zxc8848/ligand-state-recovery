"""Identity audit: does the GEOM corpus that the frozen single-stage backbone was
trained on contain molecules of the PDBBind pose benchmark?

The published pose benchmark is identity-disjoint from Diverse-80k, which is what
the archived two-stage model was trained on. The frozen backbone of the joint
pipeline (``revision_v3_one_stage_heavy_seed42``) was instead trained on
``geom_drugs_all_random1_rel10.sdf``, so disjointness has to be re-established
against that corpus before any pose number from the joint pipeline is called
held-out.

Pass 1 is a pure-text heavy-atom formula filter over the 1.1 GB SDF (no RDKit, no
sanitization); pass 2 canonicalizes only the formula matches and reports two
identity rules:

  exact      non-isomeric canonical SMILES of the heavy-atom graph
             (bond orders, formal charges and attached-H counts included);
  skeleton   element + connectivity only, the rule the joint split uses
             (blinded to charge, H, bond order and stereochemistry).

Run from the repository root:

    python experiments/prob_mask_20261002/joint_pose/audit_geom_pose_overlap.py
"""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def parse_args():
    a = argparse.ArgumentParser()
    a.add_argument('--geom', default=str(ROOT/'data/geom_drugs_all_random1_rel10.sdf'))
    a.add_argument('--poses', default=str(ROOT/'data/pdbbind_ccd_pose500.sdf'))
    a.add_argument('--output', default=str(Path(__file__).resolve().parent/'geom_overlap_names.json'))
    a.add_argument('--keep_hits_sdf', default=None,
                   help='optional path to write the formula-matched GEOM records to')
    return a.parse_args()


def heavy_formula_from_block(block):
    na = int(block[3][0:3])
    counts = collections.Counter()
    for row in block[4:4+na]:
        sym = row[31:34].strip()
        if sym != 'H':
            counts[sym] += 1
    return ''.join(f'{k}{v}' for k, v in sorted(counts.items()))


def main():
    a = parse_args()
    sys.path.insert(0, str(ROOT/'experiments/prob_mask_20261002'))
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    from joint import skeleton
    from probe import _sanitize_with_mdl_aromaticity

    def keys(mol):
        m = Chem.RemoveHs(_sanitize_with_mdl_aromaticity(mol))
        return Chem.MolToSmiles(m, isomericSmiles=False), skeleton(m)

    poses = {}
    formulas = set()
    for mol in Chem.SDMolSupplier(a.poses, removeHs=False):
        if mol is None:
            continue
        name = mol.GetProp('_Name') if mol.HasProp('_Name') else None
        counts = collections.Counter(at.GetSymbol() for at in mol.GetAtoms()
                                    if at.GetAtomicNum() != 1)
        formulas.add(''.join(f'{k}{v}' for k, v in sorted(counts.items())))
        try:
            poses[name] = keys(mol)
        except Exception:
            continue
    print(f'poses: {len(poses)} with identity keys, {len(formulas)} distinct heavy formulas')

    hits = []
    kept = open(a.keep_hits_sdf, 'w') if a.keep_hits_sdf else None
    block = []
    scanned = 0
    with open(a.geom, 'r', errors='replace') as fh:
        for line in fh:
            block.append(line)
            if line.startswith('$$$$'):
                scanned += 1
                try:
                    if heavy_formula_from_block(block) in formulas:
                        hits.append(''.join(block))
                        if kept:
                            kept.writelines(block)
                except Exception:
                    pass
                block = []
                if scanned % 50000 == 0:
                    print(f'  pass 1: {scanned} scanned, {len(hits)} formula matches', flush=True)
    if kept:
        kept.close()
    print(f'pass 1 done: {scanned} GEOM records, {len(hits)} formula matches')

    geom_exact, geom_skel = set(), set()
    for i, text in enumerate(hits):
        mol = Chem.MolFromMolBlock(text, sanitize=False, removeHs=False)
        if mol is None:
            continue
        try:
            exact, skel = keys(mol)
        except Exception:
            continue
        geom_exact.add(exact); geom_skel.add(skel)
        if (i+1) % 5000 == 0:
            print(f'  pass 2: {i+1}/{len(hits)}', flush=True)

    over_exact = sorted(n for n, (e, _) in poses.items() if e in geom_exact)
    over_skel = sorted(n for n, (_, s) in poses.items() if s in geom_skel)
    print(f'\noverlap with the frozen backbone training corpus:')
    print(f'  exact heavy-graph identity : {len(over_exact)}/{len(poses)}')
    print(f'  element+connectivity rule  : {len(over_skel)}/{len(poses)}')
    Path(a.output).write_text(json.dumps(
        {'geom_records_scanned': scanned, 'formula_matches': len(hits),
         'n_poses': len(poses), 'exact_smiles': over_exact, 'skeleton': over_skel},
        indent=1), encoding='utf-8')
    print(f'wrote {a.output}')


if __name__ == '__main__':
    main()
