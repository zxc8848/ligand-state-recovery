"""New development identities, never reusing the old 23 charge-separated cases."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from rdkit import Chem, RDLogger

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'experiments/charged_benchmark_20261102'))
import study as S

GROUPS = S.GROUPS
ALLOWED = {6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53, 83}
SOURCE = ROOT/'data/ccd_components_pub_20261102.sdf.gz'


def classify(mol):
    charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
    Q = sum(charges)
    if Q: return 'cation' if Q > 0 else 'anion'
    positive = [i for i, q in enumerate(charges) if q > 0]
    negative = [i for i, q in enumerate(charges) if q < 0]
    if not positive and not negative: return 'uncharged_neutral'
    d = Chem.GetDistanceMatrix(mol)
    return 'separated_neutral' if min(d[i, j] for i in positive for j in negative) >= 3 else 'local_charge_separation'


def main():
    RDLogger.DisableLog('rdApp.*')
    if (HERE/'DATASET.json').exists():
        raise RuntimeError('Existing cohort: verify/reuse it; never silently select again')
    old_source = S.SOURCES
    exclusion = ROOT/'data/locked_testset_excluded_identities.json'
    old_sdfs = [ROOT/p for p in ('data/ccd_heavy_leakagefree.sdf', 'data/pdbbind_ccd_pose500.sdf',
                 'data/pdbbind_pose_locked500.sdf',
                 'work/paper2_noise_ccd_probe/ccd_sigma0p00/reference_for_scoring.sdf')]
    train_sdf = ROOT/'data/diverse_subset_80k.sdf'
    old270 = ROOT/'experiments/charged_benchmark_20261102/selected.pt'
    inputs = [SOURCE, old_source, exclusion, *old_sdfs, train_sdf, old270, Path(__file__)]
    fingerprints = {str(p.relative_to(ROOT)): S.sha(p) for p in inputs}
    spec = {'scope': 'new CCD ideal development pool; no model outcome selection',
            'source_url': 'https://files.wwpdb.org/pub/pdb/data/monomers/components-pub.sdf.gz',
            'eligibility': 'one component, closed shell, 3..60 supported heavy atoms, q -2..2, h 0..4, native roundtrip, finite ideal coordinates',
            'exclusions': 'full saved GEOM/Diverse/development identity union; identity+all-single connectivity skeleton of entire encoder Diverse80k, direct head train+val, all previous CCD/pose evaluation inputs and old charge270; also old CCD component IDs',
            'group_rule': 'sign of Q; Q0 separated min positive-negative heavy graph distance>=3; all-zero-q neutral; exclude local charge separation',
            'selection': 'group-wise rank sha256(atom-candidates-20261102|plain identity); 60% development up to100; remaining up to100 held back; no predictions on held-back records',
            'minimum_separated_development': 10,
            'not_claimed': ['scaffold-disjoint', 'real bound poses', 'measured protonation', 'new confirmatory benchmark'],
            'input_sha256': fingerprints}
    S.save(HERE/'PREPARATION_PLAN.json', spec)
    keys = json.loads(exclusion.read_text())
    identities = set(keys['plain']) | set(keys['stereo']); skeletons = set(); old_ids = set()
    for split_rows in torch.load(old_source, weights_only=False, map_location='cpu').values():
        for row in split_rows:
            identities.add(row['smiles']); skeletons.add(row['skeleton'])
    for row in torch.load(old270, weights_only=False, map_location='cpu'):
        identities.add(row['smiles']); skeletons.add(row['skeleton'])
    for path in [*old_sdfs, train_sdf]:
        count = 0
        for mol in Chem.SDMolSupplier(str(path), removeHs=False):
            if mol is None: continue
            try:
                heavy = Chem.RemoveHs(mol)
                identities.update((Chem.MolToSmiles(heavy, isomericSmiles=False), Chem.MolToSmiles(heavy, isomericSmiles=True)))
                skeletons.add(S.J.skeleton(heavy))
                if path == old_sdfs[0] and mol.HasProp('_Name'): old_ids.add(mol.GetProp('_Name'))
                count += 1
            except Exception: continue
        print('indexed exclusions', path.name, count, flush=True)
    pools = {g: [] for g in GROUPS}; rejected = collections.Counter(); seen = set(); scanned = 0
    with gzip.open(SOURCE, 'rb') as handle:
        for idx, mol in enumerate(Chem.ForwardSDMolSupplier(handle, removeHs=False)):
            scanned += 1
            if scanned % 10000 == 0: print('scanned', scanned, 'eligible', {g: len(p) for g, p in pools.items()}, flush=True)
            if mol is None: rejected['unparsed'] += 1; continue
            try:
                component = mol.GetProp('_Name').strip()
                if component in old_ids: rejected['old_CCD_component_id'] += 1; continue
                heavy = Chem.RemoveHs(Chem.Mol(mol))
                if not 3 <= heavy.GetNumAtoms() <= 60: rejected['size'] += 1; continue
                if not {a.GetAtomicNum() for a in heavy.GetAtoms()} <= ALLOWED: rejected['element'] += 1; continue
                if len(Chem.GetMolFrags(heavy)) != 1: rejected['fragments'] += 1; continue
                if any(a.GetNumRadicalElectrons() for a in heavy.GetAtoms()): rejected['radicals'] += 1; continue
                plain = Chem.MolToSmiles(heavy, isomericSmiles=False)
                stereo = Chem.MolToSmiles(heavy, isomericSmiles=True)
                if plain in identities or stereo in identities: rejected['prior_identity'] += 1; continue
                skeleton = S.J.skeleton(heavy)
                if skeleton in skeletons: rejected['prior_connectivity_skeleton'] += 1; continue
                if plain in seen: rejected['duplicate_identity'] += 1; continue
                seen.add(plain)
                group = classify(heavy)
                if group not in GROUPS: rejected['local_charge_separation'] += 1; continue
                row = S.J.make_row(mol, 'atom_dev_ccd_'+component, idx, 'fresh_ccd_ideal')
                assert row['smiles'] == plain
                if not np.isfinite(row['coord'].numpy()).all(): rejected['nonfinite_coord'] += 1; continue
                row.update(bucket=group, component_id=component)
                rank = hashlib.sha256(('atom-candidates-20261102|'+plain).encode()).hexdigest()
                pools[group].append((rank, row, heavy))
            except Exception as exc:
                rejected['reference_validation:'+str(exc).split('\n')[0][:90]] += 1
    # Keep skeleton variants together so development/held-back are disjoint too.
    splits = {'development': [], 'held_back': []}; export = {'development': [], 'held_back': []}
    skeleton_split = {}
    for group in GROUPS:
        pool = sorted(pools[group], key=lambda x: x[0])
        ndev = min(100, max(1, int(.6*len(pool)))) if pool else 0
        for _, row, mol in pool:
            split = skeleton_split.get(row['skeleton'])
            if split is None:
                split = 'development' if sum(r['bucket'] == group for r in splits['development']) < ndev else 'held_back'
                skeleton_split[row['skeleton']] = split
            if sum(r['bucket'] == group for r in splits[split]) >= 100: continue
            splits[split].append(row); export[split].append(mol)
    assert not ({r['skeleton'] for r in splits['development']} & {r['skeleton'] for r in splits['held_back']})
    for split, rows in splits.items():
        torch.save(rows, HERE/f'{split}.pt')
        writer = Chem.SDWriter(str(HERE/f'{split}.sdf'))
        for row, mol in zip(rows, export[split]):
            mol.SetProp('_Name', row['id']); writer.write(mol)
        writer.close()
    assert fingerprints == {str(p.relative_to(ROOT)): S.sha(p) for p in inputs}
    result = {'scope': spec['scope'], 'preparation_plan_sha256': S.sha(HERE/'PREPARATION_PLAN.json'),
              'source_records_scanned': scanned, 'pool_counts': {g: len(p) for g, p in pools.items()},
              'split_counts': {split: dict(collections.Counter(r['bucket'] for r in rows)) for split, rows in splits.items()},
              'rejections': dict(rejected), 'excluded_identity_count': len(identities), 'excluded_skeleton_count': len(skeletons),
              'old_23_identity_or_skeleton_hits': 0,
              'output_sha256': {str(p.relative_to(ROOT)): S.sha(p) for split in splits for p in (HERE/f'{split}.pt', HERE/f'{split}.sdf')},
              'records': {split: [{k: row[k] for k in ('id','component_id','bucket','smiles','skeleton','Q')} for row in rows] for split, rows in splits.items()}}
    S.save(HERE/'DATASET.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'output_sha256')}, indent=2), flush=True)


if __name__ == '__main__': main()
