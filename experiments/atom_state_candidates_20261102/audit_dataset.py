"""Independent leakage/split checks for the newly selected development cohort."""
import collections
import json
from pathlib import Path

import torch
from rdkit import Chem, RDLogger

import run as R

HERE = R.HERE
ROOT = R.ROOT


def main():
    RDLogger.DisableLog('rdApp.*')
    dataset = json.loads((HERE/'DATASET.json').read_text())
    prep = json.loads((HERE/'PREPARATION_PLAN.json').read_text())
    assert dataset['preparation_plan_sha256'] == R.S.sha(HERE/'PREPARATION_PLAN.json')
    for rel, want in {**prep['input_sha256'], **dataset['output_sha256']}.items():
        assert R.S.sha(ROOT/rel) == want, rel
    splits = {s: torch.load(HERE/f'{s}.pt', weights_only=False, map_location='cpu') for s in ('development', 'held_back')}
    selected = [r for rows in splits.values() for r in rows]
    identities = {r['smiles'] for r in selected}
    skeletons = {r['skeleton'] for r in selected}
    assert len(identities) == len(selected)
    assert not ({r['skeleton'] for r in splits['development']} & {r['skeleton'] for r in splits['held_back']})
    conflicts = []; checked = collections.Counter()
    keys = json.loads((ROOT/'data/locked_testset_excluded_identities.json').read_text())
    assert not identities.intersection(set(keys['plain']) | set(keys['stereo']))
    old270 = torch.load(ROOT/'experiments/charged_benchmark_20261102/selected.pt', weights_only=False, map_location='cpu')
    sources = torch.load(R.S.SOURCES, weights_only=False, map_location='cpu')
    for label, rows in [('old_charge270', old270), *[('head_'+s, rows) for s, rows in sources.items()]]:
        for row in rows:
            if row['smiles'] in identities or row['skeleton'] in skeletons:
                conflicts.append({'source': label, 'id': row['id']})
            checked[label] += 1
    paths = [ROOT/p for p in ('data/ccd_heavy_leakagefree.sdf', 'data/pdbbind_ccd_pose500.sdf',
             'data/pdbbind_pose_locked500.sdf',
             'work/paper2_noise_ccd_probe/ccd_sigma0p00/reference_for_scoring.sdf',
             'data/diverse_subset_80k.sdf')]
    new_ids = {r['component_id'] for r in selected}
    for path in paths:
        for index, mol in enumerate(Chem.SDMolSupplier(str(path), removeHs=False)):
            if mol is None: continue
            heavy = Chem.RemoveHs(mol)
            plain = Chem.MolToSmiles(heavy, isomericSmiles=False)
            skeleton = R.S.J.skeleton(heavy)
            if plain in identities or skeleton in skeletons:
                conflicts.append({'source': path.name, 'index': index})
            if path == paths[0]: assert mol.GetProp('_Name') not in new_ids
            checked[path.name] += 1
        print('audited', path.name, checked[path.name], flush=True)
    assert not conflicts, conflicts
    for s, rows in splits.items():
        assert dict(collections.Counter(r['bucket'] for r in rows)) == dataset['split_counts'][s]
        assert {r['smiles'] for r in rows} == {r['smiles'] for r in dataset['records'][s]}
    for rel, want in {**prep['input_sha256'], **dataset['output_sha256']}.items():
        assert R.S.sha(ROOT/rel) == want, rel
    receipt = {'scope': 'independent exclusion audit; no model outcomes', 'conflicts': conflicts,
               'development_n': len(splits['development']), 'held_back_n': len(splits['held_back']),
               'unique_combined_identities': len(identities), 'dev_heldback_skeleton_intersection': 0,
               'full_saved_exclusion_union_identity_hits': 0, 'checked_sources': dict(checked),
               'all_source_and_dataset_hashes_unchanged': True, 'auditor_sha256': R.S.sha(Path(__file__))}
    R.S.save(HERE/'DATASET_AUDIT.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__': main()
