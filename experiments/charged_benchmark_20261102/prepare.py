"""Prepare a new source-excluded charge-stratified CCD benchmark, without inference."""
import collections
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from rdkit import Chem, RDLogger

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'experiments/prob_mask_20261002'
sys.path.insert(0, str(OLD))
import joint as J

BUCKETS = ('cation', 'anion', 'separated_neutral', 'uncharged_neutral')
ALLOWED = {6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53, 83}
TARGET = 100


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding='utf-8')


def classify(mol):
    charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
    Q = sum(charges)
    if Q > 0:
        return 'cation'
    if Q < 0:
        return 'anion'
    positive = [i for i, q in enumerate(charges) if q > 0]
    negative = [i for i, q in enumerate(charges) if q < 0]
    if not positive and not negative:
        return 'uncharged_neutral'
    distances = Chem.GetDistanceMatrix(mol)
    # Conservative graph-based selection, not a thermodynamic zwitterion claim.
    if min(distances[i, j] for i in positive for j in negative) >= 3:
        return 'separated_neutral'
    return 'local_charge_separation'


def main():
    RDLogger.DisableLog('rdApp.*')
    sdf = ROOT / 'data/ccd_heavy_leakagefree.sdf'
    keys_path = ROOT / 'data/locked_testset_excluded_identities.json'
    sources_path = OLD / 'joint_v1/sources.pt'
    benchmarks = [ROOT / p for p in (
        'data/pdbbind_ccd_pose500.sdf', 'data/pdbbind_pose_locked500.sdf',
        'work/paper2_noise_ccd_probe/ccd_sigma0p00/reference_for_scoring.sdf')]
    inputs = [sdf, keys_path, sources_path, *benchmarks, Path(__file__)]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    keys = json.loads(keys_path.read_text())
    excluded = set(keys['plain']) | set(keys['stereo'])
    for path in benchmarks:
        for m in Chem.SDMolSupplier(str(path), removeHs=False):
            if m is None:
                raise ValueError(f'Invalid benchmark reference: {path}')
            m = Chem.RemoveHs(m)
            excluded.add(Chem.MolToSmiles(m, isomericSmiles=False))
            excluded.add(Chem.MolToSmiles(m, isomericSmiles=True))
    source_rows = torch.load(sources_path, weights_only=False, map_location='cpu')
    source_skeletons = set()
    for split in ('train', 'val'):
        for row in source_rows[split]:
            excluded.add(row['smiles'])
            source_skeletons.add(row['skeleton'])
    pools = {b: [] for b in BUCKETS}
    rejected, all_buckets = collections.Counter(), collections.Counter()
    seen, records = set(), []
    for idx, m in enumerate(Chem.SDMolSupplier(str(sdf), removeHs=False)):
        if m is None:
            rejected['unparsed'] += 1; continue
        name = m.GetProp('_Name') if m.HasProp('_Name') else str(idx)
        try:
            heavy = Chem.RemoveHs(Chem.Mol(m))
            if not 3 <= heavy.GetNumAtoms() <= 60:
                rejected['size_outside_3_60'] += 1; continue
            if not {a.GetAtomicNum() for a in heavy.GetAtoms()} <= ALLOWED:
                rejected['unsupported_element'] += 1; continue
            if len(Chem.GetMolFrags(heavy)) != 1:
                rejected['multiple_components'] += 1; continue
            if any(a.GetNumRadicalElectrons() for a in heavy.GetAtoms()):
                rejected['radical'] += 1; continue
            plain = Chem.MolToSmiles(heavy, isomericSmiles=False)
            stereo = Chem.MolToSmiles(heavy, isomericSmiles=True)
            if plain in excluded or stereo in excluded:
                rejected['known_source_or_existing_benchmark_identity'] += 1; continue
            if J.skeleton(heavy) in source_skeletons:
                rejected['head_train_val_connectivity_skeleton'] += 1; continue
            if plain in seen:
                rejected['duplicate_nonisomeric_identity'] += 1; continue
            seen.add(plain)
            bucket = classify(heavy)
            all_buckets[bucket] += 1
            if bucket not in pools:
                rejected['local_charge_separation_not_zwitterion_like'] += 1; continue
            row = J.make_row(m, 'charged_ccd_' + name, idx, 'ccd_ideal_charge_probe')
            row['bucket'] = bucket
            row['component_id'] = name
            # Selection depends only on reference eligibility and a fixed identity hash.
            rank = hashlib.sha256(('charged-20261102|' + plain).encode()).hexdigest()
            pools[bucket].append((rank, row, heavy))
        except Exception as exc:
            rejected['reference_validation:' + str(exc).split('\n')[0][:90]] += 1
    for bucket in BUCKETS:
        for _, row, mol in sorted(pools[bucket], key=lambda x: x[0])[:TARGET]:
            records.append(row)
    selected = HERE / 'selected.pt'
    if selected.exists():
        raise ValueError('Preparation already exists; do not silently replace the cohort')
    torch.save(records, selected)
    writer = Chem.SDWriter(str(HERE / 'reference.sdf'))
    manifest = []
    for row in records:
        m = Chem.MolFromSmiles(row['smiles'])
        # Re-read the source record to retain original atom numbering and coordinates.
        # A supplier supports indexed access without using any model output.
        manifest.append({k: row[k] for k in ('id', 'component_id', 'source_index',
                                           'bucket', 'smiles', 'skeleton', 'Q')})
    supplier = Chem.SDMolSupplier(str(sdf), removeHs=False)
    for row in records:
        m = Chem.RemoveHs(supplier[row['source_index']])
        m.SetProp('_Name', row['id']); writer.write(m)
    writer.close()
    assert hashes == {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    info = {
        'scope': 'new charge-enriched CCD ideal-coordinate diagnostic; not real pose validation',
        'selection_seed': 'sha256(charged-20261102|nonisomeric_identity)',
        'target_per_bucket': TARGET,
        'bucket_rule': 'Q sign; neutral separated charges require minimum positive-negative heavy-graph distance >=3; local nitro/N-oxide-like separation excluded',
        'identity_exclusion': 'saved full GEOM/Diverse/development union plus current development/locked/CCD1000 identities; direct head train+val identities and skeletons',
        'not_a_scaffold_disjoint_claim': True,
        'pool_counts': {b: len(pools[b]) for b in BUCKETS},
        'selected_counts': dict(collections.Counter(r['bucket'] for r in records)),
        'reference_charge_bucket_counts_before_vocabulary_validation': dict(all_buckets),
        'rejections': dict(rejected), 'records': manifest, 'input_sha256': hashes,
        'selected_sha256': sha(selected), 'reference_sdf_sha256': sha(HERE/'reference.sdf'),
        'no_model_outcomes_used_in_selection': True,
    }
    save(HERE / 'DATASET.json', info)
    print(json.dumps({k: info[k] for k in ('pool_counts', 'selected_counts', 'rejections')}, indent=2))


if __name__ == '__main__':
    main()
