"""Draw and freeze a locked pose test set, disjoint from everything we have used.

The 488-pose benchmark has now carried four rounds of design decisions (the
local-search neighbourhood, the verifier target, the delocalized rule form, the
step count T), so it is a development set and cannot carry a confirmatory number.
This draws a fresh set from the same PDBBind source with a different shuffle seed
and excludes, by molecular identity:

  * ``data/diverse_subset_80k.sdf``  -- training set of the archived two-stage model;
  * ``data/geom_drugs_all_random1_rel10.sdf`` -- training corpus of the frozen
    single-stage backbone the new pipeline uses. The existing 488 was never
    audited against this one, and 75 of its molecules turned out to be in it
    (``joint_pose/geom_overlap_names.json``), so excluding it up front is a
    strict improvement over the current benchmark;
  * ``data/pdbbind_ccd_pose500.sdf`` -- the development benchmark itself, plus
    every PDB entry and CCD component id in its manifest.

Identity is matched on two keys: the stereo-retaining canonical SMILES the
original preparation used, and the non-isomeric key used by the GEOM overlap
audit. A molecule is excluded if either matches, which is deliberately
conservative.

Pose and bond-order construction is imported from the archived preparation
script, unchanged, so the new set is the same kind of object as the old one.

Subcommands
-----------
  identities  build the exclusion identity set (slow: one pass over 1.4 GB of
              SDF; ~20 min). Writes a text file, no network needed.
  dryrun      enumerate candidates and apply the cheap exclusions only, to check
              the index parsing and the pool size in seconds before committing
              to the long draw. No downloads, no output files.
  draw        the real draw: fetches each component's CCD cif from RCSB, builds
              the pose with CCD bond orders, applies every exclusion, writes the
              SDF, the manifest and a freeze record with a SHA256.
  verify      re-hash the frozen SDF and assert zero identity overlap with all
              sources. Run this before the set is ever evaluated, and again
              before the numbers go into a paper.

This script never evaluates a model. Keep it that way: the locked set is touched
once, after every configuration is fixed.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = ROOT/'archive/legacy_two_stage_20261003'


def legacy():
    sys.path.insert(0, str(LEGACY))
    from scripts.prepare_pdbbind_ccd_truth import (index_codes, index_years,
                                                   mol2_atoms, ccd_data,
                                                   pose_with_ccd_truth)
    from scripts.prepare_ccd_templatefree import fetch_cif, heavy_identity
    return dict(index_codes=index_codes, index_years=index_years,
                mol2_atoms=mol2_atoms, ccd_data=ccd_data,
                pose_with_ccd_truth=pose_with_ccd_truth,
                fetch_cif=fetch_cif, heavy_identity=heavy_identity)


def both_keys(mol, heavy_identity):
    """(stereo-retaining key, non-isomeric key) for one molecule."""
    from rdkit import Chem
    stereo = heavy_identity(mol)
    plain = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)), isomericSmiles=False)
    return stereo, plain


def cmd_identities(a):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    helpers = legacy()
    hi = helpers['heavy_identity']
    sources = [('diverse80k', a.diverse), ('geom', a.geom), ('dev_poses', a.dev_poses)]
    stereo, plain, counts = set(), set(), {}
    for name, path in sources:
        if not path or not Path(path).exists():
            print(f'{name}: missing {path}, skipped')
            continue
        n = bad = 0
        t0 = time.perf_counter()
        with open(path, 'rb') as handle:
            for mol in Chem.ForwardSDMolSupplier(handle, sanitize=False, removeHs=False):
                if mol is None:
                    bad += 1
                    continue
                try:
                    mol = Chem.Mol(mol)
                    Chem.SanitizeMol(mol)
                    s, p = both_keys(mol, hi)
                except Exception:
                    bad += 1
                    continue
                stereo.add(s)
                plain.add(p)
                n += 1
                if a.limit and n >= a.limit:
                    break
                if n % 25000 == 0:
                    print(f'  {name}: {n} molecules, {time.perf_counter()-t0:.0f}s',
                          flush=True)
        counts[name] = {'molecules': n, 'unparsed': bad,
                        'seconds': time.perf_counter()-t0}
        print(f'{name}: {n} molecules ({bad} unparsed) in '
              f'{counts[name]["seconds"]:.0f}s')
    out = Path(a.identities)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'sources': {name: str(path) for name, path in sources},
        'counts': counts,
        'n_stereo_keys': len(stereo), 'n_plain_keys': len(plain),
        'stereo': sorted(stereo), 'plain': sorted(plain)}), encoding='utf-8')
    print(f'wrote {out}: {len(stereo)} stereo keys, {len(plain)} non-isomeric keys')


def load_identities(path):
    if not path or not Path(path).exists():
        return set(), set(), {}
    blob = json.loads(Path(path).read_text(encoding='utf-8'))
    return set(blob['stereo']), set(blob['plain']), blob.get('counts', {})


def used_entries(manifest):
    """(pdb ids, component ids) already attempted for the development benchmark."""
    pdb, comp = set(), set()
    if manifest and Path(manifest).exists():
        with Path(manifest).open(encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                if row.get('pdb_id'):
                    pdb.add(row['pdb_id'].strip().lower())
                if row.get('component_id'):
                    comp.add(row['component_id'].strip().upper())
    return pdb, comp


def candidate_items(a, helpers):
    root = Path(a.pdbbind_root)
    index_path = Path(a.index) if a.index else root/'index'/'INDEX_general_PL.2020R1.lst'
    codes = helpers['index_codes'](index_path)
    years = helpers['index_years'](index_path)
    found = {}
    for path in root.joinpath('P-L').rglob('*_ligand.mol2'):
        found[path.parent.name.lower()] = path
    items = [(pdb_id, path, codes[pdb_id]) for pdb_id, path in found.items()
             if pdb_id in codes
             and (a.min_year is None or years.get(pdb_id, -1) >= a.min_year)
             and (a.max_year is None or years.get(pdb_id, 10**9) <= a.max_year)]
    random.Random(a.seed).shuffle(items)
    return items, years, len(found)


def cmd_dryrun(a):
    helpers = legacy()
    items, years, n_found = candidate_items(a, helpers)
    pdb_used, comp_used = used_entries(a.dev_manifest)
    stereo, plain, counts = load_identities(a.identities)
    kept = [it for it in items
            if it[0] not in pdb_used and it[2].strip().upper() not in comp_used]
    print(f'mol2 ligands found         {n_found}')
    print(f'in the PL index            {len(items)}')
    print(f'development benchmark used {len(pdb_used)} PDB entries, '
          f'{len(comp_used)} CCD components')
    print(f'after entry/component exclusion {len(kept)} candidates '
          f'(shuffle seed {a.seed})')
    print(f'identity exclusion set     {len(stereo)} stereo / {len(plain)} '
          f'non-isomeric keys' + ('' if stereo else '  <- run `identities` first'))
    if counts:
        print('  built from: ' + ', '.join(
            f'{k} n={v["molecules"]}' for k, v in counts.items()))
    print(f'\nfirst 10 candidates: ' +
          ', '.join(f'{p}/{c}' for p, _, c in kept[:10]))
    print('\nno files written, nothing downloaded')


def cmd_draw(a):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    helpers = legacy()
    out = Path(a.output_sdf)
    if out.exists() and not a.force:
        raise SystemExit(f'refusing to overwrite {out}; the locked set must not be '
                         f'redrawn casually. Pass --force only if it has never '
                         f'been evaluated.')
    stereo, plain, id_counts = load_identities(a.identities)
    if not stereo and not a.allow_without_identities:
        raise SystemExit(f'{a.identities} is missing or empty; run the `identities` '
                         f'subcommand first, or pass --allow_without_identities to '
                         f'draw a set that is NOT audited against the training '
                         f'corpora (not acceptable for a locked test set)')
    items, years, n_found = candidate_items(a, helpers)
    pdb_used, comp_used = used_entries(a.dev_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(out))
    rows, written = [], 0
    reasons = {}
    t0 = time.perf_counter()
    for pdb_id, path, component_id in items:
        row = {'pdb_id': pdb_id, 'year': years.get(pdb_id, ''),
               'component_id': component_id, 'source': str(path),
               'status': 'skipped', 'reason': ''}
        try:
            if pdb_id in pdb_used:
                raise ValueError('pdb_entry_in_development_benchmark')
            if component_id.strip().upper() in comp_used:
                raise ValueError('ccd_component_in_development_benchmark')
            source = helpers['fetch_cif'](component_id, Path(a.ccd_cache),
                                          a.timeout, a.retries)
            atoms, bonds = helpers['ccd_data'](source)
            mol = helpers['pose_with_ccd_truth'](helpers['mol2_atoms'](path),
                                                 atoms, bonds)
            n_heavy = mol.GetNumAtoms()
            if not a.min_heavy <= n_heavy <= a.max_heavy:
                raise ValueError(f'heavy_atoms={n_heavy}')
            s_key, p_key = both_keys(mol, helpers['heavy_identity'])
            if s_key in stereo:
                raise ValueError('identity_in_training_or_devset_stereo')
            if p_key in plain:
                raise ValueError('identity_in_training_or_devset_nonisomeric')
            mol.SetProp('_Name', pdb_id)
            mol.SetProp('pdb_id', pdb_id)
            mol.SetProp('component_id', component_id)
            mol.SetProp('source', 'PDBBind_pose__CCD_truth')
            mol.SetIntProp('total_formal_charge',
                           sum(at.GetFormalCharge() for at in mol.GetAtoms()))
            writer.write(mol)
            row.update(status='written', heavy_atoms=n_heavy,
                       total_charge=sum(at.GetFormalCharge() for at in mol.GetAtoms()))
            written += 1
        except Exception as exc:
            key = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            reasons[key.split(':')[0][:60]] = reasons.get(key.split(':')[0][:60], 0)+1
            row.update(status='error', reason=f'{type(exc).__name__}: {exc}')
        rows.append(row)
        if len(rows) % 100 == 0:
            print(f'attempted={len(rows)} written={written} '
                  f'{time.perf_counter()-t0:.0f}s', flush=True)
        if a.max_mols is not None and written >= a.max_mols:
            break
    writer.close()

    fields = sorted({k for row in rows for k in row})
    with Path(a.manifest).open('w', newline='', encoding='utf-8') as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    freeze = {
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'output_sdf': str(out), 'manifest': str(a.manifest),
        'sha256': digest, 'written': written, 'attempted': len(rows),
        'shuffle_seed': a.seed,
        'mol2_ligands_found': n_found, 'in_index': len(items),
        'skip_reasons': dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        'exclusions': {
            'identity_file': str(a.identities),
            'n_stereo_keys': len(stereo), 'n_nonisomeric_keys': len(plain),
            'identity_sources': id_counts,
            'development_benchmark_pdb_entries': len(pdb_used),
            'development_benchmark_ccd_components': len(comp_used)},
        'status': 'FROZEN AND NOT YET EVALUATED',
        'protocol': 'Touch this set exactly once, after every configuration '
                    '(architecture, decoder, neighbourhood, delocalized rule, '
                    'step count, verifier target) is fixed on the 488-pose '
                    'development set. Any tuning after a number is read from '
                    'this set makes it a development set too.'}
    Path(a.freeze).write_text(json.dumps(freeze, indent=1), encoding='utf-8')
    print(json.dumps(freeze, indent=1))


def cmd_verify(a):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    helpers = legacy()
    out = Path(a.output_sdf)
    freeze = json.loads(Path(a.freeze).read_text(encoding='utf-8'))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f'sha256 recorded {freeze["sha256"]}')
    print(f'sha256 now      {digest}  '
          f'{"MATCH" if digest == freeze["sha256"] else "*** CHANGED ***"}')
    stereo, plain, _ = load_identities(a.identities)
    dev_s, dev_p = set(), set()
    for mol in Chem.SDMolSupplier(str(a.dev_poses), removeHs=False):
        if mol is None:
            continue
        try:
            s, p = both_keys(mol, helpers['heavy_identity'])
        except Exception:
            continue
        dev_s.add(s); dev_p.add(p)
    hit_s = hit_p = hit_dev = n = 0
    for mol in Chem.SDMolSupplier(str(out), removeHs=False):
        if mol is None:
            continue
        n += 1
        try:
            s, p = both_keys(mol, helpers['heavy_identity'])
        except Exception:
            continue
        hit_s += s in stereo
        hit_p += p in plain
        hit_dev += (s in dev_s) or (p in dev_p)
    print(f'molecules in the locked set            {n}')
    print(f'identity hits vs training corpora      stereo {hit_s}, '
          f'non-isomeric {hit_p}')
    print(f'identity hits vs the 488 development set {hit_dev}')
    ok = (digest == freeze['sha256']) and hit_s == 0 and hit_p == 0 and hit_dev == 0
    print('\nVERDICT: ' + ('clean and unchanged' if ok else
                           'NOT clean -- do not use as a confirmatory test set'))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['identities', 'dryrun', 'draw', 'verify'])
    ap.add_argument('--pdbbind_root', default=str(ROOT/'data/pdbbind'))
    ap.add_argument('--index', default=None)
    ap.add_argument('--ccd_cache', default=str(ROOT/'data/ccd_cif_cache'))
    ap.add_argument('--diverse', default=str(ROOT/'data/diverse_subset_80k.sdf'))
    ap.add_argument('--geom', default=str(ROOT/'data/geom_drugs_all_random1_rel10.sdf'))
    ap.add_argument('--dev_poses', default=str(ROOT/'data/pdbbind_ccd_pose500.sdf'))
    ap.add_argument('--dev_manifest',
                    default=str(ROOT/'data/pdbbind_ccd_pose500_manifest.csv'))
    ap.add_argument('--identities',
                    default=str(ROOT/'data/locked_testset_excluded_identities.json'))
    ap.add_argument('--output_sdf', default=str(ROOT/'data/pdbbind_pose_locked500.sdf'))
    ap.add_argument('--manifest',
                    default=str(ROOT/'data/pdbbind_pose_locked500_manifest.csv'))
    ap.add_argument('--freeze', default=str(ROOT/'data/pdbbind_pose_locked500_freeze.json'))
    ap.add_argument('--seed', type=int, default=20261004)
    ap.add_argument('--max_mols', type=int, default=500)
    ap.add_argument('--min_year', type=int, default=None)
    ap.add_argument('--max_year', type=int, default=None)
    ap.add_argument('--min_heavy', type=int, default=4)
    ap.add_argument('--max_heavy', type=int, default=80)
    ap.add_argument('--timeout', type=float, default=30.0)
    ap.add_argument('--retries', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0, help='identities: stop early (smoke test)')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--allow_without_identities', action='store_true')
    a = ap.parse_args()
    fn = {'identities': cmd_identities, 'dryrun': cmd_dryrun,
          'draw': cmd_draw, 'verify': cmd_verify}[a.action]
    code = fn(a)
    sys.exit(code or 0)


if __name__ == '__main__':
    main()
