"""Audit (not replace) the frozen pose inputs against public RCSB mmCIF and CCD.

Pass a directory of files.rcsb.org/download/<pdb_id>.cif.gz downloads. The old
coordinates locate the historical instance and atom order, not a new test set.
The report records identifiers and hashes but does not export PDBBind poses.
"""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import importlib.util
from io import StringIO
from itertools import chain
import json
import math
from pathlib import Path

from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from rdkit import Chem, rdBase

ROOT = Path(__file__).resolve().parents[1]
TOL = 0.000051  # Half a 4-decimal SDF unit, plus floating-point tolerance.


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coordinate_key(element, xyz):
    return element, tuple(round(float(x), 4) for x in xyz)


def public_atom_site(path, component):
    # Current RCSB files use a single, one-row-per-line atom_site loop. Filter
    # by literal substring (a superset) before parsing; the exact code is
    # checked against label_comp_id below. Fall back for other mmCIF layouts.
    fast = False
    selected = []
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            if line.startswith('_atom_site.group_PDB'):
                headers = [line]
                for line in handle:
                    if not line.startswith('_atom_site.'):
                        break
                    headers.append(line)
                for line in chain((line,), handle):
                    if line.strip() == '#':
                        fast = True
                        break
                    if not line.startswith(('ATOM ', 'HETATM ')):
                        break
                    if component in line:
                        selected.append(line)
                break
    if fast:
        if not selected:
            return {}
        return MMCIF2Dict(StringIO('data_pose\nloop_\n'+''.join(headers+selected)))
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return MMCIF2Dict(handle)


def public_conformers(path, component):
    data = public_atom_site(path, component)
    if not data:
        return
    prefix = '_atom_site.'
    count = len(data[prefix + 'label_comp_id'])

    def col(key, fallback=None):
        return data[prefix + key] if fallback is None else data.get(prefix + key, [fallback] * count)

    groups = defaultdict(list)
    columns = (col('label_comp_id'), col('label_atom_id'), col('type_symbol'),
               col('label_asym_id'), col('auth_asym_id'), col('auth_seq_id'),
               col('pdbx_PDB_ins_code', '?'), col('pdbx_PDB_model_num', '1'),
               col('label_alt_id', '.'), col('Cartn_x'), col('Cartn_y'), col('Cartn_z'))
    for code, name, elem, chain, auth_chain, seq, ins, model, alt, x, y, z in zip(*columns, strict=True):
        if code != component or elem.upper() in {'H', 'D', 'T'}:
            continue
        groups[(model, chain, auth_chain, seq, ins)].append((name, elem.capitalize(), (float(x), float(y), float(z)), alt))
    yield from split_altlocs(groups)


def split_altlocs(groups):
    for instance, atoms in sorted(groups.items()):
        alts = sorted({a[3] for a in atoms} - {'?', '.'}) or ['.']
        for alt in alts:
            rows = [(n, e, xyz) for n, e, xyz, a in atoms if a in {'?', '.', alt}]
            if len({n for n, _, _ in rows}) == len(rows):
                yield (*instance, alt), rows


def legacy_pdb_conformers(path, component):
    groups = defaultdict(list)
    model = '1'
    with gzip.open(path, 'rt', encoding='ascii') as handle:
        for line in handle:
            if line.startswith('MODEL '):
                model = line[10:14].strip()
            if line[:6] not in {'ATOM  ', 'HETATM'} or line[17:20].strip() != component:
                continue
            elem = line[76:78].strip().capitalize()
            if elem in {'H', 'D', 'T'}:
                continue
            chain, seq, ins = line[21:22].strip(), line[22:26].strip(), line[26:27].strip() or '?'
            name, alt = line[12:16].strip(), line[16:17].strip() or '.'
            xyz = tuple(float(line[i:i+8]) for i in (30, 38, 46))
            groups[(model, None, chain, seq, ins)].append((name, elem, xyz, alt))
    yield from split_altlocs(groups)


def chemistry(mol):
    atoms = [(a.GetAtomicNum(), a.GetFormalCharge(), a.GetTotalNumHs()) for a in mol.GetAtoms()]
    bonds = sorted((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                    max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                    4 if b.GetIsAromatic() else b.GetBondTypeAsDouble()) for b in mol.GetBonds())
    return atoms, bonds


def compare(mol, path, ccd, build, read=public_conformers):
    component = mol.GetProp('component_id')
    conformer = mol.GetConformer()
    target = [(a.GetSymbol(), tuple(conformer.GetAtomPosition(a.GetIdx()))) for a in mol.GetAtoms()]
    keys = [coordinate_key(*x) for x in target]
    by_key = Counter(keys)
    if len(by_key) != len(keys):
        return {'status': 'historical_coordinate_collision'}
    refs = list(read(path, component))
    result = {'mmcif_sha256': sha256(path), 'candidate_conformers': len(refs)}
    if not refs:
        return {**result, 'status': 'no_public_component'}
    atoms, bonds = ccd
    expected_names = {n: e for n, (e, _) in atoms.items() if e != 'H'}
    matched = []
    same_inventory = 0
    max_overlap = 0
    for instance, rows in refs:
        if {n: e for n, e, _ in rows} != expected_names:
            continue
        same_inventory += 1
        public = {coordinate_key(e, xyz): (n, e, xyz) for n, e, xyz in rows}
        overlap = len(public.keys() & by_key.keys())
        max_overlap = max(max_overlap, overlap)
        if overlap != len(target):
            continue
        ordered = [public[key] for key in keys]
        max_delta = max(abs(a-b) for (_, _, p), (_, q) in zip(ordered, target, strict=True) for a, b in zip(p, q))
        if not math.isfinite(max_delta) or max_delta > TOL:
            continue
        reconstructed = build(ordered, atoms, bonds)
        matched.append((instance, ordered, max_delta, chemistry(reconstructed) == chemistry(mol)))
    result.update(complete_ccd_conformers=same_inventory, best_exact_coordinate_overlap=max_overlap)
    if not matched:
        return {**result, 'status': 'coordinate_mismatch' if same_inventory else 'inventory_mismatch'}
    identical = [match for match in matched if match[3]]
    if not identical:
        return {**result, 'status': 'ccd_state_mismatch', 'coordinate_matches': len(matched)}
    instance, ordered, delta, _ = identical[0]
    fields = ('model', 'label_asym_id', 'auth_asym_id', 'auth_seq_id', 'insertion_code', 'alt_id')
    return {**result, 'status': 'matches_at_sdf_precision', 'coordinate_matches': len(matched),
            'state_matches': len(identical), 'max_abs_coordinate_delta_angstrom': delta,
            'public_instance': dict(zip(fields, instance, strict=True)), 'ccd_atom_order': [x[0] for x in ordered]}


def diagnose_geometry(mol, path, source_atoms, expected_names):
    import numpy as np

    source = [(n, e, xyz) for n, e, xyz in source_atoms if e != 'H']
    conf = mol.GetConformer()
    reference = [(a.GetSymbol(), tuple(conf.GetAtomPosition(a.GetIdx()))) for a in mol.GetAtoms()]
    if [coordinate_key(e, xyz) for _, e, xyz in source] != [coordinate_key(*x) for x in reference]:
        raise ValueError('Source MOL2 atom order is not verified against the locked SDF')
    old = np.array([xyz for _, xyz in reference])
    old_centered = old - old.mean(axis=0)
    options = []
    for instance, rows in public_conformers(path, mol.GetProp('component_id')):
        if {n: e for n, e, _ in rows} != expected_names:
            continue
        lookup = {n: xyz for n, _, xyz in rows}
        new = np.array([lookup[n] for n, _, _ in source])
        centered = new - new.mean(axis=0)
        u, _, vt = np.linalg.svd(centered.T @ old_centered)
        u[:, -1] *= np.linalg.det(u @ vt)
        raw = np.sqrt(np.mean(np.sum((new - old)**2, axis=1)))
        aligned = np.sqrt(np.mean(np.sum((centered @ (u @ vt) - old_centered)**2, axis=1)))
        options.append((float(aligned), float(raw), instance, new))
    if not options:
        return {}
    aligned, raw, instance, new = min(options, key=lambda x: x[:3])
    return {'minimum_name_aligned_rmsd_angstrom': aligned, 'raw_rmsd_of_same_instance_angstrom': raw,
            'max_abs_unrounded_delta_angstrom': float(np.max(np.abs(new - old))),
            'exact_after_3_decimal_quantization': bool(np.array_equal(np.round(new, 3), old)),
            'exact_after_4_decimal_quantization': bool(np.array_equal(np.round(new, 4), old)),
            'max_abs_delta_after_3_decimal_quantization': float(np.max(np.abs(np.round(new, 3) - old))),
            'ccd_atom_order': [n for n, _, _ in source],
            'minimum_rmsd_instance': dict(zip(('model', 'label_asym_id', 'auth_asym_id', 'auth_seq_id', 'insertion_code', 'alt_id'), instance, strict=True))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mmcif-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ccd-dir', type=Path, default=ROOT/'data/ccd_cif_cache')
    parser.add_argument('--versioned-dir', type=Path, help='Also try public wwPDB versioned mmCIFs when the current entry does not match.')
    parser.add_argument('--pdb-dir', type=Path, help='Also try public RCSB legacy PDB files for coordinate precision differences.')
    parser.add_argument('--resume-downloads', action='store_true', help='Recheck unavailable downloads; reuse rows only when all previously checked inputs still match their recorded hashes.')
    parser.add_argument('--diagnose-local-source', action='store_true', help='Use historical MOL2 atom names for rigid-alignment diagnosis only, not for replay.')
    args = parser.parse_args()
    source = ROOT/'archive/legacy_two_stage_20261003/scripts/prepare_pdbbind_ccd_truth.py'
    spec = importlib.util.spec_from_file_location('historical_ccd_builder', source)
    historical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical)
    population = json.loads((ROOT/'data/pdbbind_pose_locked500_population_v3_1.json').read_text())
    sdf = ROOT/'data/pdbbind_pose_locked500.sdf'
    if sha256(sdf) != population['sdf_sha256']:
        raise ValueError('Frozen SDF hash mismatch')
    eligible = set(population['eligible_names'])
    local_sources = {}
    if args.diagnose_local_source:
        with (ROOT/'data/pdbbind_pose_locked500_manifest.csv').open(encoding='utf-8-sig', newline='') as manifest:
            local_sources = {r['pdb_id']: Path(r['source']) for r in csv.DictReader(manifest) if r['status'] == 'written'}
    previous = {}
    if args.resume_downloads:
        report = json.loads(args.output.read_text(encoding='utf-8'))
        if report['schema'] != 'public-locked-pose-audit-v1' or report['frozen_sdf_sha256'] != population['sdf_sha256'] or report['rdkit'] != rdBase.rdkitVersion:
            raise ValueError('Existing audit is incompatible')
        previous = {r['pdb_id']: r for r in report['records']}
    result = []
    for mol in Chem.SDMolSupplier(str(sdf), removeHs=False):
        if mol is None:
            raise ValueError('Frozen SDF is not fully parseable')
        pdb = mol.GetProp('_Name')
        if pdb not in eligible:
            continue
        code = mol.GetProp('component_id')
        row = {'pdb_id': pdb, 'component_id': code, 'heavy_atoms': mol.GetNumAtoms()}
        path = args.mmcif_dir/f'{pdb}.cif.gz'
        ccd_path = args.ccd_dir/f'{code}.cif'
        cached = previous.get(pdb, {})
        if (cached.get('component_id') == code and cached.get('status') == 'matches_at_sdf_precision'
                and 'versioned_file' not in cached and 'legacy_pdb_file' not in cached and path.exists() and ccd_path.exists()
                and cached.get('mmcif_sha256') == sha256(path) and cached.get('ccd_sha256') == sha256(ccd_path)):
            result.append(cached)
            continue
        try:
            if not path.exists():
                row['status'] = 'mmcif_unavailable'
            elif not ccd_path.exists():
                row['status'] = 'ccd_unavailable'
            else:
                row['ccd_sha256'] = sha256(ccd_path)
                ccd = historical.ccd_data(ccd_path)
                row.update(compare(mol, path, ccd, historical.pose_with_ccd_truth))
                if args.versioned_dir and row['status'] != 'matches_at_sdf_precision':
                    row['current_status'] = row['status']
                    checks = {}
                    for version in sorted(args.versioned_dir.glob(f'pdb_0000{pdb}_xyz_v*.cif.gz')):
                        verdict = compare(mol, version, ccd, historical.pose_with_ccd_truth)
                        checks[version.name] = verdict['status']
                        if verdict['status'] == 'matches_at_sdf_precision':
                            row['current_mmcif_sha256'] = row['mmcif_sha256']
                            row.update(verdict, versioned_file=version.name)
                            break
                    row['versioned_checks'] = checks
                if args.diagnose_local_source and row['status'] == 'coordinate_mismatch':
                    try:
                        row['geometry_diagnostic'] = diagnose_geometry(mol, path, historical.mol2_atoms(local_sources[pdb]),
                                                                       {n: e for n, (e, _) in ccd[0].items() if e != 'H'})
                    except Exception as exc:
                        row['geometry_diagnostic_error'] = f'{type(exc).__name__}: {exc}'
                if args.pdb_dir and row['status'] == 'coordinate_mismatch':
                    legacy = args.pdb_dir/f'{pdb}.pdb.gz'
                    if legacy.exists():
                        verdict = compare(mol, legacy, ccd, historical.pose_with_ccd_truth, read=legacy_pdb_conformers)
                        row['legacy_pdb_status'] = verdict['status']
                        if verdict['status'] == 'matches_at_sdf_precision':
                            row['current_mmcif_sha256'] = row['mmcif_sha256']
                            row.update(verdict, legacy_pdb_file=legacy.name)
        except Exception as exc:
            row.update(status='parse_or_build_error', error=f'{type(exc).__name__}: {exc}')
        result.append(row)
    if len(result) != len(eligible) or {r['pdb_id'] for r in result} != eligible:
        raise ValueError('Audit population does not match the locked manifest')
    summary = dict(Counter(r['status'] for r in result))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'schema': 'public-locked-pose-audit-v1', 'rdkit': rdBase.rdkitVersion,
                                       'frozen_sdf_sha256': population['sdf_sha256'], 'n': len(result),
                                       'counts': summary, 'records': result}, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
