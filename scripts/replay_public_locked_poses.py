"""Rebuild a local public-source pose candidate; the replay never reads PDBBind.

Pin instance/atom order and source hashes with --audit once. Later replay with
only --plan plus the public source cache. A non-exact public coordinate is not
silently rounded or replaced with its historical PDBBind value.
"""

import argparse
import importlib.util
import json
from pathlib import Path

from rdkit import Chem, rdBase

from audit_public_locked_pose_rebuild import ROOT, chemistry, legacy_pdb_conformers, public_conformers, sha256


def load_builder():
    source = ROOT/'archive/legacy_two_stage_20261003/scripts/prepare_pdbbind_ccd_truth.py'
    spec = importlib.util.spec_from_file_location('historical_ccd_builder', source)
    historical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical)
    return historical


def from_audit(audit):
    rows = []
    for r in audit['records']:
        exact = r['status'] == 'matches_at_sdf_precision'
        if not exact and (r['status'] != 'coordinate_mismatch' or 'geometry_diagnostic' not in r):
            raise ValueError(f"Unresolved public pose {r['pdb_id']}")
        d = r.get('geometry_diagnostic', {})
        origin, filename = 'mmcif', r['pdb_id']+'.cif.gz'
        if r.get('versioned_file'):
            origin, filename = 'versions', r['versioned_file']
        if r.get('legacy_pdb_file'):
            origin, filename = 'pdb', r['legacy_pdb_file']
        rows.append(dict(pdb_id=r['pdb_id'], component_id=r['component_id'], origin=origin,
                         filename=filename, coordinate_sha256=r['mmcif_sha256'], ccd_sha256=r['ccd_sha256'],
                         instance=r['public_instance'] if exact else d['minimum_rmsd_instance'],
                         atom_order=r['ccd_atom_order'] if exact else d['ccd_atom_order'],
                         exact_against_historical_coordinates=exact))
    if len(rows) != audit['n'] or len({r['pdb_id'] for r in rows}) != audit['n']:
        raise ValueError('Duplicate or missing audit records')
    return dict(schema='public-pose-replay-v1', n=audit['n'], rdkit=rdBase.rdkitVersion, records=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True, help='Contains mmcif/, versions/, pdb/.')
    p.add_argument('--ccd-dir', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--audit', type=Path, help='Generate a pinned plan and validate against the historical SDF.')
    p.add_argument('--output-sdf', type=Path, required=True)
    args = p.parse_args()
    old = {}
    if args.audit:
        audit = json.loads(args.audit.read_text(encoding='utf-8'))
        plan = from_audit(audit)
        frozen = ROOT/'data/pdbbind_pose_locked500.sdf'
        if sha256(frozen) != audit['frozen_sdf_sha256']:
            raise ValueError('Historical SDF differs from the audit input')
        old = {mol.GetProp('_Name'): mol for mol in Chem.SDMolSupplier(str(frozen), removeHs=False)}
    else:
        plan = json.loads(args.plan.read_text(encoding='utf-8'))
    if plan['schema'] != 'public-pose-replay-v1' or plan['rdkit'] != rdBase.rdkitVersion:
        raise ValueError('Plan schema or RDKit version mismatch')
    if len(plan['records']) != plan['n'] or len({r['pdb_id'] for r in plan['records']}) != plan['n']:
        raise ValueError('Duplicate or missing planned records')
    historical = load_builder()
    molecules = []
    keys = ('model', 'label_asym_id', 'auth_asym_id', 'auth_seq_id', 'insertion_code', 'alt_id')
    for r in plan['records']:
        code, name = r['component_id'], r['pdb_id']
        path = args.source_root/r['origin']/r['filename']
        ccd = args.ccd_dir/f'{code}.cif'
        if sha256(path) != r['coordinate_sha256'] or sha256(ccd) != r['ccd_sha256']:
            raise ValueError(f'Source hash mismatch for {name}')
        read = legacy_pdb_conformers if r['origin'] == 'pdb' else public_conformers
        lookup = {inst: rows for inst, rows in read(path, code)}
        atoms = {x[0]: x for x in lookup[tuple(r['instance'][k] for k in keys)]}
        if set(atoms) != set(r['atom_order']) or len(atoms) != len(r['atom_order']):
            raise ValueError(f'Atom order or inventory mismatch for {name}')
        mol = historical.pose_with_ccd_truth([atoms[n] for n in r['atom_order']], *historical.ccd_data(ccd))
        if old:
            if chemistry(mol) != chemistry(old[name]):
                raise ValueError(f'Bond/charge/H labels disagree for {name}')
            current = mol.GetConformer().GetPositions()
            original = old[name].GetConformer().GetPositions()
            same = bool((current == original).all())
            if same != r['exact_against_historical_coordinates']:
                raise ValueError(f'Coordinate equivalence differs from the audit for {name}')
            if not same and float(abs(current-original).max()) > 0.001:
                raise ValueError(f'Coordinate differences exceed 0.001 angstrom for {name}')
        mol.SetProp('_Name', name)
        mol.SetProp('pdb_id', name)
        mol.SetProp('component_id', code)
        mol.SetProp('source', 'public_wwPDB_pose__CCD_truth')
        mol.SetIntProp('total_formal_charge', Chem.GetFormalCharge(mol))
        molecules.append(mol)
    args.output_sdf.parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(args.output_sdf)) as writer:
        for mol in molecules:
            writer.write(mol)
    if args.audit:
        args.plan.write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')
    print(f"{len(molecules)} public-source records, {sum(r['exact_against_historical_coordinates'] for r in plan['records'])} coordinate-exact; SDF sha256={sha256(args.output_sdf)}")


if __name__ == '__main__':
    main()
