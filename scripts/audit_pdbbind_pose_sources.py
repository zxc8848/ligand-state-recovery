"""Check that copied MOL2 heavy coordinates match the existing pose500 SDF.

This is a storage/provenance audit, not a benchmark or bond-label validation.
No models are loaded and no ligand chemistry is used as inference input.
"""
from collections import Counter
import csv
import json
from pathlib import Path
import re

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]


def mol2_coordinates(path):
    reading = False
    result = []
    table = Chem.GetPeriodicTable()
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if line.startswith('@<TRIPOS>ATOM'):
            reading = True
            continue
        if line.startswith('@<TRIPOS>'):
            reading = False
        if reading and line.strip():
            fields = line.split()
            symbol = re.match(r'[A-Za-z]+', fields[5].split('.')[0]).group()
            number = table.GetAtomicNumber(symbol)
            if number != 1:
                result.append((number, *(round(float(x), 4) for x in fields[2:5])))
    return Counter(result)


def main():
    with (ROOT/'data/pdbbind_ccd_pose500_manifest.csv').open(encoding='utf-8-sig', newline='') as handle:
        sources = {r['pdb_id']: r['source'] for r in csv.DictReader(handle) if r['status'] == 'written'}
    checked = []; mismatches = []
    supplier = Chem.SDMolSupplier(str(ROOT/'data/pdbbind_ccd_pose500.sdf'), sanitize=False, removeHs=False)
    for mol in supplier:
        if mol is None:
            raise ValueError('Unparseable existing pose SDF')
        key = mol.GetProp('_Name')
        source = ROOT / sources[key]
        conformer = mol.GetConformer()
        expected = Counter((atom.GetAtomicNum(), *(round(float(x), 4) for x in conformer.GetAtomPosition(atom.GetIdx())))
                           for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1)
        actual = mol2_coordinates(source)
        same = actual == expected
        checked.append(dict(pdb_id=key, source=str(source.relative_to(ROOT)), heavy_atoms=sum(expected.values()),
                            heavy_coordinate_multiset_matches=same))
        if not same:
            mismatches.append(key)
    if len(checked) != len(sources):
        raise ValueError('Source manifest / SDF counts differ')
    result = dict(date='2026-10-04', checked=len(checked), matched=len(checked)-len(mismatches),
                  mismatch_ids=mismatches, tolerance='coordinates rounded to 4 decimals; element-aware unordered multiset',
                  scope='Source pose coordinate audit only. Does not validate PDBBind bond labels or establish leakage-free test status.',
                  records=checked)
    output = ROOT/'results/pdbbind_pose_source_audit_20261004.json'
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))
    if mismatches:
        raise ValueError('Historical pose coordinates differ from copied source')


if __name__ == '__main__':
    main()
