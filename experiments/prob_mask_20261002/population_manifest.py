"""Freeze the element-based population, before any prediction-based filtering."""
import argparse
import hashlib
import json
from pathlib import Path


def population(sdf, allowed):
    from rdkit import Chem
    from evaluation_common import reference_population
    refs, excluded = reference_population(sdf, allowed)
    details = []
    for mol in Chem.SDMolSupplier(str(sdf), removeHs=False):
        if mol.GetProp('_Name') in excluded:
            details.append({'name': mol.GetProp('_Name'), 'unsupported_atomic_nums':
                sorted({a.GetAtomicNum() for a in mol.GetAtoms()
                        if a.GetAtomicNum() != 1 and a.GetAtomicNum() not in allowed})})
    return {'schema': 'bondnet.fixed-population.v1',
            'sdf_sha256': hashlib.sha256(Path(sdf).read_bytes()).hexdigest(),
            'allowed_atomic_nums': list(allowed), 'raw_n': len(refs)+len(excluded),
            'eligible_n': len(refs), 'eligible_names': sorted(refs),
            'excluded_by_element': sorted(details, key=lambda r: r['name'])}


def validate_population(sdf, manifest):
    from evaluation_common import reference_population, ALLOWED
    expected = json.loads(Path(manifest).read_text(encoding='utf-8'))
    if expected['allowed_atomic_nums'] != ALLOWED:
        raise ValueError('Population allowed elements differ from the pre-existing rule')
    actual = population(sdf, ALLOWED)
    if actual != expected:
        raise ValueError('Frozen population names/counts/SDF/element rule changed')
    return reference_population(sdf, ALLOWED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sdf', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--raw_n', type=int, required=True)
    ap.add_argument('--eligible_n', type=int, required=True)
    a = ap.parse_args()
    from evaluation_common import ALLOWED
    data = population(a.sdf, ALLOWED)
    if (data['raw_n'], data['eligible_n']) != (a.raw_n, a.eligible_n):
        raise ValueError('Raw/eligible population count mismatch')
    if a.write:
        p = Path(a.manifest)
        if p.exists(): raise ValueError('Do not overwrite an existing population manifest')
        p.write_text(json.dumps(data, indent=1), encoding='utf-8')
    else:
        validate_population(a.sdf, a.manifest)
    print(f'Population verified: raw={data["raw_n"]}, eligible={data["eligible_n"]}, '
          f'excluded={len(data["excluded_by_element"])}')


if __name__ == '__main__': main()
