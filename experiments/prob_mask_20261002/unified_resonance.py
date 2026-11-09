"""Versioned, fixed-denominator graph scoring shared by every method.

Legacy classify is diagnostic only. Native (B,h,q) identity is a separate endpoint.
"""
import argparse
import json
from pathlib import Path
from evaluation_common import ALLOWED, score_dump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per_edge', required=True)
    ap.add_argument('--reference_sdf', required=True)
    ap.add_argument('--label', default='')
    ap.add_argument('--allowed_atomic_nums', type=int, nargs='+', default=ALLOWED)
    ap.add_argument('--expected_n', type=int)
    ap.add_argument('--population_manifest')
    ap.add_argument('--output', default='')
    a = ap.parse_args()
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    out = score_dump(a.per_edge, a.reference_sdf, a.allowed_atomic_nums, a.expected_n, a.population_manifest)
    out['label'] = a.label
    print(f'{a.label} n={out["n"]}; missing={len(out["missing_predictions"])}')
    for key in ('strict', 'unified_resonance', 'smiles_species', 'classify_resonance'):
        print(f'  {key:24s} {out[key]:.4f}')
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(out, indent=1), encoding='utf-8')
        print('wrote', a.output)


if __name__ == '__main__':
    main()
