"""Cross-training-seed summary of the noise / CCD sweep.

For every condition, reports the new pipeline's absolute accuracy and its paired
delta against the archived two-stage pipeline as mean +- SD over training seeds,
so no number in the paper rests on a single seed. Standard library only.
"""
import argparse
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONDS = ['pdbbind_sigma0p00', 'pdbbind_sigma0p05', 'pdbbind_sigma0p10',
         'pdbbind_sigma0p15', 'ccd_sigma0p00', 'ccd_sigma0p10']
SHORT = {'pdbbind_sigma0p00': 'pose 0.00', 'pdbbind_sigma0p05': 'pose 0.05',
         'pdbbind_sigma0p10': 'pose 0.10', 'pdbbind_sigma0p15': 'pose 0.15',
         'ccd_sigma0p00': 'CCD  0.00', 'ccd_sigma0p10': 'CCD  0.10'}
EP = [('strict', 'strict'), ('resonance_aware', 'resonance'),
      ('canonical_smiles', 'SMILES')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='seed20261002,seed20261003,seed20261004')
    ap.add_argument('--selector', default='localmap_deloc')
    ap.add_argument('--scope', default='all', choices=['all', 'geom_disjoint'])
    ap.add_argument('--kind', default='fullgraph', choices=['fullgraph', 'refedges'],
                    help='which dump the scorer outputs came from; fullgraph is '
                         'primary, refedges is the published-protocol variant')
    ap.add_argument('--output', default='')
    a = ap.parse_args()
    # the fullgraph outputs carry a suffix; the published-protocol ones do not
    suf = '' if a.kind == 'refedges' else '_fullgraph'
    seeds = [s.strip() for s in a.seeds.split(',') if s.strip()]
    out = {'seeds': seeds, 'selector': a.selector, 'scope': a.scope, 'conditions': {}}

    print(f'selector {a.selector}   scope {a.scope}   seeds {", ".join(seeds)}')
    print()
    print('new pipeline, absolute accuracy: mean +- SD over training seeds')
    print(f'{"condition":11s} {"n":>5s}  ' +
          '  '.join(f'{lab:>18s}' for _, lab in EP) + f'  {"old (fixed)":>22s}')
    rows = {}
    for c in CONDS:
        vals = {lab: [] for _, lab in EP}
        dels = {lab: [] for _, lab in EP}
        old = {}
        n = None
        for s in seeds:
            f = HERE/'joint_v6'/s/c/f'comparison_vs_old_{a.selector}{suf}.json'
            if not f.exists():
                continue
            d = json.loads(f.read_text(encoding='utf-8'))[a.scope]
            n = d['n']
            for k, lab in EP:
                vals[lab].append(d[k]['new'])
                dels[lab].append(d[k]['delta'])
                old[lab] = d[k]['old']
        if n is None:
            print(f'{SHORT[c]:11s}  -- not run --'); continue
        cells = []
        for _, lab in EP:
            v = vals[lab]
            cells.append(f'{st.mean(v):.4f}+-{(st.stdev(v) if len(v)>1 else 0):.4f}')
        oldcell = ' / '.join(f'{old[lab]:.3f}' for _, lab in EP)
        print(f'{SHORT[c]:11s} {n:5d}  ' + '  '.join(f'{x:>18s}' for x in cells) +
              f'  {oldcell:>22s}')
        rows[c] = {'n': n, 'new': {lab: vals[lab] for _, lab in EP},
                   'delta': {lab: dels[lab] for _, lab in EP},
                   'old': old}
    out['conditions'] = rows

    print()
    print('paired delta vs the archived two-stage, in percentage points:'
          ' mean +- SD over seeds, then each seed')
    print(f'{"condition":11s} {"endpoint":10s} {"mean":>8s} {"SD":>7s} {"worst":>8s}   per-seed')
    for c in CONDS:
        if c not in rows:
            continue
        for _, lab in EP:
            d = [100*x for x in rows[c]['delta'][lab]]
            worst = min(d) if st.mean(d) >= 0 else max(d)
            per = ' '.join(f'{x:+6.2f}' for x in d)
            print(f'{SHORT[c] if lab=="strict" else "":11s} {lab:10s} '
                  f'{st.mean(d):+8.2f} {(st.stdev(d) if len(d)>1 else 0):7.2f} '
                  f'{worst:+8.2f}   {per}')
        print()

    if a.output:
        Path(a.output).write_text(json.dumps(out, indent=1), encoding='utf-8')
        print('wrote', a.output)


if __name__ == '__main__':
    main()
