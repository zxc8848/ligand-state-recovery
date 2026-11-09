"""Aggregate convention_split.py across training seeds.

On the no-delocalizable-group subset the three endpoints collapse into one
(the resonance scorer has nothing to forgive), so that subset gives a single
unambiguous number per condition -- the cleanest noise-response curve available.
Standard library only.
"""
import argparse
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ['ccd_sigma0p00', 'pdbbind_sigma0p00', 'pdbbind_sigma0p05',
         'ccd_sigma0p10', 'pdbbind_sigma0p10', 'pdbbind_sigma0p15']
SHORT = {'ccd_sigma0p00': 'CCD  ideal  sigma 0.00', 'pdbbind_sigma0p00': 'pose sigma 0.00',
         'pdbbind_sigma0p05': 'pose sigma 0.05', 'ccd_sigma0p10': 'CCD  sigma 0.10',
         'pdbbind_sigma0p10': 'pose sigma 0.10', 'pdbbind_sigma0p15': 'pose sigma 0.15'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='seed20261002,seed20261003,seed20261004')
    ap.add_argument('--kind', default='fullgraph', choices=['fullgraph', 'refedges'],
                    help='which convention_split output to read. fullgraph reads '
                         'convention_split_fullgraph.json, refedges reads '
                         'convention_split.json (the published-protocol variant).')
    ap.add_argument('--output', default='')
    a = ap.parse_args()
    seeds = [s.strip() for s in a.seeds.split(',') if s.strip()]
    blobs = {}
    for s in seeds:
        suf = '' if a.kind == 'refedges' else '_fullgraph'
        p = HERE/'joint_v6'/s/f'convention_split{suf}.json'
        if not p.exists():
            raise SystemExit(f'missing {p}; run convention_split.py --ckpt {s} --kind {a.kind} first')
        blobs[s] = json.loads(p.read_text(encoding='utf-8'))['conditions']

    out = {'seeds': seeds, 'kind': a.kind, 'subsets': {}}
    for subset, tags in (('reference: no deloc group', ['strict']),
                         ('reference: has deloc group', ['strict', 'reson', 'SMILES']),
                         ('all (published)', ['strict', 'reson', 'SMILES'])):
        print(f'### {subset}')
        hdr = f'{"condition":24s} {"n":>5s}'
        for t in tags:
            hdr += f'{t+" mean":>13s}{"SD":>7s}{"worst":>8s}  sig'
        print(hdr)
        rows = {}
        for c in ORDER:
            if c not in blobs[seeds[0]] or subset not in blobs[seeds[0]][c]:
                continue
            n = blobs[seeds[0]][c][subset]['n']
            line = f'{SHORT[c]:24s} {n:5d}'
            rec = {'n': n}
            for t in tags:
                d = [100*blobs[s][c][subset][t]['delta'] for s in seeds]
                ps = [blobs[s][c][subset][t]['p'] for s in seeds]
                sd = st.stdev(d) if len(d) > 1 else 0.0
                worst = min(d) if st.mean(d) >= 0 else max(d)
                nsig = sum(1 for x in ps if x < 0.05)
                line += f'{st.mean(d):+13.2f}{sd:7.2f}{worst:+8.2f}  {nsig}/{len(ps)}'
                rec[t] = {'per_seed': d, 'p': ps, 'mean': st.mean(d), 'sd': sd,
                          'worst': worst, 'n_significant': nsig}
            print(line)
            rows[c] = rec
        out['subsets'][subset] = rows
        print()
    if a.output:
        Path(a.output).write_text(json.dumps(out, indent=1), encoding='utf-8')
        print('wrote', a.output)


if __name__ == '__main__':
    main()
