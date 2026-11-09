"""Aggregate the search-vs-filter ablation over a condition x training-seed grid.

Reads joint_v7/<ckpt>/<condition>/pose_localmap_<tag>.json written by
pose_ablate_search.py and prints, per condition, the mean +- SD over seeds of

  * the decoder's total gain (greedy -> full local MAP),
  * the share of that gain carried by score maximisation alone (no filter),
  * the share carried by single-edge moves under the buildability filter,
  * the genuinely coupled part (distance <=2 minus distance <=1).

The question this answers: does the mechanism change with coordinate noise?
Standard library only.
"""
import argparse
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHORT = {'pdbbind_sigma0p00': 'pose 0.00', 'pdbbind_sigma0p05': 'pose 0.05',
         'pdbbind_sigma0p10': 'pose 0.10', 'pdbbind_sigma0p15': 'pose 0.15',
         'ccd_sigma0p00': 'CCD  0.00', 'ccd_sigma0p10': 'CCD  0.10'}


def pick(c, maxdist=None, use_filter=True):
    pool = [x for x in c if maxdist is None or x['dist'] <= maxdist]
    if not pool:
        pool = [x for x in c if x['dist'] == 0]
    key = (lambda x: (x['good'], x['score'])) if use_filter else (lambda x: x['score'])
    return max(pool, key=key)


SEL = {
    'greedy':      lambda c: next(x for x in c if x['dist'] == 0),
    'filter_d0':   lambda c: pick(c, 0, True),
    'search_only': lambda c: pick(c, None, False),
    'd1':          lambda c: pick(c, 1, True),
    'd2':          lambda c: pick(c, 2, True),
    'full':        lambda c: pick(c, None, True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(HERE/'joint_v7'))
    ap.add_argument('--seeds', default='seed20261002,seed20261003,seed20261004')
    ap.add_argument('--conditions', default='pdbbind_sigma0p00,pdbbind_sigma0p05,'
                                            'pdbbind_sigma0p10,pdbbind_sigma0p15,'
                                            'ccd_sigma0p00,ccd_sigma0p10')
    ap.add_argument('--tag', default='abl')
    ap.add_argument('--output', default='')
    a = ap.parse_args()
    seeds = [s.strip() for s in a.seeds.split(',') if s.strip()]
    conds = [c.strip() for c in a.conditions.split(',') if c.strip()]
    root = Path(a.root)
    out = {'seeds': seeds, 'conditions': {}}

    print(f'{"condition":11s} {"n":>5s} {"greedy":>14s} {"full":>14s} {"gain pp":>13s}'
          f' {"score share":>16s} {"d<=1 share":>16s} {"coupled pp":>13s} {"oracle":>14s}')
    for c in conds:
        vals = {k: [] for k in SEL}
        oracle, ns = [], []
        missing = []
        for s in seeds:
            p = root/s/c/f'pose_localmap_{a.tag}.json'
            if not p.exists():
                missing.append(s)
                continue
            recs = json.loads(p.read_text(encoding='utf-8'))['records']
            ns.append(len(recs))
            for k, fn in SEL.items():
                sel = [fn(r['candidates_detail']) for r in recs]
                vals[k].append(sum(x['correct'] for x in sel)/len(recs))
            oracle.append(sum(any(x['exactB'] for x in r['candidates_detail'])
                              for r in recs)/len(recs))
        if not ns:
            print(f'{SHORT.get(c, c):11s}  -- not run --')
            continue
        if missing:
            print(f'   (note: {c} missing seeds {missing})')

        def ms(xs):
            return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)
        g, gs = ms(vals['greedy']); f, fs = ms(vals['full'])
        gains = [100*(vals['full'][i]-vals['greedy'][i]) for i in range(len(ns))]
        gm, gsd = ms(gains)
        shr_s = [(vals['search_only'][i]-vals['greedy'][i])/(vals['full'][i]-vals['greedy'][i])
                 for i in range(len(ns)) if vals['full'][i] != vals['greedy'][i]]
        shr_1 = [(vals['d1'][i]-vals['greedy'][i])/(vals['full'][i]-vals['greedy'][i])
                 for i in range(len(ns)) if vals['full'][i] != vals['greedy'][i]]
        cpl = [100*(vals['d2'][i]-vals['d1'][i]) for i in range(len(ns))]
        sm, ssd = ms(shr_s); dm, dsd = ms(shr_1); cm, csd = ms(cpl)
        om, osd = ms(oracle)
        assert all(x == vals['greedy'][i] for i, x in enumerate(vals['filter_d0'])), \
            'the filter changed something with no alternatives; that cannot happen'
        print(f'{SHORT.get(c, c):11s} {ns[0]:5d} {g:8.4f}±{gs:.4f} {f:8.4f}±{fs:.4f} '
              f'{gm:+8.2f}±{gsd:4.2f} {100*sm:9.1f}%±{100*ssd:4.1f} '
              f'{100*dm:9.1f}%±{100*dsd:4.1f} {cm:+8.2f}±{csd:4.2f} {om:8.4f}±{osd:.4f}')
        out['conditions'][c] = {
            'n': ns[0], 'seeds_present': len(ns),
            'selectors': {k: vals[k] for k in SEL},
            'gain_pp': gains, 'score_share': shr_s, 'd1_share': shr_1,
            'coupled_pp': cpl, 'oracle_in_beam': oracle}
    print()
    print('score share = greedy -> argmax score with the filter ignored')
    print('d<=1 share  = greedy -> single-edge move under the buildability filter')
    print('coupled     = distance<=2 minus distance<=1, i.e. moves that need two edges at once')
    print('filter_d0 (the filter with no alternatives) equals greedy in every cell, as asserted')
    if a.output:
        Path(a.output).write_text(json.dumps(out, indent=1), encoding='utf-8')
        print('wrote', a.output)


if __name__ == '__main__':
    main()
