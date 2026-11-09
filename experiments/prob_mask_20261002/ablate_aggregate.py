"""Aggregate pose_ablate_search.py runs into the search-vs-filter decomposition.

The published greedy -> local-MAP gain bundles two changes: maximising the joint
score over a coupled neighbourhood, and a hard buildability filter that outranks
the score. This separates them by recomputing every selector variant offline
from the recorded beam, so no model is re-run and nothing is tuned.

Selectors (none of them consults a reference label):

  greedy              candidate 0 (per-edge argmax), no filter
  greedy+filter(d=0)  the filter with no alternatives to choose from
  search only         argmax score over the whole beam, filter ignored
  d<=k + filter       argmax (buildable, score) over Hamming distance <= k
  full localmap       argmax (buildable, score) over the whole beam

Standard library only.
"""
import argparse
import json
import statistics as st
from pathlib import Path


def pick(cands, maxdist=None, use_filter=True):
    pool = [x for x in cands if maxdist is None or x['dist'] <= maxdist]
    if not pool:
        pool = [x for x in cands if x['dist'] == 0]
    key = (lambda x: (x['good'], x['score'])) if use_filter else (lambda x: x['score'])
    return max(pool, key=key)


def pick_bond_only(cands):
    """First buildable candidate when the SAME beam is ordered by edge probability.

    Lexicographic max on (good, bscore) is exactly 'walk the pool in descending
    sum log P(B) and take the first one that builds', with the highest-bscore
    candidate as the fallback when none builds. This is the rung that isolates
    the conditional atom-state term: it has the identical candidate budget and
    the identical hard filter as `full localmap`, and differs only in not using
    log P(h,q | B) to rank.
    """
    return max(cands, key=lambda x: (x['good'], x['bscore']))


SELECTORS = [
    ('greedy (baseline)',      lambda c: next(x for x in c if x['dist'] == 0)),
    ('greedy+filter (d=0)',    lambda c: pick(c, 0, True)),
    ('search only, no filter', lambda c: pick(c, None, False)),
    ('d<=1 + filter',          lambda c: pick(c, 1, True)),
    ('d<=2 + filter',          lambda c: pick(c, 2, True)),
    ('first buildable by P(B)', pick_bond_only),
    ('full localmap',          lambda c: pick(c, None, True)),
]

# selectors that need fields older dumps do not carry
NEEDS = {'first buildable by P(B)': 'bscore'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True,
                    help='pose_localmap_<tag>.json files from pose_ablate_search.py')
    ap.add_argument('--labels', nargs='*', default=None)
    ap.add_argument('--output', default='')
    a = ap.parse_args()
    labels = a.labels or [Path(p).parent.name or f'run{i}' for i, p in enumerate(a.runs)]
    data = []
    for p in a.runs:
        recs = json.loads(Path(p).read_text(encoding='utf-8'))['records']
        if not recs or 'candidates_detail' not in recs[0]:
            raise SystemExit(f'{p} has no candidates_detail; run pose_ablate_search.py, '
                             f'not pose_localmap.py')
        data.append(recs)
    names = [{r['name'] for r in recs} for recs in data]
    if any(len(n) != len(recs) for n, recs in zip(names, data)) or any(n != names[0] for n in names[1:]):
        raise ValueError('Ablation seeds must have identical, unique molecule names')

    table, fails = {}, {}
    print(f'{"selector":26s}' + ''.join(f'{l:>11s}' for l in labels) +
          f'{"mean":>9s}{"SD":>8s}   buildfail')
    for lab, fn in SELECTORS:
        need = NEEDS.get(lab)
        if need and not all(need in c for recs in data for r in recs for c in r['candidates_detail']):
            print(f'{lab:26s}  skipped: dumps carry no {need!r} field '
                  f'(re-run pose_ablate_search.py to get it)')
            continue
        vals, bf = [], []
        for recs in data:
            sel = [fn(r['candidates_detail']) for r in recs]
            vals.append(sum(x['correct'] for x in sel)/len(recs))
            bf.append(sum(not x['good'] for x in sel))
        table[lab], fails[lab] = vals, bf
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        print(f'{lab:26s}' + ''.join(f'{v:11.4f}' for v in vals) +
              f'{st.mean(vals):9.4f}{sd:8.4f}   {"/".join(map(str, bf))}')

    g, f = table['greedy (baseline)'], table['full localmap']
    tot = [f[i]-g[i] for i in range(len(data))]
    sdt = st.stdev(tot) if len(tot) > 1 else 0.0
    print(f'\ngreedy -> full: {100*st.mean(tot):+.2f}pp +- {100*sdt:.2f}')
    print('decomposition (share computed per run, then averaged):')
    out = {'labels': labels, 'selectors': table, 'buildfail': fails, 'shares': {}}
    for lab in ['greedy+filter (d=0)', 'search only, no filter', 'd<=1 + filter',
                'd<=2 + filter', 'first buildable by P(B)']:
        if lab not in table:
            continue
        dd = [100*(table[lab][i]-g[i]) for i in range(len(data))]
        fr = [(table[lab][i]-g[i])/(f[i]-g[i]) if f[i] != g[i] else float('nan')
              for i in range(len(data))]
        sdd = st.stdev(dd) if len(dd) > 1 else 0.0
        sfr = st.stdev(fr) if len(fr) > 1 else 0.0
        print(f'  greedy -> {lab:24s} {st.mean(dd):+6.2f}pp +- {sdd:.2f}   '
              f'share {100*st.mean(fr):5.1f}% +- {100*sfr:.1f}%')
        out['shares'][lab] = {'delta_pp': dd, 'share': fr}
    cpl = [100*(table['d<=2 + filter'][i]-table['d<=1 + filter'][i]) for i in range(len(data))]
    sc = st.stdev(cpl) if len(cpl) > 1 else 0.0
    print(f'\ngenuinely coupled part (d<=2 minus d<=1): {st.mean(cpl):+.2f}pp +- {sc:.2f}')
    oracle = [sum(any(x['exactB'] for x in r['candidates_detail']) for r in recs)/len(recs)
              for recs in data]
    so = st.stdev(oracle) if len(oracle) > 1 else 0.0
    print(f'reference assignment inside the beam: {st.mean(oracle):.4f} +- {so:.4f}')
    print('The oracle above is EXACT-B membership, whereas selector correctness is '
          'native molecular identity. They are different endpoints; do not divide '
          'them or infer the bottleneck from their ratio.')
    out['oracle_in_beam'] = oracle
    if a.output:
        Path(a.output).write_text(json.dumps(out, indent=1), encoding='utf-8')
        print('wrote', a.output)


if __name__ == '__main__':
    main()
