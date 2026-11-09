"""Summarize fixed-seed matched pilots; the unit remains the molecule, not seed × molecule."""
import argparse
import json
import statistics as st
from pathlib import Path

MODES = ('known_q', 'frozen_fixed', 'frozen_free', 'frozen_multi_q', 'trained_free', 'trained_multi_q')
DATASETS = {'pose': 488, 'ccd_dev': 332}
PAIRS = (('frozen_free', 'trained_free'), ('frozen_multi_q', 'trained_multi_q'),
         ('frozen_fixed', 'trained_multi_q'))
METRICS = ('native', 'net_correct', 'buildable', 'strict', 'resonance', 'graph_smiles',
           'q_exact', 'q_macro_f1', 'native_when_net_correct')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def path_at(root, seed, ds, mode, suffix='.json'):
    name = f'{ds}_{mode}{suffix}'
    path = root/f'matched_s{seed}'/name
    if not path.exists() and not mode.startswith('trained_'):
        path = root/f'frozen_s{seed}'/name
    return path


def cell_at(root, seed, ds, mode):
    return load(path_at(root, seed, ds, mode))


def mean_sd(items):
    vals = [x for x in items if x is not None]
    return {'mean': st.mean(vals) if vals else None, 'sd': st.stdev(vals) if len(vals)>1 else None,
            'values': items}


def matched(rows_a, rows_b):
    if [(r['name'], r['q_ref']) for r in rows_a] != [(r['name'], r['q_ref']) for r in rows_b]:
        raise ValueError('Mismatched prediction rows')
    good_a, good_b = [sum(r['native'] for r in rr) for rr in (rows_a, rows_b)]
    gain = sum(not a['native'] and b['native'] for a,b in zip(rows_a, rows_b))
    loss = sum(a['native'] and not b['native'] for a,b in zip(rows_a, rows_b))
    both_q = [(a,b) for a,b in zip(rows_a, rows_b) if a['net_correct'] and b['net_correct']]
    return dict(n=len(rows_a), delta=(good_b-good_a)/len(rows_a), gain=gain, loss=loss,
                both_net_correct=len(both_q), a_native_among_both_net_correct=sum(a['native'] for a,_ in both_q),
                b_native_among_both_net_correct=sum(b['native'] for _,b in both_q))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[20261002,20261003,20261004])
    ap.add_argument('--datasets', nargs='+', choices=tuple(DATASETS), default=list(DATASETS))
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = {'seeds': a.seeds, 'note': 'Seed means and sample SD, shared encoder and molecules; descriptive development pilot.'}
    for ds in a.datasets:
        n = DATASETS[ds]
        rec = {s:{m:cell_at(a.root, s, ds, m) for m in MODES} for s in a.seeds}
        names = [r['name'] for r in rec[a.seeds[0]]['known_q']['records']]
        assert len(names) == len(set(names)) == n
        for cells in rec.values():
            for cell in cells.values():
                assert [r['name'] for r in cell['records']] == names
                assert all(not r['native'] or r['net_correct'] for r in cell['records'])
        groups = rec[a.seeds[0]]['known_q']['summary']
        agg = {g:{m:{k:mean_sd([rec[s][m]['summary'][g].get(k) for s in a.seeds]) for k in METRICS}
                  for m in MODES} for g in groups}
        splits = {'all':lambda r:True, 'net_neutral':lambda r:r['net_ref']==0,
                  'net_charged':lambda r:r['net_ref']!=0}
        paired = {}
        for g, include in splits.items():
            paired[g] = {f'{m2}_minus_{m1}':{str(s): matched(
                [r for r in rec[s][m1]['records'] if include(r)],
                [r for r in rec[s][m2]['records'] if include(r)]) for s in a.seeds} for m1,m2 in PAIRS}
        result[ds] = {'group_n':{g:d['n'] for g,d in groups.items()}, 'metrics':agg, 'paired':paired}
        for m in MODES:
            print(ds, m, '  '.join(f'{g}: native={100*agg[g][m]["native"]["mean"]:.2f}, '
                                   f'net={100*agg[g][m]["net_correct"]["mean"]:.2f}'
                                   for g in splits))
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
