"""Three-sampling-seed re-check of the in-distribution validation split.

Why: the single-seed run of the t-conditioned ``diffusion`` checkpoint showed the
largest single gain anywhere in this line of work -- charge_challenge species
coverage 0.572 (w05) / 0.564 (w1) against the baseline's three-seed means of
0.532 / 0.525 -- but the baseline's own sampling-seed SD is about 1.3-2 points, so
one seed cannot support it. This aggregates several ``improve.py evaluate`` runs
that differ only in ``--seed`` and reports, per model:

  * per-seed coverage and the sampling-seed mean and sample SD;
  * a paired comparison against a reference model, where each molecule's hit
    indicator is first averaged over the seeds and the mean difference then gets a
    95% interval from a bootstrap over *connectivity skeletons*, which is the
    protocol the earlier multi-seed ledger used (``joint_v1/MULTISEED.md``).

Standard library only, so it runs under any Python that can read the JSON.

Caveats this script cannot remove, and that belong next to any number it prints:
the 550-molecule validation split is drawn from the frozen backbone's own
train+val pool, so it is not held out for the backbone; the intervals are
sampling-seed intervals for one trained model each, not training-seed intervals;
and no multiple-comparison correction is applied.
"""
import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_args():
    a = argparse.ArgumentParser()
    a.add_argument('--dir', default=str(HERE/'joint_v2'))
    a.add_argument('--models', nargs='+', default=['baseline', 'diffusion'])
    a.add_argument('--seeds', nargs='+', default=['20261002', '20261003', '20261004'])
    a.add_argument('--pattern', default='ms_{model}_s{seed}.json')
    a.add_argument('--reference', default='baseline',
                   help='model the others are compared against')
    a.add_argument('--sigma', type=float, default=0.2)
    a.add_argument('--configs', nargs='+', default=['w05', 'w1'])
    a.add_argument('--subsets', nargs='+',
                   default=['geom_core', 'charge_challenge', 'nH'])
    a.add_argument('--reps', type=int, default=2000)
    a.add_argument('--bootstrap_seed', type=int, default=20261004)
    a.add_argument('--output', default=str(HERE/'joint_v2/charge_multiseed.json'))
    return a.parse_args()


SUBSET_RULE = {
    'all': lambda r: True,
    'geom_core': lambda r: r['domain'] == 'geom_core',
    'charge_challenge': lambda r: r['domain'] == 'diverse_charge',
    'net_charged': lambda r: r['Q'] != 0,
    'nH': lambda r: r['has_nH'],
}


def load(path, sigma, config):
    """{molecule id: record} for one run, restricted to one sigma and config."""
    blob = json.loads(Path(path).read_text(encoding='utf-8'))
    out = {}
    for r in blob['records']:
        if abs(r['sigma']-sigma) < 1e-9 and r['method'] == config:
            out[r['id']] = r
    return out, blob.get('checkpoint_epoch')


def cluster_bootstrap(deltas, clusters, reps, seed):
    """95% interval for the mean of per-molecule deltas, resampling clusters."""
    groups = defaultdict(list)
    for d, c in zip(deltas, clusters):
        groups[c].append(d)
    keys = list(groups)
    if not keys:
        return None, None
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        pool = []
        for _ in range(len(keys)):
            pool.extend(groups[keys[rng.randrange(len(keys))]])
        if pool:
            means.append(sum(pool)/len(pool))
    means.sort()
    lo = means[int(0.025*len(means))]
    hi = means[min(len(means)-1, int(0.975*len(means)))]
    return lo, hi


def main():
    a = parse_args()
    runs = {}
    missing = []
    for m in a.models:
        for s in a.seeds:
            path = Path(a.dir)/a.pattern.format(model=m, seed=s)
            if path.exists():
                runs[(m, s)] = path
            else:
                missing.append(str(path))
    if missing:
        print('missing runs:')
        for x in missing:
            print('  ', x)
    if not runs:
        raise SystemExit('no runs found')

    report = {'args': vars(a), 'per_model': {}, 'paired': {}}
    for config in a.configs:
        loaded = {}
        for (m, s), path in runs.items():
            recs, epoch = load(path, a.sigma, config)
            if recs:
                loaded[(m, s)] = recs
        models = sorted({m for m, _ in loaded})
        print(f'\n===== sigma={a.sigma:g}  config={config}  '
              f'(species coverage@8, mean over sampling seeds) =====')
        header = f"{'model':11s} {'subset':17s} {'n':>4s} " + \
                 ' '.join(f'{s[-4:]:>7s}' for s in a.seeds) + \
                 f" {'mean':>7s} {'SD':>6s} {'valid':>6s} {'uniq':>5s}"
        print(header)
        for m in models:
            for subset in a.subsets:
                keep = SUBSET_RULE[subset]
                per_seed, valid, uniq, n = [], [], [], None
                for s in a.seeds:
                    recs = loaded.get((m, s))
                    if recs is None:
                        per_seed.append(None)
                        continue
                    sel = [r for r in recs.values() if keep(r)]
                    n = len(sel)
                    per_seed.append(sum(any(r['species_hits']) for r in sel)/max(n, 1))
                    valid.append(sum(r['valid_count'] for r in sel)/max(8*n, 1))
                    uniq.append(sum(r['valid_unique'] for r in sel)/max(n, 1))
                vals = [x for x in per_seed if x is not None]
                if not vals:
                    continue
                mean = sum(vals)/len(vals)
                sd = statistics.stdev(vals) if len(vals) > 1 else float('nan')
                cells = ' '.join(f'{x:7.4f}' if x is not None else f'{"-":>7s}'
                                 for x in per_seed)
                print(f'{m:11s} {subset:17s} {n:>4d} {cells} {mean:7.4f} '
                      f'{sd:6.4f} {sum(valid)/len(valid):6.4f} {sum(uniq)/len(uniq):5.2f}')
                report['per_model'].setdefault(config, {}).setdefault(m, {})[subset] = {
                    'n': n, 'per_seed': per_seed, 'mean': mean,
                    'sd': None if math.isnan(sd) else sd,
                    'valid_fraction': sum(valid)/len(valid),
                    'valid_unique': sum(uniq)/len(uniq)}

        ref = a.reference
        if ref not in models:
            continue
        print(f'\n--- paired vs {ref}: per-molecule mean over seeds, '
              f'skeleton cluster bootstrap 95% CI ---')
        for m in models:
            if m == ref:
                continue
            for subset in a.subsets:
                keep = SUBSET_RULE[subset]
                ids = None
                for s in a.seeds:
                    for who in (ref, m):
                        recs = loaded.get((who, s))
                        if recs is None:
                            continue
                        here = {k for k, r in recs.items() if keep(r)}
                        ids = here if ids is None else (ids & here)
                if not ids:
                    continue
                deltas, clusters = [], []
                for mid in sorted(ids):
                    ra, rb = [], []
                    skel = None
                    for s in a.seeds:
                        if (ref, s) in loaded and mid in loaded[(ref, s)]:
                            r = loaded[(ref, s)][mid]
                            ra.append(any(r['species_hits']))
                            skel = r['skeleton']
                        if (m, s) in loaded and mid in loaded[(m, s)]:
                            rb.append(any(loaded[(m, s)][mid]['species_hits']))
                    if not ra or not rb:
                        continue
                    deltas.append(sum(rb)/len(rb)-sum(ra)/len(ra))
                    clusters.append(skel)
                if not deltas:
                    continue
                delta = sum(deltas)/len(deltas)
                lo, hi = cluster_bootstrap(deltas, clusters, a.reps, a.bootstrap_seed)
                better = sum(1 for d in deltas if d > 0)
                worse = sum(1 for d in deltas if d < 0)
                flag = '' if (lo is None or lo <= 0 <= hi) else '  <- interval excludes 0'
                print(f'{m:11s} {subset:17s} n={len(deltas):>4d} '
                      f'delta={delta*100:+6.2f}pp  95% CI [{lo*100:+6.2f},{hi*100:+6.2f}]pp  '
                      f'molecules better/worse {better}/{worse}{flag}')
                report['paired'].setdefault(config, {}).setdefault(m, {})[subset] = {
                    'n': len(deltas), 'delta': delta, 'ci': [lo, hi],
                    'molecules_better': better, 'molecules_worse': worse,
                    'n_skeletons': len(set(clusters))}

    Path(a.output).write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(f'\nwrote {a.output}')
    print('Reminder: this split is not held out for the frozen backbone, the '
          'intervals are sampling-seed only (one trained model each), and no '
          'multiple-comparison correction is applied.')


if __name__ == '__main__':
    main()
