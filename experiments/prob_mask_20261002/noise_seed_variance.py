"""How much of the reported spread is training seed, and how much is the noise draw?

Every perturbed condition in the paper draws its coordinates once. The reported
standard deviations are across three training seeds, so they do not contain the
variance of the noise realisation itself. A reader cannot tell whether a gap of a
few points between two noise levels is larger than the sampling noise of one draw.

This compares the two sources directly. Decode the same molecules at the same
sigma under several noise seeds with one fixed training seed; that spread is the
noise-sampling component. Put it next to the across-training-seed spread already
reported for the same condition.

    python noise_seed_variance.py --runs <json> <json> ... --label "pose sigma=0.10"
                                  [--training_seed_sd 0.0043] [--output <json>]

Each run is a pose_localmap.py summary json produced with the same --sigma and a
different --noise_seed. Standard library only.
"""
import argparse
import json
import statistics as st
from pathlib import Path

FIELDS = [('localmap_species', 'native identity'),
          ('localmap_strictB', 'strict bond graph'),
          ('localmap_valid', 'constructible'),
          ('baseline_species', 'argmax native (stage A)')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--label', default='')
    ap.add_argument('--training_runs', nargs='*', default=[],
                    help='runs that vary the TRAINING seed at one fixed noise '
                         'seed, on the same molecules. Required for a meaningful '
                         'ratio: an SD measured on a different molecule count is '
                         'not comparable, because sampling error scales with n.')
    ap.add_argument('--training_seed_sd', type=float, default=None,
                    help='fallback only. A published SD from the full benchmark '
                         'is NOT comparable with a subset run and is reported as '
                         'context, never as a ratio.')
    ap.add_argument('--output', default='')
    a = ap.parse_args()

    runs = []
    for path in a.runs:
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        summary = d['summary'] if 'summary' in d else d
        args = summary.get('args', {})
        if 'noise_seed' not in args:
            raise SystemExit(f'{path}: no noise_seed recorded; this run predates '
                             'the --noise_seed option and cannot be compared')
        runs.append((args['noise_seed'], args.get('sigma'), summary))
    sigmas = {r[1] for r in runs}
    if len(sigmas) != 1:
        raise SystemExit(f'runs mix sigma values {sigmas}; compare one condition')
    seeds = [r[0] for r in runs]
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f'duplicate noise seeds {seeds}; each run needs its own')
    print(f'{a.label or "condition"}: sigma={sigmas.pop()}, '
          f'{len(runs)} noise seeds {sorted(seeds)}, one fixed training seed\n')

    out = {'label': a.label, 'noise_seeds': sorted(seeds), 'endpoints': {}}
    print(f'  {"endpoint":26s}{"mean":>9s}{"noise-seed SD":>15s}{"range":>18s}')
    for key, name in FIELDS:
        vals = [s[key] for _, _, s in runs if key in s]
        if len(vals) < 2:
            continue
        sd = st.stdev(vals)
        print(f'  {name:26s}{st.mean(vals):9.4f}{sd:15.4f}'
              f'{min(vals):9.4f}-{max(vals):.4f}')
        out['endpoints'][key] = {'values': vals, 'mean': st.mean(vals), 'sd': sd}

    nsd = out['endpoints'].get('localmap_species', {}).get('sd')
    if a.training_runs:
        tvals, tn = [], set()
        for path in a.training_runs:
            d = json.loads(Path(path).read_text(encoding='utf-8'))
            summary = d['summary'] if 'summary' in d else d
            tvals.append(summary['localmap_species']); tn.add(summary['n'])
        nn = {s['n'] for _, _, s in runs}
        if tn != nn:
            raise SystemExit(f'molecule counts differ: noise runs {nn}, '
                             f'training runs {tn}. The ratio would be meaningless.')
        tsd = st.stdev(tvals)
        print(f'\n  matched comparison on the same {nn.pop()} molecules, '
              f'native identity:')
        print(f'    across {len(tvals)} training seeds, one noise seed : SD {tsd:.4f}')
        print(f'    across {len(runs)} noise seeds, one training seed  : SD {nsd:.4f}')
        if tsd > 0:
            ratio = nsd/tsd
            print(f'    ratio {ratio:.2f}x')
            print('    The published interval is across training seeds only, so it '
                  f'{"understates" if ratio > 1 else "already covers"} the')
            print('    variance of drawing the perturbation once.')
            out['matched_training_sd'] = tsd
            out['ratio_noise_to_training'] = ratio
        out['training_values'] = tvals
    elif a.training_seed_sd is not None and nsd is not None:
        print(f'\n  across-noise-seed SD (this run, n={ {s["n"] for _,_,s in runs} }): '
              f'{nsd:.4f}')
        print(f'  published across-training-seed SD: {a.training_seed_sd:.4f}')
        print('  NOT comparable: the published value comes from the full benchmark '
              'and this run is a subset.')
        print('  Pass --training_runs on the same molecules to get a ratio.')
        out['published_training_seed_sd_not_comparable'] = a.training_seed_sd
    if a.output:
        Path(a.output).write_text(json.dumps(out, indent=1), encoding='utf-8')
        print(f'\nwrote {a.output}')


if __name__ == '__main__':
    main()
