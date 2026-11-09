"""Training-seed spread on the pose endpoints.

Reads the published-protocol scorer output for each training seed and reports the
per-seed value, the mean and the sample SD, for every endpoint and selector. The
point is to replace "one trained network scored 0.8381" with a number that has a
training-seed interval attached, which is what the comparison baseline already had
(three seeds) and the new pipeline did not.

Standard library only.
"""
import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_args():
    a = argparse.ArgumentParser()
    a.add_argument('--dir', default=str(HERE/'joint_v5'))
    a.add_argument('--seeds', nargs='+',
                   default=['20261002', '20261003', '20261004'])
    a.add_argument('--selectors', nargs='+', default=['greedy', 'localmap'])
    a.add_argument('--pattern', default='seed{seed}/species_{selector}.json')
    a.add_argument('--training', default='seed{seed}/training.json')
    a.add_argument('--output', default=str(HERE/'joint_v5/seeds_summary.json'))
    return a.parse_args()


KEYS = [('classify_recovery', 'resonance-aware'),
        ('species_recovery', 'canonical-SMILES'),
        ('skeleton_recovery', 'neutral skeleton')]


def main():
    a = parse_args()
    root = Path(a.dir)
    report = {'args': vars(a), 'training': {}, 'endpoints': {}}

    print('=== training ===')
    print(f"{'seed':10s} {'epoch':>5s} {'sum CE':>8s}   (joint_v1 reference: epoch 13, 0.11626)")
    for s in a.seeds:
        path = root/a.training.format(seed=s)
        if not path.exists():
            print(f'{s:10s} missing {path}')
            continue
        h = json.loads(path.read_text(encoding='utf-8'))['history']
        best = min(h, key=lambda r: r['val_bond_CE']+r['val_atom_joint_CE'])
        ce = best['val_bond_CE']+best['val_atom_joint_CE']
        print(f'{s:10s} {best["epoch"]:>5d} {ce:8.5f}')
        report['training'][s] = {'epoch': best['epoch'], 'sum_ce': ce}

    for sel in a.selectors:
        rows = {}
        for s in a.seeds:
            path = root/a.pattern.format(seed=s, selector=sel)
            if not path.exists():
                continue
            d = json.loads(path.read_text(encoding='utf-8'))
            rows[s] = d
        if not rows:
            print(f'\n=== {sel}: no scorer output found ===')
            continue
        print(f'\n=== {sel}: published-protocol endpoints across training seeds ===')
        header = f"{'endpoint':20s} " + ' '.join(f'{s[-4:]:>8s}' for s in a.seeds) + \
                 f" {'mean':>8s} {'SD':>7s}"
        print(header)
        for key, label in KEYS:
            vals = [rows[s][key] if s in rows and key in rows[s] else None
                    for s in a.seeds]
            got = [v for v in vals if v is not None]
            if not got:
                continue
            mean = sum(got)/len(got)
            sd = statistics.stdev(got) if len(got) > 1 else float('nan')
            cells = ' '.join(f'{v:8.4f}' if v is not None else f'{"-":>8s}' for v in vals)
            print(f'{label:20s} {cells} {mean:8.4f} {sd:7.4f}')
            report['endpoints'].setdefault(sel, {})[label] = {
                'per_seed': vals, 'mean': mean,
                'sd': None if len(got) < 2 else sd}
        fails = [rows[s].get('prediction_build_failures') for s in a.seeds if s in rows]
        if any(f is not None for f in fails):
            print(f'{"build failures":20s} ' +
                  ' '.join(f'{f:8d}' if f is not None else f'{"-":>8s}' for f in fails))
            report['endpoints'].setdefault(sel, {})['build_failures'] = fails

    Path(a.output).write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(f'\nwrote {a.output}')
    print('The SD here is over training seeds for a fixed decoder and a fixed '
          'sampling path; it does not cover sampling seeds, and the 488-pose set '
          'is a development set.')


if __name__ == '__main__':
    main()
