"""Summarize every improvement arm and paired skeleton-cluster intervals."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent / 'joint_v2'


@lru_cache(maxsize=None)
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def subset(records, name):
    if name == 'geom_core':
        return [r for r in records if r['domain'] == 'geom_core']
    if name == 'charge_challenge':
        return [r for r in records if r['domain'] == 'diverse_charge']
    return [r for r in records if r['has_nH']]


def paired(files, method, name, sigma=.2):
    values = {}
    for file in files:
        data = read(file)
        records = subset(data['records'], name)
        left = {r['id']: r for r in records if r['sigma'] == sigma and r['method'] == method}
        right = {r['id']: r for r in records if r['sigma'] == sigma and r['method'] == 'w1'}
        if not left or not right:
            continue
        assert left.keys() == right.keys()
        for key in left:
            values.setdefault(key, dict(skeleton=left[key]['skeleton'], a=[], b=[]))
            values[key]['a'].append(float(any(left[key]['species_hits'])))
            values[key]['b'].append(float(any(right[key]['species_hits'])))
    if not values:
        return None
    groups = {}
    for value in values.values():
        groups.setdefault(value['skeleton'], []).append(np.mean(value['a']) - np.mean(value['b']))
    totals = np.array([sum(v) for v in groups.values()]); counts = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(20261003); samples = []
    for _ in range(4000):
        ix = rng.integers(len(groups), size=len(groups))
        samples.append(totals[ix].sum() / counts[ix].sum())
    return dict(method=method, subset=name, n=len(values), sampling_seeds=len(next(iter(values.values()))['a']),
                gap=totals.sum()/counts.sum(), ci95=np.quantile(samples, [.025, .975]).tolist())


def model_comparison(left_files, right_files, method, name, sigma):
    left_data = {read(f)['args']['seed']: read(f) for f in left_files}
    right_data = {read(f)['args']['seed']: read(f) for f in right_files}
    common = sorted(left_data.keys() & right_data.keys()); groups = {}; differences = {}
    for seed in common:
        left = {r['id']: r for r in subset(left_data[seed]['records'], name)
                if r['method'] == method and r['sigma'] == sigma}
        right = {r['id']: r for r in subset(right_data[seed]['records'], name)
                 if r['method'] == method and r['sigma'] == sigma}
        if not left or not right:
            continue
        assert left.keys() == right.keys()
        for key in left:
            differences.setdefault(key, []).append(float(any(left[key]['species_hits'])) - float(any(right[key]['species_hits'])))
            groups[key] = left[key]['skeleton']
    if not differences:
        return None
    grouped = {}
    for key, values in differences.items():
        grouped.setdefault(groups[key], []).append(np.mean(values))
    totals = np.array([sum(v) for v in grouped.values()]); counts = np.array([len(v) for v in grouped.values()])
    rng = np.random.default_rng(20261003); samples = []
    for _ in range(4000):
        ix = rng.integers(len(grouped), size=len(grouped))
        samples.append(totals[ix].sum()/counts[ix].sum())
    return dict(method=method, subset=name, sigma=sigma, n=len(differences),
                sampling_seeds=len(next(iter(differences.values()))), gap=totals.sum()/counts.sum(),
                ci95=np.quantile(samples, [.025, .975]).tolist())


def main():
    files = [HERE/'scan.json'] + sorted(HERE.glob('replicate_*.json'))
    files = [f for f in files if f.exists()]
    lines = ['# Joint improvements: results', '',
             'Developmental validation. Same frozen geometry, 8 raw proposals, no replenishment.', '',
             '## First sampling-seed screening', '',
             '| Method | Noise | GEOM coverage | Charge coverage | nH coverage | Decoder ms/input |',
             '|---|---:|---:|---:|---:|---:|']
    scan = read(files[0])
    for sigma in scan['args']['noises']:
        for method in scan['args']['configs']:
            rr = {r['subset']: r for r in scan['summary'] if r['sigma'] == sigma and r['method'] == method}
            if len(rr) < 4:
                continue
            lines.append(f"| {method} | {sigma:.2f} | {rr['geom_core']['species_coverage']:.3f} | {rr['charge_challenge']['species_coverage']:.3f} | {rr['nH']['species_coverage']:.3f} | {rr['all']['decoder_ms']:.2f} |")
    lines += ['', '## Three sampling-seed means for prior weights', '',
              '| Weight | Noise | GEOM | Charge | nH | All valid fraction |',
              '|---|---:|---:|---:|---:|---:|']
    evaluations = [read(f) for f in files]
    prior_means = []
    for sigma in scan['args']['noises']:
        for method in ['w025', 'w05', 'w1']:
            means = {}
            for name in ['geom_core', 'charge_challenge', 'nH', 'all']:
                ss = [r for data in evaluations for r in data['summary']
                      if r['sigma'] == sigma and r['method'] == method and r['subset'] == name]
                means[name] = {'coverage': float(np.mean([r['species_coverage'] for r in ss])),
                               'validity': float(np.mean([r['validity'] for r in ss])), 'seeds': len(ss)}
            prior_means.append(dict(method=method, sigma=sigma, means=means))
            lines.append(f"| {method} | {sigma:.2f} | {means['geom_core']['coverage']:.3f} | {means['charge_challenge']['coverage']:.3f} | {means['nH']['coverage']:.3f} | {means['all']['validity']:.3f} |")
    comparisons = []
    lines += ['', '## Paired exploratory comparisons at noise=0.2', '',
              'Average sampling seeds within molecule, then bootstrap blinded skeleton clusters.',
              'No multiple-comparison adjustment; configurations inspected on this development set.', '',
              '| Method vs w1 | Subset | Seeds | Difference pp | 95% interval pp |',
              '|---|---|---:|---:|---:|']
    for method in scan['args']['configs']:
        if method == 'w1':
            continue
        for name in ['geom_core', 'charge_challenge', 'nH']:
            result = paired(files, method, name)
            if result:
                comparisons.append(result)
                lo, hi = result['ci95']
                lines.append(f"| {method} | {name} | {result['sampling_seeds']} | {result['gap']*100:+.1f} | [{lo*100:+.1f}, {hi*100:+.1f}] |")
    oracle = HERE/'oracle.json'
    if oracle.exists():
        data = read(oracle)
        lines += ['', '## Oracle diagnostics — NOT deployment results', '',
                  '| Condition | Noise | GEOM | Charge | Missing candidate molecules (all) |',
                  '|---|---:|---:|---:|---:|']
        for sigma in data['args']['noises']:
            for method in data['args']['configs']:
                rr = {r['subset']: r for r in data['summary'] if r['sigma'] == sigma and r['method'] == method}
                if len(rr) < 4:
                    continue
                lines.append(f"| {method} | {sigma:.2f} | {rr['geom_core']['species_coverage']:.3f} | {rr['charge_challenge']['species_coverage']:.3f} | {rr['all']['missing_candidate_molecules']} |")
    lines += ['', '## Matched continuation training', '']
    training = {}
    for arm in ['control_train', 'block_train']:
        file = HERE/arm/'training.json'
        if file.exists():
            data = read(file)
            selected = min(data['history'], key=lambda r: r['selection'])
            training[arm] = selected
            lines.append(f"- {arm}: selected epoch {selected['epoch']}; bond CE={selected['bond_CE']:.4f}, conditional h/q CE={selected['atom_conditional_CE']:.4f}.")
    for name in ['control_evaluation.json', 'block_evaluation.json']:
        file = HERE/name
        if not file.exists():
            continue
        data = read(file)
        lines += ['', f'### {name}', '', '| Method | Noise | GEOM | Charge | nH |', '|---|---:|---:|---:|---:|']
        for sigma in data['args']['noises']:
            for method in data['args']['configs']:
                rr = {r['subset']: r for r in data['summary'] if r['sigma'] == sigma and r['method'] == method}
                if len(rr) < 4:
                    continue
                lines.append(f"| {method} | {sigma:.2f} | {rr['geom_core']['species_coverage']:.3f} | {rr['charge_challenge']['species_coverage']:.3f} | {rr['nH']['species_coverage']:.3f} |")
    training_pairs = []
    control = [HERE/'control_evaluation.json'] + sorted(HERE.glob('trained_control_seed_*.json'))
    block = [HERE/'block_evaluation.json'] + sorted(HERE.glob('trained_block_seed_*.json'))
    control = [f for f in control if f.exists()]; block = [f for f in block if f.exists()]
    if control and block:
        lines += ['', '## Matched model comparisons', '',
                  'One training seed, paired sampling seeds. Developmental comparisons, not confirmatory tests.', '',
                  '| Contrast | Weight | Noise | Subset | Sampling seeds | Gap pp | 95% interval pp |',
                  '|---|---|---:|---|---:|---:|---:|']
        for contrast, left, right in [('block - control', block, control),
                                      ('block - parent', block, files), ('control - parent', control, files)]:
            for sigma in [0., .1, .2]:
                for method in ['w05', 'w1']:
                    for name in ['geom_core', 'charge_challenge', 'nH']:
                        result = model_comparison(left, right, method, name, sigma)
                        if not result:
                            continue
                        result['contrast'] = contrast; training_pairs.append(result)
                        lo, hi = result['ci95']
                        lines.append(f"| {contrast} | {method} | {sigma:.2f} | {name} | {result['sampling_seeds']} | {result['gap']*100:+.1f} | [{lo*100:+.1f}, {hi*100:+.1f}] |")
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (HERE/'paired.json').write_text(json.dumps(dict(comparisons=comparisons, prior_means=prior_means,
                                                   training=training, training_pairs=training_pairs), indent=2), encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
