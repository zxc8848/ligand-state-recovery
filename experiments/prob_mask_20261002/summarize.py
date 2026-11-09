"""Summarize paired developmental results, without selecting on a test set."""
import json
import argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='.')
    args = parser.parse_args()
    outdir = HERE/args.run
    ev = json.loads((outdir/'evaluation.json').read_text(encoding='utf-8'))
    prep = json.loads((HERE/'prepare.json').read_text(encoding='utf-8'))
    train = json.loads((outdir/'training.json').read_text(encoding='utf-8'))
    lines = ['# Probabilistic mask pilot: measured results (2026-10-02)', '',
             'Developmental validation only. No test-set or manuscript changes.', '',
             f"- Train/validation identities: {prep['counts']['train']}/{prep['counts']['val']}.",
             f"- Decoder parameters: {train['parameters']:,}; selected epoch: {ev['checkpoint_epoch']}.",
             f"- Training time: {train['history'][-1]['elapsed_seconds']:.1f} s.",
             '- Neutral connected heavy-only GEOM; predicts B, not a joint B/q/h posterior.',
             '- Strict coverage penalizes missing and extra bonds. Eight raw candidates; no replenishment.',
             '- Frozen training used noise up to 0.15 Å; this pilot also trains at 0.2 Å. Gains against the frozen model do not isolate architectural novelty.',
             '- Timings below exclude cached geometry encoding, RDKit and transfer/setup costs.',
             '- Species/validity columns use default implicit H, not a learned H/q posterior. Do not use them as final species-recovery claims.', '',
             '| Noise (Å) | Method | Budget | Strict coverage | Species (default H) | Connected valid (default H) | Valid unique/input | Decoder ms/input set |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in ev['summary']:
        name = r['method'] + (f"-{r['rounds']}r" if r['method'] in ['mask', 'remask'] else '')
        lines.append(f"| {r['sigma']:.2f} | {name} | {r['budget']} | {r['strict_coverage']:.3f} | {r['species_coverage']:.3f} | {r['connected_valid_fraction']:.3f} | {r['mean_valid_unique']:.2f} | {r['decoder_ms_per_molecule_set']:.2f} |")
    lines += ['', '## Paired differences versus decoder-independent sampling', '',
              'Percent-point differences with molecule-level paired bootstrap 95% intervals (2,000 resamples).',
              'These are unadjusted exploratory intervals, not confirmatory significance tests.', '',
              '| Noise | Mask rounds | Strict coverage Δ (95% interval) | Species coverage Δ (95% interval) |',
              '|---|---:|---:|---:|']
    rng = np.random.default_rng(20261002)
    for sigma in sorted({r['sigma'] for r in ev['records']}):
        base = {r['id']: r for r in ev['records'] if r['sigma'] == sigma and r['method'] == 'decoder_independent'}
        variants = sorted({(r['method'], r['rounds']) for r in ev['records'] if r['method'] in ['mask', 'remask']})
        for method, rounds in variants:
            records = [r for r in ev['records'] if r['sigma'] == sigma and r['method'] == method and r['rounds'] == rounds]
            cols = []
            for key in ['strict_hit', 'species_hit']:
                diff = np.array([int(r[key])-int(base[r['id']][key]) for r in records])
                boot = diff[rng.integers(0, len(diff), (2000, len(diff)))].mean(1)*100
                lo, hi = np.quantile(boot, [.025, .975])
                cols.append(f'{diff.mean()*100:+.1f} ({lo:+.1f}, {hi:+.1f})')
            lines.append(f"| {sigma:.2f} | {method}-{rounds} | {cols[0]} | {cols[1]} |")
    lines += ['', '## Backbone audit and scope', '',
              f"Rotation audit (16 clean validation geometries): {prep['rotation_audit']}.",
              'One rotation and a small clean subset do not establish exact equivariance or noisy-coordinate invariance.',
              'The frozen backbone is not changed in this experiment.', '',
              'Feature encoding times (one shared cache pass): '+str(prep['feature_seconds'])+'.',
              'A single labeled reference cannot establish the completeness of the plausible species distribution.',
              f"True-heavy-graph default-H reconstruction failures: {len(ev.get('truth_graph_default_H_failures', []))}/{prep['counts']['val']} (see per-molecule JSON diagnostics).",
              'No claimed thermodynamic stability, calibrated posterior, or speed comparison against an implemented diffusion baseline.', '']
    (outdir/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote', outdir/'RESULTS.md')


if __name__ == '__main__':
    main()
