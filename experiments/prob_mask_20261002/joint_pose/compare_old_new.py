"""HISTORICAL legacy-classify comparison, not used by the v3 locked driver.

For unified v3 inference use ../paired_verdicts.py and versioned unified JSONs.
This file is retained to reproduce historical tables, not as the current primary
resonance scorer.

Paired comparison of the archived two-stage+MILP decoder and the single-stage
JointNet+Q-DP decoder on the 488 PDBBind poses.

Both systems are read from per-edge dumps with the same reference labels and the
same molecule set, and the species/resonance endpoints come from the archived
method-neutral scorer, so the only thing that differs is the prediction column.

Run from the repository root:

    python experiments/prob_mask_20261002/joint_pose/compare_old_new.py
"""
import argparse
import collections
import json
import random
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def parse_args():
    a = argparse.ArgumentParser()
    a.add_argument('--old_per_edge', default=str(ROOT/'results/per_edge_training_catalogue.jsonl'))
    a.add_argument('--new_per_edge', default=str(HERE/'joint_pose_sigma0_refedges.jsonl'))
    a.add_argument('--old_species', default=str(HERE/'baseline_species.json'))
    a.add_argument('--new_species', default=str(HERE/'new_species_refedges.json'))
    a.add_argument('--overlap', default=str(HERE/'geom_overlap_names.json'))
    a.add_argument('--output', default=None)
    a.add_argument('--reps', type=int, default=2000,
                   help='bootstrap replicates for the paired interval')
    return a.parse_args()


def load_dump(path):
    per = collections.defaultdict(list)
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            per[row['name']].append(row)
    return per


def strict_flags(per, names):
    return {m: all(r['pred'] == r['true'] for r in per[m]) for m in names}


def typed_recall(per, names):
    hit = tot = 0
    for m in names:
        for r in per[m]:
            if r['true'] > 0:
                tot += 1
                hit += r['pred'] == r['true']
    return hit/tot, tot


def bootstrap_ci(a, b, reps=2000, seed=20261004):
    """Percentile interval for the paired difference, resampling molecules.

    The unit is the molecule. NOTE (2026-10-04): the original wording here said
    each pose is a distinct ligand with no cluster structure. That is wrong --
    the 488 development poses carry 471 distinct non-isomeric canonical
    identities: 12 duplicate groups, largest 5, 29 molecules involved. External
    review caught it. cluster_bootstrap.py resamples identity clusters instead
    and the intervals come out 0.94-1.06x the width of these, with no zero-
    exclusion verdict changed, so the molecule-level interval is kept as the
    reported one and the cluster-level interval is reported alongside it. The
    claim was still false and should not have been written
    (unlike the GEOM/diverse validation split,
    where connectivity skeletons are resampled instead).
    """
    keys = sorted(set(a) & set(b))
    deltas = [float(b[k])-float(a[k]) for k in keys]
    if not deltas:
        return None, None
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(deltas[rng.randrange(n)] for _ in range(n))/n
                   for _ in range(reps))
    return means[int(0.025*reps)], means[min(reps-1, int(0.975*reps))]


def mcnemar(a, b):
    """Exact two-sided McNemar on paired boolean dicts (a = old, b = new)."""
    n_old = sum(1 for k in a if a[k] and not b[k])
    n_new = sum(1 for k in a if b[k] and not a[k])
    n = n_old + n_new
    if n == 0:
        return n_old, n_new, 1.0
    k = min(n_old, n_new)
    p = sum(comb(n, i) for i in range(k+1))/2**n*2
    return n_old, n_new, min(1.0, p)


def main():
    a = parse_args()
    old = load_dump(a.old_per_edge)
    new = load_dump(a.new_per_edge)
    sp_old = {r['name']: r for r in json.loads(Path(a.old_species).read_text())['per_molecule']}
    sp_new = {r['name']: r for r in json.loads(Path(a.new_species).read_text())['per_molecule']}
    names = sorted(sp_old)
    subsets = {'all': names}
    if Path(a.overlap).exists():
        over = set(json.loads(Path(a.overlap).read_text())['skeleton'])
        subsets['geom_disjoint'] = [n for n in names if n not in over]

    report = {}
    for label, subset in subsets.items():
        k = len(subset)
        so, sn = strict_flags(old, subset), strict_flags(new, subset)
        rows = {'n': k,
                'strict': {'old': sum(so.values())/k, 'new': sum(sn.values())/k},
                'typed_recall': {'old': typed_recall(old, subset)[0],
                                 'new': typed_recall(new, subset)[0]}}
        n_old, n_new, p = mcnemar(so, sn)
        lo, hi = bootstrap_ci(so, sn, a.reps)
        rows['strict']['discordant_old_only'] = n_old
        rows['strict']['discordant_new_only'] = n_new
        rows['strict']['mcnemar_p'] = p
        rows['strict']['delta'] = rows['strict']['new']-rows['strict']['old']
        rows['strict']['ci95'] = [lo, hi]
        for key, name in [('classify_species', 'resonance_aware'),
                          ('smiles_species', 'canonical_smiles')]:
            ao = {n: sp_old[n][key] for n in subset}
            an = {n: sp_new[n][key] for n in subset}
            n_old, n_new, p = mcnemar(ao, an)
            lo, hi = bootstrap_ci(ao, an, a.reps)
            rows[name] = {'old': sum(ao.values())/k, 'new': sum(an.values())/k,
                          'delta': sum(an.values())/k-sum(ao.values())/k,
                          'ci95': [lo, hi],
                          'discordant_old_only': n_old, 'discordant_new_only': n_new,
                          'mcnemar_p': p}
        report[label] = rows

        print(f'\n=== {label}  n={k} ===')
        for key in ['strict', 'resonance_aware', 'canonical_smiles']:
            r = rows[key]
            ci = r.get('ci95') or [None, None]
            ci_txt = ('' if ci[0] is None
                      else f'  delta {r["delta"]*100:+.2f}pp CI [{ci[0]*100:+.2f},{ci[1]*100:+.2f}]pp')
            print(f'  {key:17s} old {r["old"]:.4f}  new {r["new"]:.4f}{ci_txt}  '
                  f'(old-only {r["discordant_old_only"]}, new-only {r["discordant_new_only"]}, '
                  f'exact McNemar p={r["mcnemar_p"]:.3g})')
        print(f'  typed_recall      old {rows["typed_recall"]["old"]:.4f}  '
              f'new {rows["typed_recall"]["new"]:.4f}')

    if a.output:
        Path(a.output).write_text(json.dumps(report, indent=1), encoding='utf-8')
        print(f'\nwrote {a.output}')


if __name__ == '__main__':
    main()
