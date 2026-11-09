"""All paired statistics consume the same versioned per-molecule flags."""
import argparse
import collections
import json
from pathlib import Path
from evaluation_common import DEFINITION, load_verdicts, reference_population
from cluster_bootstrap import boot_clusters, boot_molecules, identity_key, mcnemar


def compare(old_path, new_path, reference_sdf, reps=2000, seed=11, expected_n=None, population_manifest=None):
    _, old = load_verdicts(old_path)
    _, new = load_verdicts(new_path)
    refs, _ = reference_population(reference_sdf)
    if population_manifest:
        from population_manifest import validate_population
        refs, _ = validate_population(reference_sdf, population_manifest)
    if set(old) != set(new) or set(old) != set(refs):
        raise ValueError('Verdicts and reference must cover the identical fixed population')
    names = sorted(refs)
    if expected_n is not None and len(names) != expected_n:
        raise ValueError(f'Expected {expected_n} verdicts, got {len(names)}')
    for name in names:
        if not old[name]['missing_prediction'] and not new[name]['missing_prediction']:
            if old[name]['truth_hash'] != new[name]['truth_hash']:
                raise ValueError(f'{name}: methods used different strict reference labels')
    keys = {n: identity_key(refs[n]) for n in names}
    groups = collections.Counter(keys.values())
    out = {'definition': DEFINITION, 'old_verdicts': str(old_path),
           'new_verdicts': str(new_path), 'n_molecules': len(names),
           'n_identities': len(groups), 'duplicate_groups': sum(v > 1 for v in groups.values()),
           'reps': reps, 'seed': seed, 'endpoints': {}}
    for endpoint, field in [('strict', 'strict'), ('resonance', 'unified_resonance'),
                            ('SMILES', 'smiles_species')]:
        pairs = [(int(old[n][field]), int(new[n][field])) for n in names]
        byid = collections.defaultdict(list)
        for n, pair in zip(names, pairs): byid[keys[n]].append(pair)
        mci = boot_molecules(pairs, reps, seed)
        cci = boot_clusters(list(byid.values()), reps, seed)
        ro, rn = [sum(p[i] for p in pairs)/len(pairs) for i in (0, 1)]
        out['endpoints'][endpoint] = {
            'old': ro, 'new': rn, 'delta': rn-ro,
            'discordant_old_only': sum(x and not y for x, y in pairs),
            'discordant_new_only': sum(y and not x for x, y in pairs),
            'mcnemar_p': mcnemar(pairs), 'molecule_ci95': list(mci),
            'identity_cluster_ci95': list(cci),
            'cluster_ci_excludes_zero': cci[0] > 0 or cci[1] < 0,
            'width_ratio': (cci[1]-cci[0])/(mci[1]-mci[0]) if mci[1] > mci[0] else None}
    for tag, values in [('old', old), ('new', new)]:
        den = sum(values[n]['true_edge_count'] for n in names)
        out[f'{tag}_typed_recall'] = sum(values[n]['true_edge_hits'] for n in names)/den if den else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old_verdicts', required=True)
    ap.add_argument('--new_verdicts', required=True)
    ap.add_argument('--reference_sdf', required=True)
    ap.add_argument('--expected_n', type=int)
    ap.add_argument('--population_manifest')
    ap.add_argument('--reps', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    out = compare(a.old_verdicts, a.new_verdicts, a.reference_sdf, a.reps, a.seed, a.expected_n, a.population_manifest)
    Path(a.output).write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f'paired n={out["n_molecules"]}, identities={out["n_identities"]}')
    for ep, r in out['endpoints'].items():
        print(f'{ep}: {r["delta"]:+.4f}, cluster CI={r["identity_cluster_ci95"]}, p={r["mcnemar_p"]:.4g}')


if __name__ == '__main__':
    main()
