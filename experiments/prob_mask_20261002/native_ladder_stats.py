"""Native molecular-identity contrasts A->B and B->C, not charge-blind graphs."""
import argparse
import collections
import json
from pathlib import Path
from cluster_bootstrap import boot_clusters, boot_molecules, identity_key, mcnemar
from evaluation_common import reference_population


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--beam', required=True)
    ap.add_argument('--reference_sdf', required=True)
    ap.add_argument('--expected_n', type=int)
    ap.add_argument('--population_manifest')
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    import torch
    import joint as J
    refs, _ = reference_population(a.reference_sdf)
    if a.population_manifest:
        from population_manifest import validate_population
        refs, _ = validate_population(a.reference_sdf, a.population_manifest)
    rows = json.loads(Path(a.beam).read_text(encoding='utf-8'))['records']
    if len(rows) != len(refs) or {r['name'] for r in rows} != set(refs):
        raise ValueError('Native beam must contain exactly the reference population')
    if a.expected_n is not None and len(rows) != a.expected_n:
        raise ValueError('Native denominator differs from frozen size')
    flags = collections.defaultdict(dict)
    for r in rows:
        ref = J.make_row(refs[r['name']], r['name'], -1, 'evaluation')
        if r['z'] != ref['z'].tolist() or r['Q'] != ref['Q']:
            raise ValueError('Native record Z/Q mismatch')
        for selector in ('greedy', 'bondonly', 'localmap'):
            cand = r['candidates_detail'][r['selected_indices'][selector]]
            states = cand['atom_states']
            if len(states) != len(r['z']) or sum(s//5-2 for s in states) != r['Q']:
                raise ValueError('Native states violate inventory/total charge')
            build_row = dict(ref, pairs=torch.tensor(r['pairs'], dtype=torch.long))
            smi, good = J.build_species(build_row, torch.tensor(cand['bond_labels']), torch.tensor(states))
            ok = bool(good and smi == ref['smiles'])
            if cand['smiles'] != smi or cand['good'] != good or cand['correct'] != ok:
                raise ValueError('Native serialized prediction fails independent rebuild')
            flags[selector][r['name']] = ok
    names = sorted(refs)
    out = {'endpoint': 'native_predicted_B_h_q_identity', 'n_molecules': len(names), 'endpoints': {}}
    for tag, old, new in [('A_to_B_search_and_filter', 'greedy', 'bondonly'),
                          ('B_to_C_atom_score_ranking', 'bondonly', 'localmap')]:
        pairs = [(int(flags[old][n]), int(flags[new][n])) for n in names]
        clusters = collections.defaultdict(list)
        for n, pair in zip(names, pairs): clusters[identity_key(refs[n])].append(pair)
        ro, rn = [sum(p[i] for p in pairs)/len(pairs) for i in (0, 1)]
        out['endpoints'][tag] = {'old': ro, 'new': rn, 'delta': rn-ro,
            'mcnemar_p': mcnemar(pairs), 'molecule_ci95': boot_molecules(pairs, 2000, 11),
            'identity_cluster_ci95': boot_clusters(list(clusters.values()), 2000, 11)}
    Path(a.output).write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(out, indent=1))


if __name__ == '__main__': main()
