"""Coverage@N of the joint sampler's candidate sets, scored through the archived
paper-2 build path (``species_signature``), so the numbers sit directly next to the
published MILP candidate-set curves (species 0.793 -> 0.885, strict labelling
0.559 -> 0.660 on the same 488 poses).

Run from the repository root:

    python experiments/prob_mask_20261002/joint_pose/coverage_paper2_path.py
"""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEGACY = ROOT/'archive/legacy_two_stage_20261003'


def parse_args():
    a = argparse.ArgumentParser()
    a.add_argument('--summary', default=str(Path(__file__).resolve().parent/'joint_pose_sigma0_summary.json'))
    a.add_argument('--baseline_per_edge', default=str(ROOT/'results/per_edge_training_catalogue.jsonl'))
    a.add_argument('--reference_sdf', default=str(ROOT/'data/pdbbind_ccd_pose500.sdf'))
    a.add_argument('--legacy_root', default=str(LEGACY))
    a.add_argument('--k', type=int, default=8)
    a.add_argument('--output', default=None)
    return a.parse_args()


def main():
    a = parse_args()
    sys.path.insert(0, a.legacy_root)
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    from scripts.evaluate_candidate_sets import species_signature
    from scripts.evaluate_species_recovery import _graph_candidate

    base = collections.defaultdict(list)
    for line in Path(a.baseline_per_edge).read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            base[row['name']].append(row)

    data = json.loads(Path(a.summary).read_text(encoding='utf-8'))
    recs = {r['name']: r for r in data['records']}
    seeds = [str(s) for s in data['args']['seeds']]

    mols = {}
    for mol in Chem.SDMolSupplier(a.reference_sdf, removeHs=False):
        if mol is None:
            continue
        name = mol.GetProp('_Name')
        if name in recs:
            mols[name] = Chem.RemoveHs(mol, sanitize=True)

    K = a.k
    sp = {s: [0]*K for s in seeds}
    st = {s: [0]*K for s in seeds}
    n = 0
    for name, rec in recs.items():
        mol = mols[name]
        elems = [at.GetAtomicNum() for at in mol.GetAtoms()]
        truth = {(r['i'], r['j']): r['true'] for r in base[name] if r['true'] > 0}
        adj = collections.defaultdict(set)
        for i, j in truth:
            adj[i].add(j); adj[j].add(i)
        ref_key, _, _ = species_signature(
            mol, _graph_candidate(elems, truth), adj, return_status=True)
        ref_ok = not ref_key.startswith('fallback::')
        truth_sorted = {tuple(sorted(e)): c for e, c in truth.items()}
        n += 1
        pairs = [tuple(e) for e in rec['pairs']]
        for s in seeds:
            hit_sp = hit_st = None
            for k, labels in enumerate(rec[f'cand_labels_{s}']):
                pred = {tuple(sorted(e)): c for e, c in zip(pairs, labels) if c}
                if hit_st is None and pred == truth_sorted:
                    hit_st = k
                if hit_sp is None and ref_ok:
                    key, _, _ = species_signature(
                        mol, _graph_candidate(elems, pred), adj, return_status=True)
                    if not key.startswith('fallback::') and key == ref_key:
                        hit_sp = k
            for k in range(K):
                if hit_sp is not None and hit_sp <= k:
                    sp[s][k] += 1
                if hit_st is not None and hit_st <= k:
                    st[s][k] += 1

    result = {'n_molecules': n, 'seeds': seeds, 'k': K}
    print(f'molecules: {n}')
    for label, table, key in [('species (paper-2 build path)', sp, 'species'),
                              ('strict labelling', st, 'strict_labelling')]:
        mean = [sum(table[s][k] for s in seeds)/len(seeds)/n for k in range(K)]
        result[key] = mean
        result[key+'_per_seed'] = {s: [v/n for v in table[s]] for s in seeds}
        print(f'{label}: ' + '  '.join(f'@{k+1}={v:.3f}' for k, v in enumerate(mean)))
    if a.output:
        Path(a.output).write_text(json.dumps(result, indent=1), encoding='utf-8')
        print(f'wrote {a.output}')


if __name__ == '__main__':
    main()
