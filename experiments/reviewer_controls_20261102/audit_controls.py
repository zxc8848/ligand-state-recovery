"""Additional independent arithmetic/replay audit, without changing plans/results."""
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/charged_benchmark_20261102'))
from study import sha, save, SEEDS


def read(path):
    with gzip.open(path,'rt',encoding='utf-8') as f: return json.load(f)


def main():
    original = ROOT/'experiments/atom_state_confirm_20261102'
    sources = {r['id']:r for r in torch.load(ROOT/'experiments/atom_state_candidates_20261102/held_back.pt',
                                            weights_only=False,map_location='cpu')}
    summary = json.loads((HERE/'MATCHED_SUMMARY.json').read_text())
    data = {}; exact_replay = 0; raw_differences = []
    for seed in SEEDS:
        old = {r['name']:r for r in read(original/f'evaluations/seed{seed}/predictions.json.gz')['records']}
        new = {r['name']:r for r in read(HERE/f'matched/seed{seed}/predictions.json.gz')['records']}
        assert set(old) == set(new) == set(sources)
        data[seed] = new
        for name,r in new.items():
            previous = old[name]; full = r['arms']['neural_catalogue']
            assert r['labels'] == previous['labels']
            assert full['single']['atom_states'] == previous['baseline']['atom_states']
            assert full['single']['native'] == previous['baseline']['native']
            entry = previous['methods']['plain_kbest']
            before = [entry['raw'][i]['smiles'] for i in entry['returned_indices']]
            after = [full['raw'][i]['smiles'] for i in full['returned_indices']]
            assert before == after, (seed,name,'returned species')
            for k in (1,2,4,8):
                a = any(entry['raw'][i]['native'] for i in entry['returned_indices'][:k])
                b = any(full['raw'][i]['native'] for i in full['returned_indices'][:k])
                assert a == b
            for i,(a,b) in enumerate(zip(full['raw'],entry['raw'])):
                if a['atom_states'] != b['atom_states']:
                    raw_differences.append({'seed':seed,'name':name,'rank':i,
                        'new_score':a['score'],'original_score':b['score'],
                        'changed_atoms':[[j,x,y] for j,(x,y) in enumerate(zip(a['atom_states'],b['atom_states'])) if x != y],
                        'same_canonical_identity':a['smiles'] == b['smiles'],
                        'constructible':a['constructible']})
            exact_replay += 1
    # Explicit member-list cluster bootstrap (different implementation from runner).
    independent = {}
    for scope in ('pooled','cation','anion','separated_neutral','uncharged_neutral'):
        names = sorted(n for n,r in sources.items() if scope=='pooled' or r['bucket']==scope)
        clusters = sorted({sources[n]['skeleton'] for n in names})
        members = [np.array([i for i,n in enumerate(names) if sources[n]['skeleton']==key]) for key in clusters]
        deltas = {}
        for metric in ('single_native','coverage1','coverage2','coverage4','coverage8','q_exact','h_exact'):
            matrices = {}
            for arm in ('neural_catalogue','catalogue_only'):
                matrix = []
                for seed in SEEDS:
                    values = []
                    for name in names:
                        record=data[seed][name]; e=record['arms'][arm]
                        if metric=='single_native': v=e['single']['native']
                        elif metric.startswith('coverage'):
                            v=any(e['raw'][i]['native'] for i in e['returned_indices'][:int(metric[8:])])
                        elif metric=='q_exact': v=[s//5-2 for s in e['single']['atom_states']]==record['target_q']
                        else: v=[s%5 for s in e['single']['atom_states']]==record['target_h']
                        values.append(int(v))
                    matrix.append(values)
                matrix=np.array(matrix); matrices[arm]=matrix
                assert np.allclose(matrix.mean(1),summary['cells'][scope]['arms'][arm][metric]['per_head'])
            deltas[metric]=(matrices['neural_catalogue']-matrices['catalogue_only']).mean(0)
        rng=np.random.default_rng(20261102); idx=rng.integers(0,len(members),(5000,len(members)))
        for metric in ('single_native','coverage8'):
            draws=[deltas[metric][np.concatenate([members[i] for i in sample])].mean() for sample in idx]
            interval=np.quantile(draws,[.025,.975]).tolist()
            assert np.allclose(interval,summary['cells'][scope]['neural_minus_catalogue'][metric]['cluster_ci95'])
            independent[scope+'/'+metric]=interval
    for plan_name in ('MATCHED_PLAN.json','D_CONTROL_PLAN.json'):
        for rel,want in json.loads((HERE/plan_name).read_text())['spec']['sha256'].items():
            assert sha(ROOT/rel)==want,rel
    d=json.loads((HERE/'D_CONTROL_SUMMARY.json').read_text())
    assert d['n_controls']==14 and d['all_assertions_passed']
    assert len(d['results'])==len({r['name'] for r in d['results']})==14
    out={'audit_sha256':sha(Path(__file__)),'matched_plan_sha256':sha(HERE/'MATCHED_PLAN.json'),
         'D_plan_sha256':sha(HERE/'D_CONTROL_PLAN.json'),
         'matched_summary_sha256':sha(HERE/'MATCHED_SUMMARY.json'),
         'D_summary_sha256':sha(HERE/'D_CONTROL_SUMMARY.json'),
         'full_arm_single_and_returned_species_exactly_replayed':exact_replay,
         'full_arm_endpoint_mismatches':0,'raw_assignment_differences':raw_differences,
         'independent_cluster_CIs':independent,'all_declared_input_hashes_unchanged':True,
         'original_artifacts_modified':False}
    save(HERE/'AUDIT.json',out)
    print(json.dumps(out,indent=2))
    print('Atom metrics (mean over three heads):')
    for scope,c in summary['cells'].items():
        print(scope)
        for arm,e in c['arms'].items():
            print(arm, {k:round(float(np.mean([a[k] for a in e['atom_single_per_head']])),6)
                        for k in ('q_macro_f1_truth_present_classes','q_atom_accuracy','h_atom_accuracy')},
                  'q_exact',e['q_exact']['mean'],'h_exact',e['h_exact']['mean'])


if __name__=='__main__':main()
