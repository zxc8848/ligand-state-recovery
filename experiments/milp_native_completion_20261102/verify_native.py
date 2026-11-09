"""Independent current-native rebuild, paired reporting, and new Holm12 only."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch
from rdkit import Chem,RDLogger

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OLD=ROOT/'experiments/prob_mask_20261002'
sys.path.insert(0,str(OLD))
import joint as J
from cluster_bootstrap import mcnemar


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):
    if path.suffix=='.gz':
        with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)
    return json.loads(path.read_text())


def save(path,d):path.write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf-8')


def estimate(matrix,clusters):
    x=np.asarray(matrix,float);keys=sorted(set(clusters));key={v:i for i,v in enumerate(keys)}
    counts=np.zeros(len(keys));sums=np.zeros(len(keys))
    for value,cluster in zip(x.mean(0),clusters):counts[key[cluster]]+=1;sums[key[cluster]]+=value
    rng=np.random.default_rng(20261102);idx=rng.integers(0,len(keys),(5000,len(keys)))
    boot=sums[idx].sum(1)/counts[idx].sum(1)
    return {'mean':float(x.mean()),'head_sd':float(x.mean(1).std(ddof=1)) if len(x)>1 else None,
            'per_head':x.mean(1).tolist(),'identity_cluster_ci95':np.quantile(boot,[.025,.975]).tolist()}


def main():
    torch.set_num_threads(4);RDLogger.DisableLog('rdApp.*')
    plan=read(HERE/'PLAN.json');digest=sha(HERE/'PLAN.json')
    for rel,want in plan['spec']['sha256'].items():assert sha(ROOT/rel)==want,rel
    out={'scope':'post hoc paired full-system native comparison','plan_sha256':digest,'cells':{},'tests':[]}
    verified=0;current_verified=0
    for condition,(tag,n) in plan['spec']['conditions'].items():
        path=ROOT/'data/pdbbind_pose_locked500.sdf' if tag=='locked' else ROOT/f'work/paper2_noise_ccd_probe/{tag}/reference_for_scoring.sdf'
        allowed={6,7,8,9,14,15,16,17,33,35,53,83}
        refs={}
        for mol in Chem.SDMolSupplier(str(path),removeHs=False):
            assert mol is not None;mol=Chem.RemoveHs(mol,sanitize=True)
            if {a.GetAtomicNum() for a in mol.GetAtoms()}<=allowed:refs[mol.GetProp('_Name')]=mol
        assert len(refs)==n
        folder=HERE/f'evaluations/{condition}';receipt=read(folder/'DONE.json');predfile=folder/'predictions.jsonl'
        assert receipt['sha256']==sha(predfile) and receipt['plan_sha256']==digest
        records={r['name']:r for r in [json.loads(line) for line in predfile.read_text().splitlines() if line.strip()]}
        assert len(records)==n and set(records)==set(refs)
        base={};raw_strict=0;final_strict=0;complete=0;good_count=0;q_changes=0;failures=collections.Counter()
        for name,r in records.items():
            assert r['plan_sha256']==digest and r['raw_graph_replay_equal']
            mol=refs[name];ref=J.make_row(mol,name,-1,'verification');f=r['final']
            assert r['z']==ref['z'].tolist() and r['Q']==ref['Q'] and r['target_smiles']==ref['smiles']
            raw={(i,j):c for i,j,c in r['raw_B']};raw_strict+=int(raw==ref['truth'])
            if f['complete']:
                complete+=1;q_changes+=int(f['q_changed_by_cleanup'])
                states=torch.tensor([(q+2)*5+h for q,h in zip(f['q'],f['h'])],dtype=torch.long)
                row={'z':ref['z'],'pairs':torch.tensor(f['pairs'],dtype=torch.long)}
                smi,good=J.build_species(row,torch.tensor(f['labels'],dtype=torch.long),states)
                good=bool(good and f['sum_correct'] and f['in_vocabulary'])
                assert (smi,good)==(f['smiles'],f['good'])
                graph={tuple(e):c for e,c in zip(f['pairs'],f['labels']) if c}
                final_strict+=int(graph==ref['truth']);good_count+=good
                if not good:failures['completed_but_not_native_buildable']+=1
                verified+=1
            else:failures[r['completion_error'] or 'completion_failed']+=1
            native=bool(f['good'] and f['smiles']==ref['smiles']);assert native==f['native'];base[name]=native
        matrices=[]
        for seed in plan['spec']['current_heads']:
            p=OLD/f'locked_eval/seed{seed}/pose_localmap_locked.json' if tag=='locked' else ROOT/f'experiments/catalogue_only_20261101/evaluations_v3/seed{seed}/{condition}/predictions.json.gz'
            saved={r['name']:r for r in read(p)['records']};assert set(saved)==set(refs)
            flags={}
            for name,r in saved.items():
                ref=J.make_row(refs[name],name,-1,'verification')
                if tag=='locked':
                    cand=r['candidates_detail'][r['selected_indices']['localmap']]
                    labels=cand['bond_labels'];states=cand['atom_states'];pairs=r['pairs']
                    want=cand['correct'];saved_smi=cand['smiles'];saved_good=cand['good']
                else:
                    arm=r['neural_catalogue'];cand=arm['candidates'][arm['selected_index']]
                    labels=arm['labels'];states=cand['atom_states'];pairs=r['pool']['pairs']
                    want=arm['native'];saved_smi=cand['smiles'];saved_good=cand['good']
                row={'z':ref['z'],'pairs':torch.tensor(pairs,dtype=torch.long)}
                smi,good=J.build_species(row,torch.tensor(labels,dtype=torch.long),torch.tensor(states,dtype=torch.long))
                assert (smi,good)==(saved_smi,saved_good)
                flags[name]=bool(good and smi==ref['smiles']);assert flags[name]==want
                current_verified+=1
            names=sorted(refs);matrix=[int(flags[name]) for name in names];matrices.append(matrix)
            pairs=[(int(base[name]),int(flags[name])) for name in names]
            out['tests'].append({'condition':condition,'seed':seed,'p_raw':mcnemar(pairs),
                                 'delta':float(np.mean([b-a for a,b in pairs])),
                                 'current_only':sum(a==0 and b==1 for a,b in pairs),
                                 'baseline_only':sum(a==1 and b==0 for a,b in pairs)})
        names=sorted(refs);clusters=[Chem.MolToSmiles(refs[name],isomericSmiles=False) for name in names]
        baseline=np.array([int(base[name]) for name in names]);x=np.array(matrices)
        out['cells'][condition]={'n':n,'n_identity_clusters':len(set(clusters)),
            'completed_baseline':estimate(baseline[None,:],clusters),'current_stage_C':estimate(x,clusters),
            'current_minus_baseline':estimate(x-baseline[None,:],clusters),
            'raw_strict':raw_strict/n,'final_strict':final_strict/n,'completion_count':complete,
            'native_buildable_count':good_count,'cleanup_q_change_count':q_changes,'failure_counts':dict(failures)}
    assert len(out['tests'])==12;running=0.
    for rank,test in enumerate(sorted(out['tests'],key=lambda r:r['p_raw'])):
        running=max(running,min(1.,(12-rank)*test['p_raw']));test['p_holm12']=running
    out['verification']={'completed_baseline_tuple_rebuilds':verified,'current_selected_tuple_rebuilds':current_verified,
                          'all_raw_graphs_replay_frozen':True,'all_input_hashes_unchanged':True,'no_old_test_families_changed':True}
    for rel,want in plan['spec']['sha256'].items():assert sha(ROOT/rel)==want
    save(HERE/'SUMMARY.json',out)
    lines=['# Native comparison: current C vs archived MILP + declared completion','',
           'Post hoc full-system comparison. Same project baseline, not an independent external or non-neural method.', '',
           '| Condition | n | Completed MILP native | Current C native (head SD) | Paired delta95% cluster CI, pp |',
           '|---|---:|---:|---:|---:|']
    for name,c in out['cells'].items():
        a=c['completed_baseline'];b=c['current_stage_C'];d=c['current_minus_baseline'];lo,hi=d['identity_cluster_ci95']
        lines.append(f"| {name} | {c['n']} | {100*a['mean']:.2f}% | {100*b['mean']:.2f} ± {100*b['head_sd']:.2f}% | {100*d['mean']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines+=['','Native contrasts have one new descriptive post hoc Holm family12; all per-head raw/adjusted values are in SUMMARY.json.',
            'Raw old graph parity is checked before native completion. Final normalized graphs are distinct outputs and are not substituted into frozen graph tables.',
            json.dumps(out['verification'])]
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines))


if __name__=='__main__':main()
