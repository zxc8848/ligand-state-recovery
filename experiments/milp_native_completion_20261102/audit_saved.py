"""Additional saved-output audit: raw graph parity, completion replay, molecule CIs."""
import collections
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem,RDLogger
import study as S


def main():
    RDLogger.DisableLog('rdApp.*');torch.set_num_threads(2);digest=S.check()
    result={'plan_sha256':digest,'conditions':{}};count=0
    for condition in S.CONDITIONS:
        rows=[json.loads(line) for line in (S.HERE/f'evaluations/{condition}/predictions.jsonl').read_text().splitlines() if line.strip()]
        graphs=S.old_graphs(condition);replayed=0;changes=0;latent_zero=0
        for r in rows:
            assert {(i,j):c for i,j,c in r['raw_B']}==graphs[r['name']]
            z=torch.tensor(r['z']);q=np.array(r['raw_q']);labels=np.array(r['raw_labels']);h=np.array(r['latent_h_unused'])
            candidate={'elems':z,'edges':r['raw_pairs'],'edge_labels':labels,'formal_charge':q,'implicit_h':h}
            mol,error=S.C._candidate_mol(candidate,return_error=True) if r['MILP_solved'] else (None,'MILP_unsolved')
            assert (mol is not None)==r['final']['complete']
            if mol is not None:
                f=r['final']
                assert [a.GetTotalNumHs(includeNeighbors=True) for a in mol.GetAtoms()]==f['h']
                assert [a.GetFormalCharge() for a in mol.GetAtoms()]==f['q']
                assert S.B.true_graph(mol)=={tuple(e):c for e,c in zip(f['pairs'],f['labels'])}
                assert Chem.MolToSmiles(mol,isomericSmiles=False)==f['smiles']
                changes+=int({tuple(e):c for e,c in zip(f['pairs'],f['labels'])}!=graphs[r['name']])
            latent_zero+=int(all(v==0 for v in r['latent_h_unused']));replayed+=1;count+=1
        result['conditions'][condition]={'raw_graphs_and_completion_paths_replayed':replayed,
            'final_B_different_from_raw':changes,'latent_H_all_zero_count':latent_zero,
            'MILP_unsolved':sum(not r['MILP_solved'] for r in rows),
            'completion_failed':sum(not r['final']['complete'] for r in rows),
            'completed_nonclosed_or_disconnected':sum(r['final']['complete'] and not r['final']['good'] for r in rows)}
    summary=S.read(S.HERE/'SUMMARY.json')
    for condition in S.CONDITIONS:
        c=summary['cells'][condition]
        # Generate molecule-bootstrap intervals from recorded flags, separate
        # from identity-cluster intervals and without altering the frozen plan.
        rows=[json.loads(line) for line in (S.HERE/f'evaluations/{condition}/predictions.jsonl').read_text().splitlines() if line.strip()]
        baseline={r['name']:int(r['final']['native']) for r in rows};names=sorted(baseline);matrix=[]
        for seed in S.SEEDS:
            saved={r['name']:r for r in S.read(S.current(condition,seed))['records']}
            if condition=='locked_clean':flags={n:int(r['candidates_detail'][r['selected_indices']['localmap']]['correct']) for n,r in saved.items()}
            else:flags={n:int(r['neural_catalogue']['native']) for n,r in saved.items()}
            matrix.append([flags[n]-baseline[n] for n in names])
        x=np.mean(matrix,axis=0);rng=np.random.default_rng(20261102);idx=rng.integers(0,len(names),(5000,len(names)))
        result['conditions'][condition]['paired_molecule_ci95']=np.quantile(x[idx].mean(1),[.025,.975]).tolist()
        assert abs(x.mean()-c['current_minus_baseline']['mean'])<1e-12
    result.update(total_paths_checked=count,all_input_hashes_unchanged=True,original_outputs_modified=False)
    S.save(S.HERE/'SAVED_AUDIT.json',result);S.check();print(json.dumps(result,indent=2))


if __name__=='__main__':main()
