"""Rebuild and rescore the saved missing-Q study without loading any checkpoint."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch

from summarize import DATASETS, MODES, load, path_at
from pilot import DATA, J, RDLogger, metrics, reference_population, score_dump, sha

SEEDS = (20261002, 20261003, 20261004)


def groups(rows):
    return {'all':rows, 'net_neutral':[r for r in rows if r['net_ref']==0],
            'net_charged':[r for r in rows if r['net_ref']!=0],
            'cation':[r for r in rows if r['net_ref']>0], 'anion':[r for r in rows if r['net_ref']<0],
            'internal_charge':[r for r in rows if r['net_ref']==0 and any(r['q_ref'])]}


def audit(root, ds, seed, mode, target, sdf):
    path = path_at(root, seed, ds, mode)
    edges = path.with_suffix('.jsonl')
    data = load(path)
    rows = data['records']
    assert list(target) == [r['name'] for r in rows]
    graph = defaultdict(list)
    with edges.open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r['pred']:
                graph[r['name']].append((r['i'],r['j'],r['pred']))
    verdict = {r['name']:r for r in score_dump(edges, sdf, expected_n=len(target))['per_molecule']}
    for r in rows:
        name = r['name']
        ref = target[name]
        pairs = torch.tensor([(i,j) for i,j,_ in graph[name]], dtype=torch.long).reshape(-1,2)
        labels = torch.tensor([k for _,_,k in graph[name]], dtype=torch.long)
        states = torch.tensor(r['states'], dtype=torch.long)
        assert len(states) == len(ref['z']) and bool(((states>=0)&(states<25)).all())
        smi, good = J.build_species({'z':ref['z'],'pairs':pairs},labels,states)
        assert (smi, good, bool(good and smi==ref['smiles'])) == (r['smiles'],r['buildable'],r['native']), (name,mode)
        q = (states//5-2).tolist()
        assert r['q_pred']==q and r['net_pred']==sum(q)
        assert r['q_ref']==ref['q'].tolist() and r['net_ref']==ref['Q']
        assert r['q_exact']==(q==r['q_ref']) and r['net_correct']==(sum(q)==ref['Q'])
        g = verdict[name]
        assert (g['strict'],g['unified_resonance'],g['smiles_species']) == (r['strict'],r['resonance'],r['graph_smiles'])
    assert {k:metrics(v) for k,v in groups(rows).items()}==data['summary']
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=['ccd_dev'])
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    RDLogger.DisableLog('rdApp.*')
    torch.set_num_threads(4)
    receipt = {'scope':'saved outputs only; no checkpoint or new predictions','datasets':{}}
    for ds in a.datasets:
        sdf = DATA[ds]
        refs, excluded = reference_population(sdf)
        assert not excluded and len(refs)==DATASETS[ds]
        target = {name:J.make_row(m,name,-1,ds) for name,m in refs.items()}
        n = 0
        for seed in SEEDS:
            for mode in MODES:
                n += audit(a.root,ds,seed,mode,target,sdf)
            print(f'{ds} {seed}: six cells verified',flush=True)
        receipt['datasets'][ds] = {'n':len(refs),'prediction_rows_verified':n,'reference_sha256':sha(sdf)}
    with a.output.open('x',encoding='utf-8') as f:
        json.dump(receipt,f,indent=2)
        f.write('\n')
    print(receipt)


if __name__ == '__main__':
    main()
