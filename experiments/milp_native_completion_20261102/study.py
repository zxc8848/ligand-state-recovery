"""Fixed archived single-decode system with an explicit complete-output policy."""
import argparse
import collections
from datetime import datetime,timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
from rdkit import Chem,RDLogger,rdBase

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
LEGACY=ROOT/'archive/legacy_two_stage_20261003'
sys.path.insert(0,str(LEGACY))
from scripts import benchmark_templatefree_motifs as B
from scripts import evaluate_candidate_sets as C
from bondnet.inference import chemical_repair as M

OLD=ROOT/'experiments/prob_mask_20261002'
SEEDS=(20261002,20261003,20261004)
ALLOWED={6,7,8,9,14,15,16,17,33,35,53,83}
CONDITIONS={'locked_clean':('locked',491),'pose_clean':('pdbbind_sigma0p00',488),
            'pose_noise010':('pdbbind_sigma0p10',488),'ccd_ideal':('ccd_sigma0p00',1000)}
STAGE1=ROOT/'checkpoints/diverse80k_heavy_stage1/best.pt'
REFINER=ROOT/'checkpoints/diverse80k_heavy_struct6_vector_hardneg_to50/best.pt'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def sdf(condition):
    tag,n=CONDITIONS[condition]
    return ROOT/'data/pdbbind_pose_locked500.sdf' if tag=='locked' else ROOT/f'work/paper2_noise_ccd_probe/{tag}/reference_for_scoring.sdf'


def edge_dump(condition):
    tag,_=CONDITIONS[condition]
    return OLD/'locked_eval/old_per_edge.jsonl' if tag=='locked' else ROOT/f'work/paper2_noise_ccd_probe/{tag}/bondnet_per_edge.jsonl'


def current(condition,seed):
    if condition=='locked_clean':return OLD/f'locked_eval/seed{seed}/pose_localmap_locked.json'
    return ROOT/f'experiments/catalogue_only_20261101/evaluations_v3/seed{seed}/{condition}/predictions.json.gz'


def read(path):
    if path.suffix=='.gz':
        with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)
    return json.loads(path.read_text(encoding='utf-8'))


def plan():
    paths=[Path(__file__),HERE/'README.md',HERE/'verify_native.py',STAGE1,REFINER,
           ROOT/'experiments/milp_h_path_audit_20261102/AUDIT.json']
    # Pin all archived source files actually available in the original inventory.
    manifest=read(ROOT/'results/FROZEN_MANIFEST_V3_1_2026-10-05.json')
    paths += [ROOT/p for p in manifest['files'] if p.replace('\\','/').startswith('archive/legacy_two_stage_20261003/')]
    for condition in CONDITIONS:
        paths += [sdf(condition),edge_dump(condition),*[current(condition,s) for s in SEEDS]]
    paths += [ROOT/'experiments/prob_mask_20261002/joint.py']
    spec={'status':'post hoc system comparison on already evaluated cohorts','conditions':CONDITIONS,
          'baseline':'Archived BondNet-MILP + archived predicted-graph normalization + RDKit-H completion',
          'n_heads_current':3,'current_heads':SEEDS,'baseline_models':1,
          'raw_decode':'unchanged benchmark.bondnet_predict, one joint_decode, no reference atom state',
          'raw_graph_parity':'must reproduce every frozen old per-edge graph; mismatches block completion',
          'completion':'actual evaluate_candidate_sets._candidate_mol, then freeze final B/q/H',
          'final_filters':['sanitization','one connected component','zero radical electrons','sum(q)=Q','25-state q/h vocabulary'],
          'selection':'one raw graph; no retry, no alternative candidate search, no reference selection',
          'hypothesis_family':{'post_hoc':True,'size':12,'endpoint':'native','contrasts':'current C minus completed baseline; 4 conditions x3 heads'},
          'bootstrap':{'reps':5000,'seed':20261102,'unit':'canonical non-isomeric reference identity','weights':'molecule-weighted ratio',
                       'heads':'average before resampling; head SD separately','scope':'pointwise; fixed trained heads'},
          'old_families_modified':False,'sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}}
    spec=json.loads(json.dumps(spec));p=HERE/'PLAN.json'
    if p.exists():assert read(p)['spec']==spec,'Plan drift'
    else:save(p,{'recorded_utc':datetime.now(timezone.utc).isoformat(),'spec':spec})
    return sha(p)


def check():
    for rel,want in read(HERE/'PLAN.json')['spec']['sha256'].items():assert sha(ROOT/rel)==want,rel
    return sha(HERE/'PLAN.json')


def references(condition):
    rows=[];names=set()
    for idx,mol in enumerate(Chem.SDMolSupplier(str(sdf(condition)),removeHs=False)):
        assert mol is not None and mol.GetNumConformers()>0,(condition,idx)
        mol=Chem.RemoveHs(mol,sanitize=True)
        if not {a.GetAtomicNum() for a in mol.GetAtoms()}<=ALLOWED:continue
        name=mol.GetProp('_Name');assert name not in names;names.add(name)
        rows.append((idx,name,mol))
    assert len(rows)==CONDITIONS[condition][1]
    return rows


def old_graphs(condition):
    graphs=collections.defaultdict(dict)
    for line in edge_dump(condition).read_text().splitlines():
        if not line.strip():continue
        r=json.loads(line);graphs[r['name']]
        if r['pred']:graphs[r['name']][tuple(sorted((r['i'],r['j'])))]=int(r['pred'])
    return graphs


def arguments(condition,device):
    old=sys.argv[:]
    try:
        sys.argv=[str(B.__file__),'--reference_sdf',str(sdf(condition)),'--output','unused',
                  '--methods','bondnet','--stage1_ckpt',str(STAGE1),'--refiner_ckpt',str(REFINER),'--device',device,'--noise','0']
        a=B.parse_args();a.state_catalogue=None;return a
    finally:sys.argv=old


def native_build(z,pairs,labels,q,h):
    rw=Chem.RWMol()
    for atomic,charge,hydrogen in zip(z,q,h):
        atom=Chem.Atom(int(atomic));atom.SetFormalCharge(int(charge));atom.SetNumExplicitHs(int(hydrogen));atom.SetNoImplicit(True);rw.AddAtom(atom)
    kinds={1:Chem.BondType.SINGLE,2:Chem.BondType.DOUBLE,3:Chem.BondType.TRIPLE,4:Chem.BondType.AROMATIC}
    for (i,j),c in zip(pairs,labels):
        if c:
            rw.AddBond(i,j,kinds[c])
            if c==4:rw.GetAtomWithIdx(i).SetIsAromatic(True);rw.GetAtomWithIdx(j).SetIsAromatic(True)
    try:
        mol=rw.GetMol();Chem.SanitizeMol(mol)
        good=len(Chem.GetMolFrags(mol))==1 and not any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
        return Chem.MolToSmiles(mol,isomericSmiles=False),bool(good)
    except Exception:return None,False


@torch.no_grad()
def infer(row,condition,models,args,device):
    idx,name,reference=row
    z,coords=B.coordinate_view(reference,0.,42+idx*1009)
    Q=Chem.GetFormalCharge(reference);calls=[];original=M.joint_decode
    def capture(*a,**kw):
        out=original(*a,**kw);calls.append((out,a,kw));return out
    start=time.perf_counter()
    with patch.object(M,'joint_decode',side_effect=capture):
        graph=B.bondnet_predict(z,coords,Q,*models,args,device)
    assert len(calls)==1
    result,callargs,kwargs=calls[0]
    assert kwargs['charge_logits'] is None and kwargs['implicit_h_logits'] is None
    _,positions,unique,_=M._undirected_scores(callargs[0],callargs[1])
    labels=np.array([int(result.labels[positions[e][0]]) for e in unique])
    assert graph=={e:int(c) for e,c in zip(unique,labels) if c}
    q=result.formal_charge.cpu().numpy();latent_h=result.implicit_h_count.cpu().numpy()
    final=None;error=None
    if result.solved:
        candidate={'elems':z,'edges':unique,'edge_labels':labels,'formal_charge':q,'implicit_h':latent_h}
        final,error=C._candidate_mol(candidate,return_error=True)
    else:error='MILP_unsolved'
    pred={'complete':False,'good':False,'native':False,'smiles':None,'pairs':[],'labels':[],'q':None,'h':None}
    if final is not None:
        pairs=[];classes=[]
        for bond in final.GetBonds():
            pairs.append(tuple(sorted((bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()))));classes.append(B.bond_label(bond))
        fq=[a.GetFormalCharge() for a in final.GetAtoms()];fh=[a.GetTotalNumHs(includeNeighbors=True) for a in final.GetAtoms()]
        vocab=all(-2<=a<=2 and 0<=b<=4 for a,b in zip(fq,fh))
        smi,good=native_build(z.tolist(),pairs,classes,fq,fh)
        good=bool(good and sum(fq)==Q and vocab)
        pred.update(complete=True,good=good,smiles=smi,pairs=pairs,labels=classes,q=fq,h=fh,
                    q_changed_by_cleanup=fq!=q.tolist(),sum_correct=sum(fq)==Q,in_vocabulary=vocab)
        assert smi==Chem.MolToSmiles(final,isomericSmiles=False)
    # Reference identity/atomic states enter only after output completion.
    target=Chem.MolToSmiles(reference,isomericSmiles=False)
    pred['native']=bool(pred['good'] and pred['smiles']==target)
    return {'name':name,'source_index':idx,'z':z.tolist(),'Q':Q,'raw_pairs':unique,'raw_labels':labels.tolist(),
            'raw_q':q.tolist(),'latent_h_unused':latent_h.tolist(),'MILP_solved':bool(result.solved),
            'MILP_kekulizable_flag':bool(result.kekulizable),'raw_B':[[i,j,c] for (i,j),c in sorted(graph.items())],
            'completion_error':error,'final':pred,'target_smiles':target,
            'target_q':[a.GetFormalCharge() for a in reference.GetAtoms()],
            'target_h':[a.GetTotalNumHs(includeNeighbors=True) for a in reference.GetAtoms()],
            'seconds':time.perf_counter()-start}


def models(device):
    a=arguments('pose_clean',device)
    stage1,(refiner,train_args)=B.load_bondnet(a,device)
    assert not bool(getattr(train_args,'single_stage',False))
    return stage1,refiner,train_args


def preflight(device,digest):
    models_=models(device);records=[]
    for condition in ('pose_clean','pose_noise010','ccd_ideal'):
        graphs=old_graphs(condition);a=arguments(condition,device)
        for row in sorted(references(condition),key=lambda r:r[1])[:3]:
            r=infer(row,condition,models_,a,device)
            expected=graphs[r['name']];actual={(i,j):c for i,j,c in r['raw_B']}
            records.append({'condition':condition,'name':r['name'],'raw_graph_equal':actual==expected,'seconds':r['seconds']})
    p=HERE/'PREFLIGHT.json';out={'plan_sha256':digest,'device':device,'records':records,'mismatches':[r for r in records if not r['raw_graph_equal']]}
    if p.exists():assert read(p)['plan_sha256']==digest
    else:save(p,out)
    print(json.dumps(out,indent=2),flush=True)
    assert not out['mismatches'],'Graph replay mismatch; inspect before full run'


def run(device,digest):
    assert read(HERE/'PREFLIGHT.json')['plan_sha256']==digest and not read(HERE/'PREFLIGHT.json')['mismatches']
    model=models(device)
    for condition in CONDITIONS:
        folder=HERE/f'evaluations/{condition}';folder.mkdir(parents=True,exist_ok=True)
        path=folder/'predictions.jsonl';done=folder/'DONE.json'
        if done.exists():
            receipt=read(done);assert receipt['plan_sha256']==digest and receipt['sha256']==sha(path);print('completed',condition,flush=True);continue
        existing=[]
        if path.exists():existing=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rows=references(condition);graphs=old_graphs(condition)
        assert [r['name'] for r in existing]==[r[1] for r in rows[:len(existing)]]
        for r in existing:assert r['plan_sha256']==digest
        a=arguments(condition,device);start=time.perf_counter()
        with path.open('a',encoding='utf-8') as handle:
            for i,row in enumerate(rows[len(existing):],start=len(existing)):
                r=infer(row,condition,model,a,device);r['plan_sha256']=digest
                actual={(j,k):c for j,k,c in r['raw_B']}
                r['raw_graph_replay_equal']=actual==graphs[r['name']]
                if not r['raw_graph_replay_equal']:
                    save(folder/f'MISMATCH_{i}.json',r)
                    raise ValueError(f'Frozen graph parity failed at {condition}/{r["name"]}; no completion result appended')
                handle.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n');handle.flush()
                if (i+1)%25==0:print(condition,i+1,'/',len(rows),f'{time.perf_counter()-start:.1f}s',flush=True)
        save(done,{'plan_sha256':digest,'sha256':sha(path),'n':len(rows),'device':device,'rdkit':rdBase.rdkitVersion})
    check()


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','run']);p.add_argument('--device',default='cpu')
    a=p.parse_args();torch.set_num_threads(4);RDLogger.DisableLog('rdApp.*');digest=plan()
    if a.action=='preflight':preflight(a.device,digest)
    else:run(a.device,digest)


if __name__=='__main__':main()
