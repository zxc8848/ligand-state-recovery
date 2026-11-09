"""Frozen-model charge-enriched diagnostic. No training or old locked writes."""
import argparse
import collections
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from rdkit import RDLogger

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT/'experiments/prob_mask_20261002'
sys.path.insert(0, str(OLD))
import joint as J
import jointt as T
import probe as P
from cluster_bootstrap import mcnemar

SEEDS = (20261002, 20261003, 20261004)
GROUPS = ('cation', 'anion', 'separated_neutral', 'uncharged_neutral')
ARMS = ('neural_constrained', 'catalogue_constrained', 'neural_unconstrained')
ENCODER = ROOT/'checkpoints/revision_v3_one_stage_heavy_seed42/best_e2e.pt'
SOURCES = OLD/'joint_v1/sources.pt'


def log(s):
    print(datetime.now().isoformat(timespec='seconds'), s, flush=True)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def checkpoint(seed):
    return OLD/f'joint_v5/seed{seed}/best.pt'


def freeze():
    paths=[Path(__file__), HERE/'prepare.py', HERE/'README.md', HERE/'selected.pt',
           HERE/'reference.sdf', HERE/'DATASET.json', ENCODER, SOURCES]
    paths += [OLD/f for f in ('joint.py','jointt.py','probe.py','cluster_bootstrap.py')]
    paths += [checkpoint(s) for s in SEEDS]
    spec={'schema':'bondnet.charged-diagnostic.v1','scope':'CCD ideal; source-excluded; no training',
          'head_seeds':SEEDS,'arms':ARMS,'groups':GROUPS,
          'candidate_budget':{'fraction':.1,'cap':8,'topk':2,'beam':48},
          'same_within_pair':['coordinates','features','candidate labels/order','bond scores','catalogue','builder','selection rule'],
          'Q_embedding':'true reference Q in every arm; unconstrained arm removes only sum(q)=Q decoding',
          'charge_solver':'released greedy suffix sum-product solver, not joint MAP',
          'delocalized_rewriting':False,'coordinate_noise':0,
          'primary':'native molecular identity; report each stratum and equal-weight four-stratum macro',
          'secondary':'strict graph; atom-index q/h diagnostics; no secondary hypothesis tests',
          'native_test_family':30,'test_scopes':['pooled',*GROUPS],
          'contrasts':['neural_constrained-catalogue_constrained','neural_constrained-neural_unconstrained'],
          'bootstrap_reps':2000,'training_seed_uncertainty':'three-head sample SD, fixed encoder',
          'limitations':['ideal geometry, not experimental ionic poses','reference q/h are curated annotations, not measured protonation','small separated-neutral stratum','not scaffold-disjoint','post hoc research diagnostic, not untouched confirmatory test'],
          'sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths}}
    spec=json.loads(json.dumps(spec))
    path=HERE/'PROTOCOL.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['spec']!=spec: raise ValueError('Frozen protocol/input/code drift; do not reuse outputs')
    else: save(path,{'frozen_utc':datetime.now(timezone.utc).isoformat(),'spec':spec})
    return sha(path)


def blind(row):
    return {k:row[k] for k in ('z','pairs','feat','node_feat','base','Q')} | {
        'labels':torch.zeros(len(row['pairs']),dtype=torch.long),
        'atom_target':torch.full((len(row['z']),),10,dtype=torch.long)}


def beam(blog):
    if len(blog)==0: return [torch.empty(0,dtype=torch.long,device=blog.device)]
    base=blog.argmax(-1)
    k=min(8,max(1,round(.1*len(base))))
    flagged=torch.topk(-blog.gather(1,base[:,None]).squeeze(1),k).indices.tolist()
    options=[torch.topk(blog[e],2).indices.tolist() for e in flagged]
    choices=list(itertools.product(*options))
    choices.sort(key=lambda c:-sum(float(blog[e,v]) for e,v in zip(flagged,c)))
    candidates=[]
    for choice in choices[:48]:
        b=base.clone()
        for e,v in zip(flagged,choice): b[e]=v
        candidates.append(b)
    assert torch.equal(candidates[0],base)
    return candidates


def assign_and_select(row,candidates,bscores,logits,constrained):
    n=len(row['z']); count=len(candidates)
    if constrained:
        atoms=J.constrained_atoms(logits,torch.full((n*count,),J.AMASK,dtype=torch.long,device=logits.device),
              [n]*count,torch.full((count,),int(row['Q']),device=logits.device),np.random.default_rng(11),greedy=True)
    else: atoms=logits.argmax(-1)
    parts=atoms.cpu().split(n)
    lps=logits.log_softmax(-1).cpu().split(n)
    details=[]
    for b,a,lp,bs in zip(candidates,parts,lps,bscores):
        qsum=int((a//5-2).sum())
        if constrained: assert qsum==row['Q']
        smi,good=J.build_species(row,b.cpu(),a)
        details.append({'atom_states':a.tolist(),'smiles':smi,'constructible':bool(good),
                        'qsum':qsum,'score':bs+float(lp.gather(1,a[:,None]).sum())})
    selected=max(range(count),key=lambda i:(details[i]['constructible'],details[i]['score']))
    return {'selected_index':selected,**details[selected], 'labels':candidates[selected].cpu().tolist()}


@torch.no_grad()
def infer(net,source,catalogue,device):
    row=blind(source)
    one=J.pack([row],device)
    bl,_=net(one,torch.full_like(one['labels'],J.BMASK),torch.full_like(one['atarget'],J.AMASK))
    blog=bl.log_softmax(-1); candidates=beam(blog)
    bscores=[float(blog.gather(1,b[:,None]).sum()) for b in candidates]
    batch=J.pack([row]*len(candidates),device); states=torch.cat(candidates)
    _,neural=net(batch,states,torch.full_like(batch['atarget'],J.AMASK))
    prior=J.catalogue_logits(batch,states,catalogue)
    shared={'pairs':row['pairs'].tolist(),'candidates':[b.cpu().tolist() for b in candidates],'bond_scores':bscores}
    results={ARMS[0]:assign_and_select(row,candidates,bscores,neural+prior,True),
             ARMS[1]:assign_and_select(row,candidates,bscores,prior,True),
             ARMS[2]:assign_and_select(row,candidates,bscores,neural+prior,False)}
    # Targets are attached ONLY after all predictions and selections are complete.
    for item in results.values():
        a=torch.tensor(item['atom_states'])
        graph={tuple(e):c for e,c in zip(shared['pairs'],item['labels']) if c}
        item['native']=bool(item['constructible'] and item['smiles']==source['smiles'])
        item['strict']=graph==source['truth']
        item['charge_sum_correct']=item['qsum']==source['Q']
        item['q_exact']=bool(torch.equal(a//5-2,source['q']))
        item['h_exact']=bool(torch.equal(a%5,source['h']))
    return {'name':source['id'],'bucket':source['bucket'],'Q':source['Q'],
            'target_smiles':source['smiles'],'target_q':source['q'].tolist(),'target_h':source['h'].tolist(),
            'pool_sha256':hashlib.sha256(json.dumps(shared,sort_keys=True).encode()).hexdigest(),
            'candidate_count':len(candidates),'arms':results}


def selftest():
    row={'z':torch.tensor([6,6]),'pairs':torch.tensor([[0,1]]),'Q':0}
    lp=torch.full((2,25),-20.)
    lp[:,13]=0  # carbon, q=0, h=3
    out=assign_and_select(row,[torch.tensor([1])],[0.],lp,True)
    assert out['smiles']=='CC' and out['constructible'] and out['qsum']==0
    # Known-Q embedding is distinct from enforcing the sum during decoding.
    row2=dict(row,Q=1)
    free=assign_and_select(row2,[torch.tensor([1])],[0.],lp,False)
    fixed=assign_and_select(row2,[torch.tensor([1])],[0.],lp,True)
    assert free['qsum']==0 and fixed['qsum']==1
    poison={'z':row['z'],'pairs':row['pairs'],'Q':0,'feat':torch.zeros(1,360),
            'node_feat':torch.zeros(2,360),'base':torch.zeros(1,5),
            'atom_target':torch.tensor([0,24]),'truth':{(0,1):3},'q':torch.tensor([-2,2]),'h':torch.tensor([0,4])}
    clean=blind(poison)
    assert not {'truth','q','h','smiles'} & set(clean)
    assert clean['atom_target'].tolist()==[10,10]
    assert len(beam(torch.empty(0,5)))==1
    log('selftest passed: builder, charge constraint, blinded fields, empty beam')


def features(protocol,device):
    path=HERE/'features.pt'; receipt=HERE/'FEATURES.json'
    if path.exists():
        meta=json.loads(receipt.read_text());assert meta['protocol_sha256']==protocol and meta['sha256']==sha(path)
        return torch.load(path,weights_only=False,map_location='cpu')
    rows=torch.load(HERE/'selected.pt',weights_only=False,map_location='cpu')
    backbone,args=P.load_backbone(ENCODER,device)
    geos=[P.geometry(r,0.,np.random.default_rng(11),args['cutoff']) for r in rows]
    out=[]
    for i in range(0,len(geos),8):out.extend(J.extract(backbone,geos[i:i+8],device))
    torch.save(out,path);save(receipt,{'protocol_sha256':protocol,'sha256':sha(path),'n':len(out),'cutoff':args['cutoff']})
    del backbone
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    return out


def cell(seed,rows,catalogue,device,protocol):
    root=HERE/f'evaluations/seed{seed}'; done=root/'DONE.json'; pred=root/'predictions.json.gz'
    if done.exists():
        d=json.loads(done.read_text());assert d['protocol_sha256']==protocol and d['sha256']==sha(pred)
        log(f'completed cell verified: {seed}');return
    net,_=T.load_joint(checkpoint(seed),device)
    records=[];start=time.perf_counter()
    for i,row in enumerate(rows):
        records.append(infer(net,row,catalogue,device))
        if (i+1)%25==0:log(f'seed {seed}: {i+1}/{len(rows)}; {time.perf_counter()-start:.1f}s')
    root.mkdir(parents=True,exist_ok=True)
    with gzip.open(pred,'wt',encoding='utf-8') as f:json.dump({'protocol_sha256':protocol,'records':records},f,separators=(',',':'))
    save(done,{'protocol_sha256':protocol,'sha256':sha(pred),'n':len(records),'seconds':time.perf_counter()-start})
    del net
    if str(device).startswith('cuda'):torch.cuda.empty_cache()


def metric(records,arm):
    items=[r['arms'][arm] for r in records]
    out={k:statistics.mean(x[k] for x in items) for k in
         ('native','strict','constructible','charge_sum_correct','q_exact','h_exact')}
    truth=np.array([v for r in records for v in r['target_q']])
    pred=np.array([s//5-2 for x in items for s in x['atom_states']])
    htrue=np.array([v for r in records for v in r['target_h']])
    hpred=np.array([s%5 for x in items for s in x['atom_states']])
    out['q_atom_accuracy']=float(np.mean(truth==pred));out['h_atom_accuracy']=float(np.mean(htrue==hpred))
    cm=np.zeros((5,5),dtype=int)
    for t,p in zip(truth,pred):cm[t+2,p+2]+=1
    f1=[]
    for q in sorted(set(truth)):
        j=q+2;tp=cm[j,j];den=cm[j,:].sum()+cm[:,j].sum()
        f1.append(2*tp/den if den else 0.)
    out['q_macro_f1_truth_present_classes']=float(np.mean(f1))
    tp=int(((truth!=0)&(truth==pred)).sum());fp=int(((pred!=0)&(truth!=pred)).sum());fn=int(((truth!=0)&(truth!=pred)).sum())
    out['nonzero_q_micro_f1']=2*tp/(2*tp+fp+fn) if np.any(truth!=0) else None
    out['q_confusion_matrix_minus2_to_plus2']=cm.tolist();out['n_atoms']=len(truth)
    return out


def summarize(protocol):
    data={}
    for seed in SEEDS:
        with gzip.open(HERE/f'evaluations/seed{seed}/predictions.json.gz','rt') as f:
            p=json.load(f);assert p['protocol_sha256']==protocol
            data[seed]={r['name']:r for r in p['records']}
    names=sorted(data[SEEDS[0]])
    assert all(sorted(d)==names for d in data.values())
    scopes={'pooled':names}|{g:[n for n in names if data[SEEDS[0]][n]['bucket']==g] for g in GROUPS}
    result={'scope':'new CCD ideal charge-enriched diagnostic; not a real-pose or untouched confirmatory test',
            'protocol_sha256':protocol,'n':len(names),'cells':{},'tests':[],'native_contrasts':{}}
    for scope,ns in scopes.items():
        result['cells'][scope]={'n':len(ns),'arms':{}}
        for arm in ARMS:
            values=[metric([data[s][n] for n in ns],arm) for s in SEEDS]
            result['cells'][scope]['arms'][arm]={'per_seed':values,'mean':{},'sd':{}}
            for key in values[0]:
                if isinstance(values[0][key],(float,int)):
                    result['cells'][scope]['arms'][arm]['mean'][key]=statistics.mean(v[key] for v in values)
                    result['cells'][scope]['arms'][arm]['sd'][key]=statistics.stdev(v[key] for v in values)
        for control in ARMS[1:]:
            ds=np.array([[int(data[s][n]['arms'][ARMS[0]]['native'])-int(data[s][n]['arms'][control]['native']) for n in ns] for s in SEEDS])
            rng=np.random.default_rng(11);idx=rng.integers(0,len(ns),(2000,len(ns)))
            ci=np.quantile(ds.mean(0)[idx].mean(1),[.025,.975]).tolist()
            key=scope+'/'+control
            result['native_contrasts'][key]={'mean_difference':float(ds.mean()),'conditional_molecule_bootstrap_ci95':ci}
            for si,seed in enumerate(SEEDS):
                pairs=[(data[seed][n]['arms'][control]['native'],data[seed][n]['arms'][ARMS[0]]['native']) for n in ns]
                result['tests'].append({'scope':scope,'control':control,'seed':seed,'p_raw':mcnemar(pairs),'delta':float(ds[si].mean())})
    assert len(result['tests'])==30
    running=0.
    for rank,t in enumerate(sorted(result['tests'],key=lambda t:t['p_raw'])):
        running=max(running,min(1.,(30-rank)*t['p_raw']));t['p_holm']=running
    result['equal_weight_stratum_macro_native']={}
    for arm in ARMS:
        vals=[statistics.mean(result['cells'][g]['arms'][arm]['per_seed'][i]['native'] for g in GROUPS) for i in range(3)]
        result['equal_weight_stratum_macro_native'][arm]={'per_seed':vals,'mean':statistics.mean(vals),'sd':statistics.stdev(vals)}
    result['composition']={scope:{'n':len(ns),
        'Q_histogram':dict(collections.Counter(data[SEEDS[0]][n]['Q'] for n in ns)),
        'heavy_atom_min_median_max':[min(len(data[SEEDS[0]][n]['target_q']) for n in ns),
          statistics.median(len(data[SEEDS[0]][n]['target_q']) for n in ns),
          max(len(data[SEEDS[0]][n]['target_q']) for n in ns)]} for scope,ns in scopes.items()}
    result['equal_weight_macro_native_contrasts']={}
    for control in ARMS[1:]:
        rng=np.random.default_rng(11);boot=np.zeros(2000)
        for g in GROUPS:
            ns=scopes[g]
            d=np.array([[int(data[s][n]['arms'][ARMS[0]]['native'])-int(data[s][n]['arms'][control]['native']) for n in ns] for s in SEEDS]).mean(0)
            idx=rng.integers(0,len(ns),(2000,len(ns)));boot+=d[idx].mean(1)/len(GROUPS)
        result['equal_weight_macro_native_contrasts'][control]={
            'mean_difference':result['equal_weight_stratum_macro_native'][ARMS[0]]['mean']-result['equal_weight_stratum_macro_native'][control]['mean'],
            'conditional_stratified_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist()}
    save(HERE/'SUMMARY.json',result)
    lines=['# Charge-enriched frozen-model diagnostic','','CCD ideal coordinates; no training. See PROTOCOL.json for exclusions and limits.','',
           '| Stratum | n | Neural + Q constraint | Catalogue + Q constraint | Neural, no sum constraint |','|---|---:|---:|---:|---:|']
    for scope in scopes:
        c=result['cells'][scope];vals=[f"{100*c['arms'][a]['mean']['native']:.2f} +/- {100*c['arms'][a]['sd']['native']:.2f}" for a in ARMS]
        lines.append('| '+scope+' | '+str(c['n'])+' | '+' | '.join(vals)+' |')
    vals=[f"{100*result['equal_weight_stratum_macro_native'][a]['mean']:.2f} +/- {100*result['equal_weight_stratum_macro_native'][a]['sd']:.2f}" for a in ARMS]
    lines+=['| Four-stratum equal-weight macro | -- | '+' | '.join(vals)+' |','',
            'Values are native identity percentages, mean +/- sample SD across the three original heads.',
            'The 30 native McNemar comparisons have a separate descriptive Holm family. No prior family is changed.',
            'Atom-index q/h metrics and confusion matrices are in SUMMARY.json; they do not identify experimentally measured protonation states.']
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    log('\n'+'\n'.join(lines))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['selftest','run','report']);ap.add_argument('--device',default='cuda')
    a=ap.parse_args();RDLogger.DisableLog('rdApp.*');torch.set_num_threads(4)
    if a.action=='selftest':selftest();return
    protocol=freeze()
    if a.action=='run':
        selftest();rows=features(protocol,a.device)
        sources=torch.load(SOURCES,weights_only=False,map_location='cpu');catalogue=J.make_catalogue(sources['train'])
        for seed in SEEDS:cell(seed,rows,catalogue,a.device,protocol)
    assert freeze()==protocol
    summarize(protocol)


if __name__=='__main__':main()
