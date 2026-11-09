"""Trace archived hydrogen semantics with tiny API controls, not a model benchmark."""
import ast
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch

import numpy as np
import torch
from rdkit import Chem,RDLogger,rdBase

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LEGACY=ROOT/'archive/legacy_two_stage_20261003'
REPAIR=LEGACY/'bondnet/inference/chemical_repair.py'
CANDIDATE=LEGACY/'scripts/evaluate_candidate_sets.py'
BENCHMARK=LEGACY/'scripts/benchmark_templatefree_motifs.py'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module


def summary(mol):
    if mol is None:return {'exists':False,'smiles':None,'closed_connected':False,'q':None,'h':None,'radicals':None}
    return {'exists':True,'smiles':Chem.MolToSmiles(mol,isomericSmiles=False),
            'closed_connected':len(Chem.GetMolFrags(mol))==1 and not any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()),
            'q':[a.GetFormalCharge() for a in mol.GetAtoms()],
            'h':[a.GetTotalNumHs(includeNeighbors=True) for a in mol.GetAtoms()],
            'radicals':[a.GetNumRadicalElectrons() for a in mol.GetAtoms()]}


def reference_case(smiles,delocalize=False):
    mol=Chem.MolFromSmiles(smiles);assert mol is not None
    z=torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()],dtype=torch.long)
    unique=[tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))) for b in mol.GetBonds()]
    labels=[4 if b.GetIsAromatic() else int(b.GetBondTypeAsDouble()) for b in mol.GetBonds()]
    if delocalize:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum()!=6:continue
            os=[a.GetIdx() for a in atom.GetNeighbors() if a.GetAtomicNum()==8 and a.GetDegree()==1]
            if len(os)==2:
                for o in os:labels[unique.index(tuple(sorted((atom.GetIdx(),o))))]=5
    # Oracle-like toy logits isolate API semantics, not model accuracy.
    directed=unique+[(j,i) for i,j in unique]
    logits=torch.full((len(directed),6),-12.)
    for i,c in enumerate(labels+labels):logits[i,c]=12.
    return mol,z,unique,torch.tensor(directed,dtype=torch.long),logits


def static_calls(path,functions):
    text=path.read_text(encoding='utf-8');tree=ast.parse(text);lines=text.splitlines();out=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            name=node.func.id if isinstance(node.func,ast.Name) else getattr(node.func,'attr','')
            if name in functions:
                out.append({'call':name,'line':node.lineno,'keywords':{k.arg:ast.unparse(k.value) for k in node.keywords},
                            'source':'\n'.join(lines[node.lineno-1:node.end_lineno])})
    return out


def main():
    torch.set_num_threads(2);RDLogger.DisableLog('rdApp.*')
    inputs=[Path(__file__),REPAIR,REPAIR.with_name('structured_decoder.py'),CANDIDATE,BENCHMARK,
            LEGACY/'scripts/evaluate_species_recovery.py',LEGACY/'scripts/run_molcraft_bondnet.py',
            ROOT/'scripts/run_locked_eval.ps1',ROOT/'experiments/prob_mask_20261002/locked_eval/old_benchmark.json']
    config={'status':'post hoc source/API semantics audit, no real model inference or scoring',
            'cases':['CC','CCO','Cc1ccccc1','c1ccccc1','CC(=O)O','C[N+](C)(C)C','c1cc[nH]c1','acid_class5'],
            'toy_logits':'high confidence reference-like B only to isolate plumbing; never report as performance',
            'paths':['joint_decode latent H','explicit latent-H build','RDKit-inferred-H build',
                     'decode_top_n guard calls','candidate builder H perturbation','optional H-logit capability'],
            'sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs}}
    plan=HERE/'PLAN.json'
    if plan.exists():assert json.loads(plan.read_text())['spec']==config
    else:save(plan,{'recorded_utc':datetime.now(timezone.utc).isoformat(),'spec':config})
    # Load only the archived inference package, with no replacement of current
    # production bondnet modules and no code writes in the archive.
    pkg=types.ModuleType('milp_h_probe');pkg.__path__=[str(REPAIR.parent)];sys.modules[pkg.__name__]=pkg
    M=load_module('milp_h_probe.chemical_repair',REPAIR)
    E=load_module('milp_h_candidate_probe',CANDIDATE)
    records=[]
    for smiles,deloc in [(s,False) for s in config['cases'][:-1]]+[('CC(=O)O',True)]:
        ref,z,edges,ei,logits=reference_case(smiles,deloc)
        Q=Chem.GetFormalCharge(ref)
        raw=M.joint_decode(logits,ei,z,total_charge=Q,charge_logits=None,implicit_h_logits=None,time_limit=2.)
        _,_,unique,_=M._undirected_scores(logits,ei)
        _,rows,_,_=M._undirected_scores(logits,ei)
        labels=np.array([int(raw.labels[rows[e][0]]) for e in unique])
        q=raw.formal_charge.cpu().numpy();h=raw.implicit_h_count.cpu().numpy()
        fixed=summary(M.build_sanitized_mol(z,unique,labels,q,implicit_h=h))
        inferred=summary(M.build_sanitized_mol(z,unique,labels,q,implicit_h=None))
        candidate={'elems':z,'formal_charge':q,'implicit_h':h.copy(),'edges':unique,'edge_labels':labels}
        old_builder=summary(E._candidate_mol(candidate))
        perturbed=dict(candidate,implicit_h=np.full(len(z),4,dtype=np.int64))
        old_perturbed=summary(E._candidate_mol(perturbed))
        assert old_builder==old_perturbed,'Archived candidate builder unexpectedly reads implicit_h'
        guard_calls=[];actual=M._kekulizable
        def spy(*args,**kwargs):
            H=kwargs.get('implicit_h',args[4] if len(args)>4 else None)
            guard_calls.append({'implicit_h':None if H is None else np.asarray(H).tolist(),
                                'all_atoms_explicit_h':kwargs.get('all_atoms_explicit_h',False)})
            return actual(*args,**kwargs)
        with patch.object(M,'_kekulizable',side_effect=spy):
            ranked=M.decode_top_n(logits,ei,z,total_charge=Q,n_solutions=1,max_attempts=1,
                                  require_kekulizable=True,charge_logits=None,implicit_h_logits=None,time_limit=2.)
        assert guard_calls and any(r['implicit_h'] is None for r in guard_calls)
        ref_smi=Chem.MolToSmiles(ref,isomericSmiles=False)
        records.append({'case':'acid_class5' if deloc else smiles,'Q':Q,'reference_smiles':ref_smi,
                        'reference_h_for_scoring_only':[a.GetTotalNumHs() for a in ref.GetAtoms()],
                        'solved':raw.solved,'reported_kekulizable':raw.kekulizable,'labels':labels.tolist(),
                        'q_MILP':q.tolist(),'h_MILP':h.tolist(),'raw_h_all_zero':bool(np.all(h==0)),
                        'explicit_MILP_h_build':fixed,'inferred_H_build':inferred,
                        'archived_candidate_build':old_builder,'candidate_H_perturbation_invariant':True,
                        'fixed_h_native':bool(fixed['closed_connected'] and fixed['smiles']==ref_smi),
                        'inferred_h_native':bool(inferred['closed_connected'] and inferred['smiles']==ref_smi),
                        'candidate_native':bool(old_builder['closed_connected'] and old_builder['smiles']==ref_smi),
                        'top_n_solutions':len(ranked),'top_n_guard_H_inputs':guard_calls})
    # The API can use H logits when explicitly provided. This is not what the
    # archived heavy-only benchmark enables by default.
    ref,z,edges,ei,logits=reference_case('CCO')
    hlogits=torch.full((3,5),-12.)
    for i,h in enumerate((3,2,1)):hlogits[i,h]=12.
    learned=M.joint_decode(logits,ei,z,total_charge=0,implicit_h_logits=hlogits,time_limit=2.)
    assert learned.implicit_h_count.tolist()==[3,2,1]
    learned_build=summary(M.build_sanitized_mol(z,edges,learned.labels[:len(edges)].numpy(),
                                              learned.formal_charge.numpy(),learned.implicit_h_count.numpy()))
    assert learned_build['closed_connected'] and learned_build['smiles']=='CCO'
    old=json.loads(inputs[-1].read_text())
    assert old['input_mode']=='heavy_only' and old['heavy_latent_charge_h'] is True and old['use_heavy_atom_state_logits'] is False
    ethanol=next(r for r in records if r['case']=='CCO')
    assert not ethanol['fixed_h_native'] and ethanol['inferred_h_native'] and ethanol['candidate_native']
    assert not ethanol['explicit_MILP_h_build']['closed_connected']
    deloc=next(r for r in records if r['case']=='acid_class5')
    assert deloc['inferred_H_build']['smiles']=='CC(O)O' and deloc['archived_candidate_build']['smiles']=='CC(=O)O'
    for rel,want in config['sha256'].items():assert sha(ROOT/rel)==want,rel
    out={'plan_sha256':sha(plan),'rdkit':rdBase.rdkitVersion,'n_toy_cases':len(records),
         'toy_results':records,'API_with_H_logits':{'h':learned.implicit_h_count.tolist(),'build':learned_build},
         'frozen_heavy_only_metadata':{k:old[k] for k in ('input_mode','heavy_latent_charge_h','use_heavy_atom_state_logits')},
         'actual_call_sites':{str(p.relative_to(ROOT)):static_calls(p,{'joint_decode','decode_top_n','build_sanitized_mol','_kekulizable','_candidate_mol'})
                              for p in (REPAIR,BENCHMARK,CANDIDATE,LEGACY/'scripts/run_molcraft_bondnet.py')},
         'all_assertions_passed':True,'all_input_hashes_unchanged':True,'real_model_forward_passes':0,
         'decision':'Archived heavy-only benchmark outputs B for scoring; latent MILP H is not its deployed hydrogen prediction. A native comparator must explicitly define MILP(B,q)+RDKit-H completion (and any class5 normalization) as a baseline output policy, then score saved B/q/h with the shared native builder.'}
    save(HERE/'AUDIT.json',out)
    lines=['# Archived MILP hydrogen-path audit','','No model benchmark, training or frozen-output changes. Toy graphs isolate API semantics.','',
           '| Toy | MILP h | Explicit-h closed/native | Inferred-h closed/native | Archived candidate native |',
           '|---|---|---|---|---|']
    for r in records:
        lines.append(f"| {r['case']} | {r['h_MILP']} | {r['explicit_MILP_h_build']['closed_connected']}/{r['fixed_h_native']} | {r['inferred_H_build']['closed_connected']}/{r['inferred_h_native']} | {r['candidate_native']} |")
    lines+=['',out['decision'],'','The auxiliary H-logit API can select ethanol H=[3,2,1], but frozen heavy-only metadata disables both H and charge logits.',
            'Sanitization alone may accept radicals; native scoring must retain the common connected, closed-shell criterion.',
            'Candidate build ignores changed implicit_h in every toy case. Class5 also differs: the production generic builder maps5 to single, while the archived candidate builder performs a predicted-graph normalization before inferring H.',
            'The single-graph benchmark export discards q/h, so existing per-edge dumps cannot reconstruct the original MILP charge choices. A new explicit output run is needed for native comparison.',
            'These eight controls are plumbing checks, not estimates of baseline accuracy or an argument that learned atom states win.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines))


if __name__=='__main__':main()
