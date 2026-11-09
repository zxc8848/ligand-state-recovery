"""Complete archived raw-graph table cells from frozen dumps and verdicts only."""
import collections
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
from rdkit import Chem,RDLogger

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
CORE=ROOT/'experiments/prob_mask_20261002';sys.path.insert(0,str(CORE))
import evaluation_common as E

DEV=ROOT/'work/evaluation_fix_v3/development/unified'
AUDIT=ROOT/'experiments/mask_control_20261101/IDENTITY_AUDIT.json'
SENS=ROOT/'experiments/source_excluded_sensitivity_20261101/SUMMARY.json'
CONDS=('locked','pdbbind_sigma0p00','pdbbind_sigma0p05','pdbbind_sigma0p10','pdbbind_sigma0p15','ccd_sigma0p00','ccd_sigma0p10')
METRICS=('strict','unified_resonance','smiles_species')
SEEDS=(20261002,20261003,20261004)


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf-8')


def paths(c):
    if c=='locked':return (ROOT/'data/pdbbind_pose_locked500.sdf',CORE/'locked_eval/old_per_edge.jsonl',CORE/'locked_eval/unified_old.json')
    v=ROOT/'work/paper2_noise_ccd_probe'/c
    return v/'reference_for_scoring.sdf',v/'bondnet_per_edge.jsonl',DEV/c/'old.json'


def contrast(current,old,names,refs):
    values=np.array([[int(m[n])-int(old[n]) for n in names] for m in current],dtype=float).mean(0)
    groups=collections.defaultdict(list)
    for i,n in enumerate(names):groups[Chem.MolToSmiles(refs[n],isomericSmiles=False)].append(i)
    members=list(groups.values());sums=np.array([values[g].sum() for g in members]);counts=np.array([len(g) for g in members])
    rng=np.random.default_rng(20261102);idx=rng.integers(0,len(members),(5000,len(members)))
    draws=sums[idx].sum(1)/counts[idx].sum(1)
    return {'delta':float(values.mean()),'identity_cluster_ci95':np.quantile(draws,[.025,.975]).tolist()}


def main():
    RDLogger.DisableLog('rdApp.*');files=[Path(__file__),AUDIT,SENS,CORE/'evaluation_common.py',
        ROOT/'experiments/prob_mask_20261002/joint_pose/geom_overlap_names.json',ROOT/'data/ccd_control_identity_hits.json']
    for c in CONDS:
        files+=list(paths(c))
        if c!='locked':files+=[DEV/c/f'new_seed{s}.json' for s in SEEDS]+[DEV/c/'knodle.json']
    spec={'scope':'post hoc saved raw graph analysis; no model inference, no rescoring or new p-values',
          'typed_recall':'all correctly typed reference bonds / all reference bonds; missing predictions zero; false positives do not alter this diagnostic',
          'subsets':'reuse exactly the saved407 pose/926 CCD source-exclusion sets, verified against original audit',
          'graph_output':'archived raw MILP B before declared complete-output normalization',
          'sha256':{str(p.relative_to(ROOT)):sha(p) for p in files}}
    p=HERE/'PLAN.json'
    if p.exists():assert read(p)['spec']==spec
    else:save(p,{'recorded_utc':datetime.now(timezone.utc).isoformat(),'spec':spec})
    audit=read(AUDIT);sensitivity=read(SENS);out={'plan_sha256':sha(p),'conditions':{}}
    for c in CONDS:
        refpath,dump,verdict=paths(c);refs,excluded=E.reference_population(refpath)
        _,old=E.load_verdicts(verdict);assert set(old)==set(refs)
        predicted={n:{} for n in refs};truth_dump={n:{} for n in refs};seen=collections.defaultdict(set)
        for line in dump.read_text().splitlines():
            if not line.strip():continue
            r=json.loads(line);name=r['name']
            if name in excluded:continue
            assert name in refs;edge=tuple(sorted((r['i'],r['j'])))
            assert edge not in seen[name];seen[name].add(edge)
            if r['pred']:predicted[name][edge]=r['pred']
            if r['true']:truth_dump[name][edge]=r['true']
        counts={};correct=total=0
        for name,mol in refs.items():
            truth={tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))):4 if b.GetIsAromatic() else int(b.GetBondTypeAsDouble()) for b in mol.GetBonds()}
            assert truth==truth_dump[name],(c,name,'truth mismatch')
            ncorrect=sum(predicted[name].get(e,0)==label for e,label in truth.items())
            counts[name]={'correct':ncorrect,'total':len(truth)};correct+=ncorrect;total+=len(truth)
            assert (predicted[name]==truth)==bool(old[name]['strict'])
        names=sorted(refs);full={m:statistics.mean(int(old[n][m]) for n in names) for m in METRICS}
        result={'n':len(names),'reference_edges':total,'correctly_typed_reference_edges':correct,'typed_true_edge_recall':correct/total,
                'full':full,'per_molecule_edge_counts':counts}
        if c=='locked':assert total==14518 and len(names)==491
        else:
            pose=c.startswith('pdbbind');key='pose_clean' if pose else 'ccd_ideal'
            excluded_names=set(audit['conditions'][key]['sensitivity_exclusions'])
            previous=sensitivity['conditions'][c]
            assert excluded_names==set(previous['excluded_names'])
            chosen=[n for n in names if n not in excluded_names];assert len(chosen)==(407 if pose else 926)
            models=[E.load_verdicts(DEV/c/f'new_seed{s}.json')[1] for s in SEEDS]
            _,knodle=E.load_verdicts(DEV/c/'knodle.json')
            sub={'n':len(chosen),'excluded_names':sorted(excluded_names),'selected_names':chosen,
                 'reference_edges':sum(counts[n]['total'] for n in chosen),
                 'correctly_typed_reference_edges':sum(counts[n]['correct'] for n in chosen),'metrics':{}}
            sub['typed_true_edge_recall']=sub['correctly_typed_reference_edges']/sub['reference_edges']
            for metric in METRICS:
                ours=[statistics.mean(int(m[n][metric]) for n in chosen) for m in models]
                kn=statistics.mean(int(knodle[n][metric]) for n in chosen)
                old_rate=statistics.mean(int(old[n][metric]) for n in chosen)
                assert abs(statistics.mean(ours)-previous['metrics'][metric]['subset']['ours_mean'])<1e-12
                assert abs(kn-previous['metrics'][metric]['subset']['knodle'])<1e-12
                sub['metrics'][metric]={'archived_raw':old_rate,'current_per_head':ours,
                    'current_mean':statistics.mean(ours),'current_sd':statistics.stdev(ours),'knodle':kn,
                    'current_minus_archived':contrast([{n:m[n][metric] for n in chosen} for m in models],
                                                    {n:old[n][metric] for n in chosen},chosen,refs)}
            result['source_excluded']=sub
            if c=='pdbbind_sigma0p00':assert total==14269 and len(names)==488
        out['conditions'][c]=result
    for rel,want in spec['sha256'].items():assert sha(ROOT/rel)==want,rel
    save(HERE/'SUMMARY.json',out)
    lines=['# Archived raw-graph main-table completion','','Saved predictions and existing unified verdicts only; no inference, no new tests.', '',
           '| Condition | n | Strict | Resonance | Graph-SMILES | Typed correct / reference | Typed recall |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for c,r in out['conditions'].items():
        f=r['full'];lines.append(f"| {c} | {r['n']} | {f['strict']:.4f} | {f['unified_resonance']:.4f} | {f['smiles_species']:.4f} | {r['correctly_typed_reference_edges']}/{r['reference_edges']} | {r['typed_true_edge_recall']:.4f} |")
    lines+=['','## Fixed source-excluded resonance rates','','| Condition | n | Current D | Knodle | Archived raw |','|---|---:|---:|---:|---:|']
    for c,r in out['conditions'].items():
        if c=='locked':continue
        s=r['source_excluded'];m=s['metrics']['unified_resonance'];lines.append(f"| {c} | {s['n']} | {100*m['current_mean']:.2f} | {100*m['knodle']:.2f} | {100*m['archived_raw']:.2f} |")
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(HERE/'VERIFICATION.json',{'all_input_hashes_unchanged':True,'all_strict_flags_reproduce_edge_dicts':True,
        'all_reference_edge_counts_checked':True,'original_source_exclusion_membership_and_current_Knodle_rates_unchanged':True,
        'model_inference':False,'new_hypothesis_tests':0,'original_outputs_modified':False})
    print('\n'.join(lines))


if __name__=='__main__':main()
