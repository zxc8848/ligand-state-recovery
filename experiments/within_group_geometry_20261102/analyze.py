"""Post hoc within-motif length ranks; no models or frozen-score changes."""
import collections
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger, rdBase
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PILOT = ROOT/'experiments/prob_mask_20261002'
INPUTS = {'ideal':ROOT/'data/ccd_heavy_leakagefree.sdf',
          'pose':ROOT/'data/pdbbind_ccd_pose500.sdf'}
ALLOWED = {6,7,8,9,14,15,16,17,33,35,53,83}
EPS = 1e-6
REPS = 5000


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def bootstrap(rows, key, field):
    clusters=sorted({r[key] for r in rows}); index={v:i for i,v in enumerate(clusters)}
    sums=np.zeros(len(clusters));counts=np.zeros(len(clusters))
    for r in rows:
        j=index[r[key]];sums[j]+=r[field];counts[j]+=1
    rng=np.random.default_rng(20261102)
    ids=rng.integers(0,len(clusters),(REPS,len(clusters)))
    draws=sums[ids].sum(1)/counts[ids].sum(1)
    return {'n_clusters':len(clusters),'ci95':np.quantile(draws,[.025,.975]).tolist()}


def collect(path):
    rows=[];eligible=0;skips=collections.Counter();all_lengths={'C':{1:[],2:[]},'P':{1:[],2:[]}}
    for idx,mol in enumerate(Chem.SDMolSupplier(str(path),removeHs=False)):
        if mol is None: skips['unparseable']+=1;continue
        if mol.GetNumConformers()==0:skips['no_coordinates']+=1;continue
        try:mol=Chem.RemoveHs(mol,sanitize=True)
        except Exception:skips['removeHs_failed']+=1;continue
        if not {a.GetAtomicNum() for a in mol.GetAtoms()}<=ALLOWED:
            skips['element']+=1;continue
        eligible+=1;xyz=mol.GetConformer().GetPositions()
        identity=Chem.MolToSmiles(mol,isomericSmiles=False)
        name=mol.GetProp('_Name') if mol.HasProp('_Name') else str(idx)
        for atom in mol.GetAtoms():
            z=atom.GetAtomicNum()
            if z not in (6,15):continue
            oxy=sorted(a.GetIdx() for a in atom.GetNeighbors() if a.GetAtomicNum()==8 and a.GetDegree()==1)
            if len(oxy)<2:continue
            center=atom.GetIdx();group='C' if z==6 else 'P'
            labels=[mol.GetBondBetweenAtoms(center,o).GetBondTypeAsDouble() for o in oxy]
            lengths=np.array([np.linalg.norm(xyz[center]-xyz[o]) for o in oxy])
            assert np.isfinite(lengths).all()
            for label,length in zip(labels,lengths):
                if label in (1,2):all_lengths[group][int(label)].append(float(length))
            # Exactly one reference double and >=1 single; no aromatic/other labels.
            if labels.count(2)!=1 or any(c not in (1,2) for c in labels):
                skips[f'{group}_motif_not_one_double_rest_single']+=1;continue
            d=labels.index(2);minimum=lengths.min();tied=np.flatnonzero(lengths<=minimum+EPS)
            hit=bool(d in tied)
            score=1/len(tied) if hit else 0.
            single=lengths[np.array(labels)==1]
            row={'molecule_index':idx,'name':name,'identity':identity,'center':center,
                 'group':group,'n_oxygen':len(oxy),'oxygen_indices':oxy,
                 'reference_labels':list(map(int,labels)),'lengths':lengths.tolist(),
                 'rank_credit':score,'strict_unique_shortest':int(hit and len(tied)==1),
                 'random_chance':1/len(oxy),'credit_minus_chance':score-1/len(oxy),
                 'n_tied_minimum':len(tied),'double_length':float(lengths[d]),
                 'single_mean_minus_double':float(single.mean()-lengths[d]),
                 'single_min_minus_double':float(single.min()-lengths[d])}
            rows.append(row)
    return rows,eligible,dict(skips),all_lengths


def selftest():
    for values,double,want in (([1.,2.],0,1.),([1.,2.],1,0.),([1.,1.],0,.5),
                                ([1.,1.,1.],1,1/3),([1.,2.,2.],0,1.)):
        values=np.array(values);ties=np.flatnonzero(values<=values.min()+EPS)
        got=1/len(ties) if double in ties else 0.
        assert got==want
    rows=[{'molecule_index':0,'rank_credit':1.},{'molecule_index':0,'rank_credit':1.}]
    assert bootstrap(rows,'molecule_index','rank_credit')['ci95']==[1.,1.]


def provenance():
    files=[PILOT/'pilot_v2/decoder_best.pt',PILOT/'pilot_v2/training.json',PILOT/'probe.py',
           PILOT/'prepare.json',PILOT/'source_subset.pt',PILOT/'features.pt',PILOT/'joint_seeds_train.py']
    ck=torch.load(PILOT/'pilot_v2/decoder_best.pt',map_location='cpu',weights_only=False)
    subset=torch.load(PILOT/'source_subset.pt',map_location='cpu',weights_only=False)
    hist=json.loads((PILOT/'pilot_v2/training.json').read_text())
    assert ck['epoch']==min(hist['history'],key=lambda r:r['allmask_val_ce'])['epoch']==10
    head_args={}
    for seed in (20261002,20261003,20261004):
        path=PILOT/f'joint_v5/seed{seed}/best.pt';files.append(path)
        h=torch.load(path,map_location='cpu',weights_only=False)
        assert Path(h['args']['warm_start']).name=='decoder_best.pt'
        head_args[str(seed)]={'epoch':h['epoch'],'args':h['args']}
    return {'runtime_rdkit':rdBase.rdkitVersion,'numpy':np.__version__,'torch':torch.__version__,
            'historical_rdkit_version':'not recorded in original checkpoint; current runtime is not proof of historical version',
            'pilot':{'epoch':ck['epoch'],'args':ck['args'],'selection':hist['selection'],
                     'counts':{k:len(v) for k,v in subset.items()},'historical_prepare':json.loads((PILOT/'prepare.json').read_text())},
            'heads':head_args,'sha256':{str(p.relative_to(ROOT)):sha(p) for p in files}}


def main():
    RDLogger.DisableLog('rdApp.*');torch.set_num_threads(2);selftest()
    paths=[Path(__file__),*INPUTS.values(),ROOT/'results/terminal_oxy_cohend.json']
    config={'status':'post hoc coordinate-only diagnostic; no training or benchmark rescore',
            'selection':'same molecule eligibility as pooled audit; C/P with >=2 heavy-degree1 O, exactly one double and other bonds single',
            'rank':'two-O double shorter; >=3-O sole reference double shortest',
            'tie':'within 1e-6 angstrom of minimum: 1/k credit if double among k tied minima, zero otherwise',
            'chance':'1/n_oxygen per group', 'population_weighting':'groups; cluster bootstrap resamples molecules, ratio estimator',
            'bootstrap':{'reps':REPS,'seed':20261102,'primary':'SDF molecule record','sensitivity':'canonical non-isomeric identity'},
            'no_claims':['chemical equivalence','information-theoretic non-identifiability','whole-model ability'],
            'sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths}}
    p=HERE/'PLAN.json'
    if p.exists():assert json.loads(p.read_text())['spec']==config
    else:save(p,{'recorded_utc':datetime.now(timezone.utc).isoformat(),'spec':config})
    original=json.loads((ROOT/'results/terminal_oxy_cohend.json').read_text());result={};all_rows={}
    for domain,path in INPUTS.items():
        rows,n,skips,lengths=collect(path)
        assert n==original[domain]['n_molecules'],(domain,n)
        for symbol,key in (('C','carboxyl_CO'),('P','phosphate_PO')):
            expected=original[domain]['groups'][key]
            for label,side in ((1,'single'),(2,'double')):
                assert len(lengths[symbol][label])==expected[f'{side}_n']
                assert abs(np.mean(lengths[symbol][label])-expected[f'{side}_mean'])<1e-10
        groups={}
        for symbol in ('C','P'):
            for size in ('all','2','3plus'):
                rr=[r for r in rows if r['group']==symbol and (size=='all' or (r['n_oxygen']==2 if size=='2' else r['n_oxygen']>=3))]
                if not rr:continue
                groups[f'{symbol}_{size}']={'n_groups':len(rr),'n_molecules':len({r['molecule_index'] for r in rr}),
                    'n_identities':len({r['identity'] for r in rr}),'rank_credit':float(np.mean([r['rank_credit'] for r in rr])),
                    'strict_unique_shortest_rate':float(np.mean([r['strict_unique_shortest'] for r in rr])),
                    'random_chance':float(np.mean([r['random_chance'] for r in rr])),
                    'n_tied_minimum':sum(r['n_tied_minimum']>1 for r in rr),
                    'oxygen_counts':dict(collections.Counter(r['n_oxygen'] for r in rr)),
                    'mean_single_minus_double_angstrom':float(np.mean([r['single_mean_minus_double'] for r in rr])),
                    'molecule_cluster':bootstrap(rr,'molecule_index','rank_credit'),
                    'identity_cluster_sensitivity':bootstrap(rr,'identity','rank_credit'),
                    'paired_minus_random':{'mean':float(np.mean([r['credit_minus_chance'] for r in rr])),
                                           **bootstrap(rr,'molecule_index','credit_minus_chance')}}
        result[domain]={'eligible_molecules':n,'skips':skips,'groups':groups};all_rows[domain]=rows
    for rel,want in config['sha256'].items():assert sha(ROOT/rel)==want
    save(HERE/'ROWS.json',all_rows);save(HERE/'SUMMARY.json',{'plan_sha256':sha(p),'domains':result,'rdkit':rdBase.rdkitVersion})
    save(HERE/'PROVENANCE.json',provenance())
    lines=['# Within-group terminal oxygen length ranks','','Post hoc coordinate-only analysis. No chemical-equivalence or model non-identifiability inference.','',
           '| Domain | Centre/O count | Groups / molecules | Double-shortest credit | Molecule-cluster95% CI | Random chance |',
           '|---|---|---:|---:|---:|---:|']
    for domain,entry in result.items():
        for group,c in entry['groups'].items():
            lo,hi=c['molecule_cluster']['ci95']
            lines.append(f"| {domain} | {group} | {c['n_groups']} / {c['n_molecules']} | {100*c['rank_credit']:.2f}% | [{100*lo:.2f}, {100*hi:.2f}]% | {100*c['random_chance']:.2f}% |")
    lines+=['','Exactly one reference double and all other terminal-O bonds single. Ties within1e-6A receive1/k credit.',
            'Pooled old bond counts and length means reproduce exactly. Full per-group rows, identity-cluster sensitivity and paired-minus-random intervals are retained.',
            'The comparison tests a length-only within-motif rule, not recoverability from all Z/R/Q features or chemical resonance equivalence.',
            f'Current analysis runtime RDKit: {rdBase.rdkitVersion}. Historical checkpoint files do not themselves record an RDKit version.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
