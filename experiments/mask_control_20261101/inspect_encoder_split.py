"""Read-only provenance audit of the original encoder cache, from this workspace.

Only split metadata are extracted from the old cache; no command or source module
is executed from the old project, no 4-GB cache is copied, and no prediction runs.
"""
import collections
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import torch
from rdkit import Chem, RDLogger

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'experiments/prob_mask_20261002'
sys.path.insert(0,str(OLD))
from evaluation_common import reference_population

CACHE=Path(r'E:\zxcproject\bondnet\data\geom_drugs_all_random1_re10_c30_heavy_fixed.pt')
SOURCE=ROOT/'data/geom_drugs_all_random1_rel10.sdf'


def write(name,obj):
    (HERE/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')


def log(s): print(time.strftime('%H:%M:%S'),s,flush=True)


def main():
    torch.set_num_threads(1); RDLogger.DisableLog('rdApp.*')
    if not CACHE.is_file(): raise FileNotFoundError(CACHE)
    log('Memory-mapping original encoder cache, read only')
    payload=torch.load(CACHE,map_location='cpu',weights_only=False,mmap=True)
    samples=payload['samples']; metadata=payload['metadata']
    checkpoint=torch.load(ROOT/'checkpoints/revision_v3_one_stage_heavy_seed42/best_e2e.pt',
                          map_location='cpu',weights_only=False)
    args=checkpoint['args']
    fixed=any(s.get('split') is not None for s in samples)
    folds=collections.defaultdict(set)
    if fixed and not args.get('legacy_resplit',False) and args.get('train_groups') is None:
        for s in samples:
            folds[int(s['geom_mol_idx'])].add(s['split'])
        rule='fixed sample split labels, matching current CachedBondNetDataset.split'
    else:
        groups=sorted({int(s['geom_mol_idx']) for s in samples})
        random.Random(args['seed']).shuffle(groups)
        nval=max(1,int(len(groups)*args['val_split']))
        for k in groups[:nval]: folds[k].add('val')
        for k in groups[nval:]: folds[k].add('train')
        rule='checkpoint group shuffle by geom_mol_idx'
    if any(len(v)!=1 for v in folds.values()): raise ValueError('A GEOM group spans cache folds')
    folds={k:next(iter(v)) for k,v in folds.items()}
    inventory={'cache_path_read_only':str(CACHE),'cache_bytes':CACHE.stat().st_size,
        'cache_mtime_ns':CACHE.stat().st_mtime_ns,'sample_count':len(samples),
        'metadata':metadata,'rule':rule,'group_fold_counts':dict(collections.Counter(folds.values())),
        'limitations':'The checkpoint does not store a training-cache hash. Membership is conditional on this original cache retaining its training-time content.'}
    write('ENCODER_CACHE_METADATA.json',inventory)
    del samples,payload,checkpoint;gc.collect()
    log(f'Extracted {len(folds)} group folds: {inventory["group_fold_counts"]}')
    conditions={'pose_clean':'pdbbind_sigma0p00','ccd_ideal':'ccd_sigma0p00'}
    keys=collections.defaultdict(list); results={}
    for condition,folder in conditions.items():
        refs,_=reference_population(ROOT/'work/paper2_noise_ccd_probe'/folder/'reference_for_scoring.sdf')
        results[condition]={'n':len(refs),'matches':collections.defaultdict(list)}
        for name,mol in refs.items():
            keys[Chem.MolToSmiles(mol,isomericSmiles=False)].append((condition,name))
    props={};field=None;records=parsed=0;failures=0
    log('Streaming local GEOM SMILES properties; no coordinates/model inference')
    with SOURCE.open(encoding='utf-8',errors='replace') as f:
        for line in f:
            if line.startswith('>'):
                field='geom_mol_idx' if '<geom_mol_idx>' in line else 'geom_smiles' if '<geom_smiles>' in line else None
                continue
            if field and line.strip():
                props[field]=line.strip();field=None
            if line.startswith('$$$$'):
                records+=1
                idx=int(props.get('geom_mol_idx',-1));smi=props.get('geom_smiles')
                if idx in folds and smi:
                    mol=Chem.MolFromSmiles(smi)
                    if mol is None: failures+=1
                    else:
                        parsed+=1;key=Chem.MolToSmiles(Chem.RemoveHs(mol),isomericSmiles=False)
                        for condition,name in keys.get(key,[]):
                            results[condition]['matches'][name].append({'geom_mol_idx':idx,'split':folds[idx]})
                props={};field=None
                if records%25000==0: log(f'{records} source records; {parsed} cached identities parsed')
    for condition,r in results.items():
        r['matches']=dict(r['matches'])
        r['train_hit_names']=sorted(n for n,ms in r['matches'].items() if any(m['split']=='train' for m in ms))
        r['val_hit_names']=sorted(n for n,ms in r['matches'].items() if any(m['split']=='val' for m in ms))
        r['test_hit_names']=sorted(n for n,ms in r['matches'].items() if any(m['split']=='test' for m in ms))
        log(f'{condition}: train={len(r["train_hit_names"])} val={len(r["val_hit_names"])} test={len(r["test_hit_names"])}')
    write('ENCODER_SPLIT_IDENTITY_AUDIT.json',{'scope':'read-only development provenance audit, no predictions or locked evaluation',
        'reference_key':'RDKit non-isomeric canonical heavy SMILES', 'source_records':records,
        'parsed_cached_identities':parsed,'smiles_parse_failures':failures,
        'cache_metadata_report':'ENCODER_CACHE_METADATA.json','conditions':results,
        'limitations':inventory['limitations']})
    log('ENCODER SPLIT AUDIT COMPLETE')


if __name__=='__main__': main()
