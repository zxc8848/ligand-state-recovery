"""Read-only retained GEOM source-skeleton audit; never filters/rescores a cohort."""
import collections
import hashlib
import json
from pathlib import Path
import sys
import torch
from rdkit import Chem, RDLogger

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/prob_mask_20261002'))
import joint as J


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    RDLogger.DisableLog('rdApp.*')
    src=ROOT/'data/geom_drugs_all_random1_rel10.sdf'
    paths=[ROOT/f'experiments/atom_state_candidates_20261102/{s}.pt' for s in ('development','held_back')]
    keys=collections.defaultdict(list); cohorts={}
    for split,path in zip(('development','held_back'),paths):
        rows=torch.load(path,weights_only=False,map_location='cpu');cohorts[split]=rows
        for r in rows:keys[r['skeleton']].append((split,r['id']))
    input_hashes={str(p.relative_to(ROOT)):sha(p) for p in [src,*paths]}
    matches=collections.defaultdict(list);props={};field=None;records=parsed=0
    with src.open(encoding='utf-8',errors='replace') as f:
        for line in f:
            if line.startswith('>'):
                field='geom_smiles' if '<geom_smiles>' in line else 'geom_mol_idx' if '<geom_mol_idx>' in line else None
                continue
            if field and line.strip():props[field]=line.strip();field=None
            if line.startswith('$$$$'):
                records+=1;mol=Chem.MolFromSmiles(props.get('geom_smiles',''))
                if mol is not None and mol.GetNumAtoms():
                    parsed+=1;heavy=Chem.RemoveHs(mol);skel=J.skeleton(heavy)
                    for split,name in keys.get(skel,[]):
                        matches[(split,name)].append(props.get('geom_mol_idx',str(records-1)))
                props={};field=None
                if records%50000==0:print('GEOM skeleton audit',records,'records;',len(matches),'matched selected records',flush=True)
    result={'scope':'post-outcome read-only source audit; no cohort filtering, model inference or rescoring',
            'encoder_actual_cache':'data/geom_drugs_all_random1_re10_c30_heavy_fixed.pt, per checkpoint args',
            'source_audited':'all retained raw GEOM source records, not only training fold',
            'not_a_historical_cache_hash_guarantee':True,'records':records,'parsed':parsed,
            'input_sha256':input_hashes,'cohorts':{},'auditor_sha256':sha(Path(__file__))}
    for split,rows in cohorts.items():
        hits=[{'name':r['id'],'bucket':r['bucket'],'geom_source_ids':matches[(split,r['id'])]}
              for r in rows if (split,r['id']) in matches]
        result['cohorts'][split]={'n':len(rows),'skeleton_hit_records':len(hits),'matches':hits}
    assert input_hashes=={str(p.relative_to(ROOT)):sha(p) for p in [src,*paths]}
    (OUT/'GEOM_SKELETON_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','cohorts')},indent=2),flush=True)
    print({s:c['skeleton_hit_records'] for s,c in result['cohorts'].items()},flush=True)


if __name__=='__main__':main()
