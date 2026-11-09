"""Independent selected-graph rebuild and population/fingerprint verification."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from rdkit import Chem, RDLogger

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/prob_mask_20261002'))
import joint as J


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def build(z,pairs,labels,states):
    rw=Chem.RWMol()
    for atomic,state in zip(z,states):
        atom=Chem.Atom(int(atomic));atom.SetFormalCharge(state//5-2)
        atom.SetNumExplicitHs(state%5);atom.SetNoImplicit(True);rw.AddAtom(atom)
    kinds={1:Chem.BondType.SINGLE,2:Chem.BondType.DOUBLE,3:Chem.BondType.TRIPLE,4:Chem.BondType.AROMATIC}
    for (i,j),label in zip(pairs,labels):
        if label:
            rw.AddBond(i,j,kinds[label])
            if label==4:
                rw.GetAtomWithIdx(i).SetIsAromatic(True);rw.GetAtomWithIdx(j).SetIsAromatic(True)
    try:
        mol=rw.GetMol();Chem.SanitizeMol(mol)
        good=len(Chem.GetMolFrags(mol))==1 and all(a.GetNumRadicalElectrons()==0 for a in mol.GetAtoms())
        return Chem.MolToSmiles(mol,isomericSmiles=False),good
    except Exception:return None,False


def main():
    RDLogger.DisableLog('rdApp.*')
    protocol=json.loads((HERE/'PROTOCOL.json').read_text())
    phash=sha(HERE/'PROTOCOL.json')
    for rel,want in protocol['spec']['sha256'].items():assert sha(ROOT/rel)==want,rel
    rows=torch.load(HERE/'selected.pt',weights_only=False,map_location='cpu')
    feats={r['id']:r for r in torch.load(HERE/'features.pt',weights_only=False,map_location='cpu')}
    assert len(rows)==len(feats)==270
    identities=[r['smiles'] for r in rows];assert len(set(identities))==270
    byname={r['id']:r for r in rows}
    excluded=json.loads((ROOT/'data/locked_testset_excluded_identities.json').read_text())
    excluded=set(excluded['plain'])|set(excluded['stereo'])
    for path in (ROOT/'data/pdbbind_ccd_pose500.sdf',ROOT/'data/pdbbind_pose_locked500.sdf',ROOT/'work/paper2_noise_ccd_probe/ccd_sigma0p00/reference_for_scoring.sdf'):
        for m in Chem.SDMolSupplier(str(path),removeHs=False):
            assert m is not None
            m=Chem.RemoveHs(m)
            excluded.update((Chem.MolToSmiles(m,isomericSmiles=False),Chem.MolToSmiles(m,isomericSmiles=True)))
    sources=torch.load(ROOT/'experiments/prob_mask_20261002/joint_v1/sources.pt',weights_only=False,map_location='cpu')
    skeletons={r['skeleton'] for split in ('train','val') for r in sources[split]}
    excluded.update(r['smiles'] for split in ('train','val') for r in sources[split])
    for r in rows:
        assert r['smiles'] not in excluded and r['skeleton'] not in skeletons
        m=Chem.MolFromSmiles(r['smiles']);assert m is not None
        assert Chem.MolToSmiles(m,isomericSmiles=False) not in excluded
    n=0;counts=collections.Counter()
    for seed in protocol['spec']['head_seeds']:
        folder=HERE/f'evaluations/seed{seed}'
        done=json.loads((folder/'DONE.json').read_text())
        assert done['protocol_sha256']==phash and done['sha256']==sha(folder/'predictions.json.gz')
        with gzip.open(folder/'predictions.json.gz','rt') as f:records=json.load(f)['records']
        assert {r['name'] for r in records}==set(byname) and len(records)==270
        for r in records:
            source=byname[r['name']];feature=feats[r['name']]
            assert r['target_q']==source['q'].tolist() and r['target_h']==source['h'].tolist()
            for arm,item in r['arms'].items():
                assert len(item['atom_states'])==len(source['z'])
                assert len(item['labels'])==len(feature['pairs'])
                assert all(0<=v<=24 for v in item['atom_states'])
                assert all(0<=v<=4 for v in item['labels'])
                smi,good=build(source['z'].tolist(),feature['pairs'].tolist(),item['labels'],item['atom_states'])
                assert (smi,good)==(item['smiles'],item['constructible'])
                assert item['native']==bool(good and smi==source['smiles'])
                predicted={tuple(p):v for p,v in zip(feature['pairs'].tolist(),item['labels']) if v}
                assert item['strict']==(predicted==source['truth'])
                qsum=sum(v//5-2 for v in item['atom_states'])
                assert item['qsum']==qsum and item['charge_sum_correct']==(qsum==source['Q'])
                if arm!='neural_unconstrained':assert qsum==source['Q']
                counts[arm]+=int(item['native']);n+=1
    summary=json.loads((HERE/'SUMMARY.json').read_text())
    assert len(summary['tests'])==30
    for arm,count in counts.items():
        assert abs(summary['cells']['pooled']['arms'][arm]['mean']['native']-count/810)<1e-12
    spec_before=sha(HERE/'PROTOCOL.json')
    for rel,want in protocol['spec']['sha256'].items():assert sha(ROOT/rel)==want
    assert spec_before==phash
    clusters=collections.Counter(r['skeleton'] for r in rows)
    out={'protocol_sha256':phash,'selected_graphs_independently_rebuilt':n,
         'all_predictions_and_flags_verified':True,'all_source_and_checkpoint_hashes_unchanged':True,
         'known_source_and_old_benchmark_identity_hits':0,'head_train_val_skeleton_hits':0,
         'n':270,'unique_nonisomeric_identities':270,'distinct_connectivity_skeletons':len(clusters),
         'molecules_in_repeated_skeletons':sum(v for v in clusters.values() if v>1),
         'native_success_counts_over_three_heads':dict(counts),'native_tests':30,
         'verifier_sha256':sha(Path(__file__))}
    (HERE/'VERIFICATION.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
