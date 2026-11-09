"""Independent group-rank arithmetic and explicit-member cluster interval audit."""
import json
from pathlib import Path
import numpy as np
from analyze import sha,save,ROOT,HERE,INPUTS
from rdkit import Chem,RDLogger


def main():
    RDLogger.DisableLog('rdApp.*')
    plan=json.loads((HERE/'PLAN.json').read_text())
    for rel,want in plan['spec']['sha256'].items():assert sha(ROOT/rel)==want
    summary=json.loads((HERE/'SUMMARY.json').read_text())
    rows=json.loads((HERE/'ROWS.json').read_text())
    checked=0;intervals={}
    for domain,path in INPUTS.items():
        mols={i:m for i,m in enumerate(Chem.SDMolSupplier(str(path),removeHs=False)) if m is not None}
        for r in rows[domain]:
            mol=Chem.RemoveHs(mols[r['molecule_index']],sanitize=True)
            xyz=mol.GetConformer().GetPositions();center=r['center']
            os=sorted(a.GetIdx() for a in mol.GetAtomWithIdx(center).GetNeighbors()
                      if a.GetAtomicNum()==8 and a.GetDegree()==1)
            assert os==r['oxygen_indices']
            distances=[sum((float(xyz[center,j])-float(xyz[o,j]))**2 for j in range(3))**.5 for o in os]
            orders=[int(mol.GetBondBetweenAtoms(center,o).GetBondTypeAsDouble()) for o in os]
            assert orders==r['reference_labels'] and orders.count(2)==1 and all(o in (1,2) for o in orders)
            assert np.allclose(distances,r['lengths'],rtol=0,atol=1e-12)
            shortest=min(distances);winner=[j for j,d in enumerate(distances) if abs(d-shortest)<=1e-6]
            credit=1/len(winner) if orders.index(2) in winner else 0
            assert abs(credit-r['rank_credit'])<1e-12
            checked+=1
        for group in ('C_2','P_2','P_3plus'):
            symbol,size=group.split('_')
            rr=[r for r in rows[domain] if r['group']==symbol and (r['n_oxygen']==2 if size=='2' else r['n_oxygen']>=3)]
            ids=sorted({r['molecule_index'] for r in rr});members=[[r['rank_credit'] for r in rr if r['molecule_index']==i] for i in ids]
            rng=np.random.default_rng(20261102);draws=rng.integers(0,len(ids),(5000,len(ids)))
            values=[np.mean([v for i in sample for v in members[i]]) for sample in draws]
            interval=np.quantile(values,[.025,.975]).tolist()
            assert np.allclose(interval,summary['domains'][domain]['groups'][group]['molecule_cluster']['ci95'])
            intervals[domain+'/'+group]=interval
    prov=json.loads((HERE/'PROVENANCE.json').read_text())
    for rel,want in prov['sha256'].items():assert sha(ROOT/rel)==want
    receipt={'independent_group_ranks':checked,'explicit_member_cluster_intervals':intervals,
             'input_and_provenance_hashes_unchanged':True,'plan_sha256':sha(HERE/'PLAN.json'),
             'summary_sha256':sha(HERE/'SUMMARY.json'),'verifier_sha256':sha(Path(__file__))}
    save(HERE/'VERIFICATION.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
