"""Independent held-back output, chronology and primary-interval checks."""
from datetime import datetime
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import torch
from rdkit import RDLogger

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEV = ROOT/'experiments/atom_state_candidates_20261102'
sys.path.insert(0,str(DEV))
import run as D

S = D.S


def main():
    RDLogger.DisableLog('rdApp.*'); D.check_inputs()
    protocol = json.loads((HERE/'PROTOCOL.json').read_text())
    phash = S.sha(HERE/'PROTOCOL.json')
    for rel,want in protocol['spec']['sha256'].items(): assert S.sha(ROOT/rel) == want,rel
    started = json.loads((HERE/'STARTED.json').read_text())
    assert started['protocol_sha256'] == phash
    assert datetime.fromisoformat(protocol['frozen_utc']) < datetime.fromisoformat(started['first_forward_started_utc'])
    preflight = json.loads((HERE/'PREFLIGHT.json').read_text())
    assert not preflight['mismatches'] and preflight['checked_cases'] == 36
    assert preflight['held_back_forward_passes'] == 0
    assert datetime.fromisoformat(preflight['created_utc']) < datetime.fromisoformat(protocol['frozen_utc'])
    summary = json.loads((HERE/'SUMMARY.json').read_text())
    assert summary['protocol_sha256'] == phash
    sources = {r['id']:r for r in torch.load(DEV/'held_back.pt',weights_only=False,map_location='cpu')}
    feats = {r['id']:r for r in torch.load(HERE/'features.pt',weights_only=False,map_location='cpu')}
    dev = torch.load(DEV/'development.pt',weights_only=False,map_location='cpu')
    dev_names = {r['id'] for r in dev}
    dev_skeletons = {r['skeleton'] for r in dev}
    assert len(sources) == 280 and not set(sources)&dev_names
    assert not {r['skeleton'] for r in sources.values()} & dev_skeletons
    data = {}; rebuilt = 0; states_checked = 0
    for seed in S.SEEDS:
        folder = HERE/f'evaluations/seed{seed}'
        done = json.loads((folder/'DONE.json').read_text())
        assert done['protocol_sha256'] == phash and done['sha256'] == S.sha(folder/'predictions.json.gz')
        with gzip.open(folder/'predictions.json.gz','rt') as f: payload = json.load(f)
        assert payload['protocol_sha256'] == phash and len(payload['records']) == 280
        records = {r['name']:r for r in payload['records']}
        assert set(records) == set(sources)
        data[seed] = records
        for name,r in records.items():
            source,feat = sources[name],feats[name]
            assert r['Q'] == source['Q'] and r['bucket'] == source['bucket']
            assert r['target_smiles'] == source['smiles']
            assert r['target_q'] == source['q'].tolist() and r['target_h'] == source['h'].tolist()
            assert len(r['labels']) == len(feat['pairs']) and all(0<=c<=4 for c in r['labels'])
            assert r['baseline']['labels'] == r['labels']
            assert r['baseline_replay_equal']
            assert set(r['methods']) == {'plain_kbest'}
            e = r['methods']['plain_kbest']; raw = e['raw']
            assert 1 <= len(raw) <= 32 and raw[0]['atom_states'] == r['baseline']['atom_states']
            assert all(raw[i]['score'] <= raw[i-1]['score']+1e-9 for i in range(2,len(raw)))
            accepted=[]; seen=set()
            for i,a in enumerate(raw):
                assert len(a['atom_states']) == len(feat['z']) and all(0<=s<25 for s in a['atom_states'])
                qsum = sum(s//5-2 for s in a['atom_states']); assert qsum == r['Q']
                smi,good = D.build(feat['z'].tolist(),feat['pairs'].tolist(),r['labels'],a['atom_states'])
                assert (smi,bool(good)) == (a['smiles'],a['constructible'])
                assert a['native'] == bool(good and smi == source['smiles'])
                if i == 0:
                    assert (smi,bool(good)) == (r['baseline']['smiles'],r['baseline']['constructible'])
                    assert a['native'] == r['baseline']['native']
                if good and smi not in seen and len(accepted)<8: seen.add(smi); accepted.append(i)
                rebuilt += 1; states_checked += len(a['atom_states'])
            assert accepted == e['returned_indices']
    # Independent arithmetic for all per-head endpoints and macro estimates.
    for scope,cell in summary['cells'].items():
        names = sorted(n for n,r in sources.items() if scope=='pooled' or r['bucket']==scope)
        assert len(names) == cell['n']
        for si,seed in enumerate(S.SEEDS):
            for key in ('baseline_native','coverage1','coverage2','coverage4','coverage8'):
                values = [int(data[seed][n]['baseline']['native']) if key=='baseline_native'
                          else int(D.hit(data[seed][n],'plain_kbest',int(key[-1]))) for n in names]
                assert sum(values) == cell['counts_per_head'][str(seed)][key]
                assert abs(np.mean(values)-cell['metrics'][key]['per_head'][si]) < 1e-12
    for key,metric in summary['equal_weight_stratum_macro'].items():
        per_head = np.mean([summary['cells'][g]['metrics'][key]['per_head'] for g in S.GROUPS],axis=0)
        assert np.allclose(per_head,metric['per_head'])
    # Primary cluster CI via explicit bootstrap member lists, not ratio-vector code.
    names = sorted(n for n,r in sources.items() if r['bucket']=='separated_neutral')
    values = np.array([np.mean([D.hit(data[s][n],'plain_kbest',8) for s in S.SEEDS]) for n in names])
    clusters = sorted({sources[n]['skeleton'] for n in names})
    members = [np.array([i for i,n in enumerate(names) if sources[n]['skeleton']==c]) for c in clusters]
    rng = np.random.default_rng(protocol['spec']['bootstrap']['seed'])
    draws = rng.integers(0,len(clusters),(protocol['spec']['bootstrap']['reps'],len(clusters)))
    means=[]
    for ids in draws:
        means.append(values[np.concatenate([members[i] for i in ids])].mean())
    independent_ci=np.quantile(means,[.025,.975]).tolist()
    assert np.allclose(independent_ci,summary['cells']['separated_neutral']['metrics']['coverage8']['cluster_bootstrap_ci95'])
    assert len(json.loads((ROOT/'experiments/charged_benchmark_20261102/SUMMARY.json').read_text())['tests']) == 30
    for rel,want in protocol['spec']['sha256'].items(): assert S.sha(ROOT/rel) == want,rel
    result={'protocol_sha256':phash,'protocol_frozen_before_first_held_back_forward':True,
            'preflight_development_cases':36,'population':280,'unique_identities':len({r['smiles'] for r in sources.values()}),
            'skeleton_clusters':len({r['skeleton'] for r in sources.values()}),'development_overlap':0,
            'graphs_independently_rebuilt':rebuilt,'atom_states_checked':states_checked,
            'primary_cluster_ci_independently_reproduced':independent_ci,'all_counts_and_macros_verified':True,
            'all_frozen_inputs_unchanged':True,'formal_hypothesis_tests':0,'original_charge_study_tests_unchanged':30,
            'verifier_sha256':S.sha(Path(__file__)),'summary_sha256':S.sha(HERE/'SUMMARY.json')}
    S.save(HERE/'VERIFICATION.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
