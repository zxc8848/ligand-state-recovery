"""Post-outcome, same-B/same-budget atom-state scoring ablation."""
import argparse
import collections
from datetime import datetime, timezone
import gzip
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
CHARGE = ROOT/'experiments/charged_benchmark_20261102'
DEV = ROOT/'experiments/atom_state_candidates_20261102'
OLD = ROOT/'experiments/atom_state_confirm_20261102'
sys.path.insert(0, str(CHARGE))
sys.path.insert(0, str(DEV))
import study as S
import joint_candidates as C
from verify_outputs import build

ARMS = ('neural_catalogue', 'catalogue_only')
REPS = 5000


def read_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def plan():
    # Reuse input hashes from the completed holdout, without changing that protocol.
    original = json.loads((OLD/'PROTOCOL.json').read_text())
    hashes = dict(original['spec']['sha256'])
    paths = [Path(__file__), HERE/'README.md', OLD/'PROTOCOL.json', OLD/'SUMMARY.json',
             OLD/'features.pt', OLD/'FEATURES.json', CHARGE/'study.py', CHARGE/'verify_outputs.py']
    paths += [OLD/f'evaluations/seed{s}/predictions.json.gz' for s in S.SEEDS]
    for p in paths:
        hashes[str(p.relative_to(ROOT))] = S.sha(p)
    for rel, want in original['spec']['sha256'].items():
        assert S.sha(ROOT/rel) == want, rel
    spec = {'status': 'post-outcome reviewer diagnostic; previously evaluated 280-identity cohort',
            'n': 280, 'heads': list(S.SEEDS), 'arms': list(ARMS),
            'B': 'saved full-system selected B fixed separately for each molecule and head',
            'arm_difference': 'neural atom logits + catalogue versus catalogue logits only',
            'shared': ['B', 'Q', 'catalogue', 'greedy conditional single solver', 'k-best solver',
                       'builder', '32 raw attempts', '8 unique returned outputs', 'return order'],
            'endpoints': ['single_native', 'coverage1', 'coverage2', 'coverage4', 'coverage8',
                          'q/h exact and atom metrics on single', 'constructibility', 'returned counts'],
            'statistics': {'bootstrap': 'paired connectivity-skeleton, molecule-weighted ratio',
                           'reps': REPS, 'seed': 20261102, 'heads': 'average before resampling; SD separate',
                           'intervals': 'pointwise descriptive', 'hypothesis_tests': 0},
            'limitations': ['ideal CCD not real poses', 'B chosen by full model',
                            'not an independent external atom-state predictor',
                            'known GEOM skeleton overlaps retained; see source-scope correction',
                            'previously observed outcomes; not prospective confirmation'],
            'sha256': hashes}
    path = HERE/'MATCHED_PLAN.json'
    if path.exists():
        assert json.loads(path.read_text())['spec'] == spec, 'Plan drift'
    else:
        assert not list(HERE.glob('matched/seed*/DONE.json'))
        S.save(path, {'recorded_utc': datetime.now(timezone.utc).isoformat(), 'spec': spec})
    return S.sha(path)


def check():
    data = json.loads((HERE/'MATCHED_PLAN.json').read_text())
    for rel, want in data['spec']['sha256'].items():
        assert S.sha(ROOT/rel) == want, rel
    return S.sha(HERE/'MATCHED_PLAN.json')


def enumerate_arm(row, labels, logits, baseline):
    raw = C.proposals(logits.cpu().numpy(), row['Q'], baseline['atom_states'], 'plain_kbest')
    returned = []; seen = set()
    for i, item in enumerate(raw):
        states = item.pop('states')
        assert sum(s//5-2 for s in states) == row['Q']
        smi, good = S.J.build_species(row, labels, torch.tensor(states, dtype=torch.long))
        item.update(atom_states=states, smiles=smi, constructible=bool(good))
        if good and smi not in seen and len(returned) < 8:
            seen.add(smi); returned.append(i)
    return {'single': baseline, 'raw': raw, 'returned_indices': returned}


@torch.no_grad()
def cell(seed, rows, catalogue, device, digest):
    folder = HERE/f'matched/seed{seed}'
    path = folder/'predictions.json.gz'; done = folder/'DONE.json'
    if done.exists():
        receipt = json.loads(done.read_text())
        assert receipt['plan_sha256'] == digest and receipt['sha256'] == S.sha(path)
        print('verified completed', seed, flush=True); return
    originals = {r['name']: r for r in read_gz(OLD/f'evaluations/seed{seed}/predictions.json.gz')['records']}
    net, _ = S.T.load_joint(S.checkpoint(seed), device)
    output = []; start = time.perf_counter()
    for index, source in enumerate(rows):
        saved = originals[source['id']]
        row = S.blind(source)
        labels = torch.tensor(saved['labels'], dtype=torch.long)
        batch = S.J.pack([row], device); bd = labels.to(device)
        _, neural = net(batch, bd, torch.full_like(batch['atarget'], S.J.AMASK))
        prior = S.J.catalogue_logits(batch, bd, catalogue)
        logits = {'neural_catalogue': neural+prior, 'catalogue_only': prior}
        arms = {}
        for arm in ARMS:
            single = S.assign_and_select(row, [bd], [0.], logits[arm], True)
            # Preserve original full-arm baseline, exactly matching original budget.
            if arm == ARMS[0]:
                full_single_replay = single['atom_states'] == saved['baseline']['atom_states']
                single = {k: saved['baseline'][k] for k in ('atom_states','smiles','constructible','qsum','score','labels')}
            arms[arm] = enumerate_arm(row, labels, logits[arm], single)
        # Targets attached only after all proposal and return decisions.
        for arm in ARMS:
            for item in [arms[arm]['single'], *arms[arm]['raw']]:
                item['native'] = bool(item['constructible'] and item['smiles'] == source['smiles'])
        old_raw = saved['methods']['plain_kbest']
        current = arms[ARMS[0]]
        replay = {'single': full_single_replay,
                  'raw_states': [x['atom_states'] for x in current['raw']] == [x['atom_states'] for x in old_raw['raw']],
                  'raw_identities': [x['smiles'] for x in current['raw']] == [x['smiles'] for x in old_raw['raw']],
                  'returned_indices': current['returned_indices'] == old_raw['returned_indices']}
        output.append({'name': source['id'], 'bucket': source['bucket'], 'Q': source['Q'],
                       'target_smiles': source['smiles'], 'target_q': source['q'].tolist(),
                       'target_h': source['h'].tolist(), 'labels': labels.tolist(),
                       'full_replay': replay, 'arms': arms})
        if (index+1) % 25 == 0:
            print('matched head', seed, index+1, '/', len(rows), f'{time.perf_counter()-start:.1f}s', flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        json.dump({'plan_sha256': digest, 'records': output}, f, separators=(',', ':'), allow_nan=False)
    S.save(done, {'plan_sha256': digest, 'sha256': S.sha(path), 'n': len(output),
                  'seconds': time.perf_counter()-start, 'device': device})


def value(record, arm, metric):
    e = record['arms'][arm]
    if metric == 'single_native': return int(e['single']['native'])
    if metric.startswith('coverage'):
        return int(any(e['raw'][i]['native'] for i in e['returned_indices'][:int(metric[8:])]))
    if metric == 'single_constructible': return int(e['single']['constructible'])
    if metric == 'returned': return len(e['returned_indices'])
    states = e['single']['atom_states']
    if metric == 'q_exact': return int([s//5-2 for s in states] == record['target_q'])
    if metric == 'h_exact': return int([s%5 for s in states] == record['target_h'])
    raise ValueError(metric)


def estimate(matrix, skeletons):
    x = np.asarray(matrix, dtype=float); assert x.shape == (3, len(skeletons))
    keys = sorted(set(skeletons)); mapping = {k:i for i,k in enumerate(keys)}
    sums = np.zeros(len(keys)); counts = np.zeros(len(keys))
    for v, sk in zip(x.mean(0), skeletons):
        sums[mapping[sk]] += v; counts[mapping[sk]] += 1
    rng = np.random.default_rng(20261102)
    idx = rng.integers(0, len(keys), (REPS, len(keys)))
    boot = sums[idx].sum(1)/counts[idx].sum(1)
    return {'mean': float(x.mean()), 'head_sd': float(x.mean(1).std(ddof=1)),
            'per_head': x.mean(1).tolist(), 'cluster_ci95': np.quantile(boot,[.025,.975]).tolist()}


def atom_metrics(records, arm):
    true_q = np.array([q for r in records for q in r['target_q']])
    true_h = np.array([h for r in records for h in r['target_h']])
    pred = np.array([s for r in records for s in r['arms'][arm]['single']['atom_states']])
    q = pred//5-2; h = pred%5
    cm = np.zeros((5,5), dtype=int)
    np.add.at(cm, (true_q+2, q+2), 1)
    f1 = [2*cm[v+2,v+2]/max(1, cm[v+2].sum()+cm[:,v+2].sum()) for v in sorted(set(true_q))]
    return {'q_macro_f1_truth_present_classes': float(np.mean(f1)),
            'q_atom_accuracy': float(np.mean(q == true_q)), 'h_atom_accuracy': float(np.mean(h == true_h)),
            'q_confusion_matrix': cm.tolist(), 'n_atoms': len(pred)}


def verify_and_report(digest, rows):
    source_rows = {r['id']:r for r in torch.load(DEV/'held_back.pt', weights_only=False, map_location='cpu')}
    byname = {r['id']:r for r in rows}
    assert len(source_rows) == len(byname) == 280 and set(source_rows) == set(byname)
    data = {}; rebuilt = 0; mismatches = []; raw_counts = collections.Counter()
    for seed in S.SEEDS:
        folder = HERE/f'matched/seed{seed}'
        receipt = json.loads((folder/'DONE.json').read_text())
        assert receipt['plan_sha256'] == digest and receipt['sha256'] == S.sha(folder/'predictions.json.gz')
        payload = read_gz(folder/'predictions.json.gz'); assert payload['plan_sha256'] == digest
        records = {r['name']:r for r in payload['records']}; assert len(records) == 280 and set(records) == set(byname)
        data[seed] = records
        for name, r in records.items():
            feat = byname[name]; ref = source_rows[name]
            assert r['target_smiles'] == ref['smiles'] and r['target_q'] == ref['q'].tolist() and r['target_h'] == ref['h'].tolist()
            if not all(r['full_replay'].values()): mismatches.append({'seed':seed, 'name':name, **r['full_replay']})
            for arm in ARMS:
                e = r['arms'][arm]; assert len(e['raw']) == 32 and e['raw'][0]['atom_states'] == e['single']['atom_states']
                accepted = []; seen = set(); raw_counts[arm] += len(e['raw'])
                for i,a in enumerate(e['raw']):
                    states = a['atom_states']
                    assert len(states) == len(feat['z']) and all(0 <= s < 25 for s in states)
                    assert sum(s//5-2 for s in states) == r['Q']
                    smi, good = build(feat['z'].tolist(), feat['pairs'].tolist(), r['labels'], states)
                    assert (smi, bool(good)) == (a['smiles'], a['constructible'])
                    assert a['native'] == bool(good and smi == ref['smiles'])
                    if i == 0:
                        assert (smi, bool(good), a['native']) == (e['single']['smiles'],e['single']['constructible'],e['single']['native'])
                    if good and smi not in seen and len(accepted) < 8:
                        seen.add(smi); accepted.append(i)
                    rebuilt += 1
                assert accepted == e['returned_indices']
                assert all(e['raw'][i]['score'] <= e['raw'][i-1]['score']+1e-8 for i in range(2,32))
    out = {'scope':'post-outcome same-B, same-budget diagnostic', 'plan_sha256':digest,
           'cells':{}, 'verification':{'independent_raw_builds':rebuilt,'raw_attempt_counts':dict(raw_counts),
                                     'full_replay_mismatches':mismatches, 'inputs_unchanged':True}}
    metrics = ('single_native','coverage1','coverage2','coverage4','coverage8','q_exact','h_exact','single_constructible','returned')
    for scope in ('pooled', *S.GROUPS):
        names = sorted(n for n,r in source_rows.items() if scope == 'pooled' or r['bucket'] == scope)
        sk = [source_rows[n]['skeleton'] for n in names]
        cell = {'n':len(names), 'n_skeletons':len(set(sk)), 'arms':{}, 'neural_minus_catalogue':{}}
        matrices = {}
        for arm in ARMS:
            matrices[arm] = {m:np.array([[value(data[s][n],arm,m) for n in names] for s in S.SEEDS]) for m in metrics}
            cell['arms'][arm] = {m:estimate(x,sk) for m,x in matrices[arm].items()}
            cell['arms'][arm]['atom_single_per_head'] = [atom_metrics([data[s][n] for n in names],arm) for s in S.SEEDS]
        for m in metrics:
            cell['neural_minus_catalogue'][m] = estimate(matrices[ARMS[0]][m]-matrices[ARMS[1]][m],sk)
        out['cells'][scope] = cell
    check(); S.save(HERE/'MATCHED_SUMMARY.json',out)
    lines = ['# Same-B / same-budget catalogue control', '',
             'Post-outcome diagnostic on 280 previously evaluated CCD ideal identities; fixed full-model B. Not an independent external baseline.', '',
             '| Group | n | Neural single | Catalogue single | Neural cov@8 | Catalogue cov@8 | Paired cov@8 delta (95% CI), pp |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for scope,c in out['cells'].items():
        fmt = lambda arm,m: f"{100*c['arms'][arm][m]['mean']:.2f} ± {100*c['arms'][arm][m]['head_sd']:.2f}"
        d = c['neural_minus_catalogue']['coverage8']; lo,hi = d['cluster_ci95']
        lines.append(f"| {scope} | {c['n']} | {fmt(ARMS[0],'single_native')} | {fmt(ARMS[1],'single_native')} | {fmt(ARMS[0],'coverage8')} | {fmt(ARMS[1],'coverage8')} | {100*d['mean']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines += ['', 'Percentages: mean ± sample SD across three fixed heads. CIs average heads then resample connectivity-skeleton clusters; pointwise, no p-values.',
              f"Independent raw builds: {rebuilt}; full-arm replay mismatches: {len(mismatches)}.",
              'Single-answer q/h exact and atom macro-F1 are in MATCHED_SUMMARY.json; no reference state enters proposal generation.',
              'The selected B is inherited from the neural full model, so this isolates atom-state scoring conditional on that B, not end-to-end neural superiority.']
    (HERE/'MATCHED_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('action',choices=['run','verify']); ap.add_argument('--device',default='cpu')
    args = ap.parse_args(); torch.set_num_threads(4); RDLogger.DisableLog('rdApp.*')
    if args.action == 'run':
        C.selftest(); digest = plan()
    else: digest = check()
    rows = torch.load(OLD/'features.pt', weights_only=False, map_location='cpu')
    if args.action == 'run':
        cat = S.J.make_catalogue(torch.load(S.SOURCES, weights_only=False, map_location='cpu')['train'])
        for seed in S.SEEDS: cell(seed,rows,cat,args.device,digest)
    verify_and_report(digest,rows)


if __name__ == '__main__': main()
