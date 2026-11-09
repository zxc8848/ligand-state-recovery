"""Prospective frozen held-back estimation; no training or strategy tuning."""
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
DEV = ROOT/'experiments/atom_state_candidates_20261102'
sys.path.insert(0, str(DEV))
import run as D

S, C = D.S, D.C
REPS = 5000


@torch.no_grad()
def infer(net, source, catalogue, device):
    start = time.perf_counter()
    baseline = S.infer(net, source, catalogue, device)['arms']['neural_constrained']
    row = S.blind(source)
    labels = torch.tensor(baseline['labels'], dtype=torch.long)
    batch = S.J.pack([row], device); bd = labels.to(device)
    _, neural = net(batch, bd, torch.full_like(batch['atarget'], S.J.AMASK))
    logits = neural + S.J.catalogue_logits(batch, bd, catalogue)
    replay = S.assign_and_select(row, [bd], [0.], logits, True)
    pstart = time.perf_counter()
    raw = C.proposals(logits.cpu().numpy(), row['Q'], baseline['atom_states'], 'plain_kbest')
    accepted = []; seen = set()
    for i, item in enumerate(raw):
        states = item.pop('states')
        assert sum(s//5-2 for s in states) == row['Q']
        smi, good = S.J.build_species(row, labels, torch.tensor(states))
        item.update(atom_states=states, smiles=smi, constructible=bool(good))
        if good and smi not in seen and len(accepted) < 8:
            seen.add(smi); accepted.append(i)
    entry = {'raw': raw, 'returned_indices': accepted, 'seconds': time.perf_counter()-pstart}
    # No reference atomic state or identity has entered candidate generation.
    for item in raw:
        item['native'] = bool(item['constructible'] and item['smiles'] == source['smiles'])
    graph = {tuple(p): c for p, c in zip(row['pairs'].tolist(), labels.tolist()) if c}
    return {'name': source['id'], 'bucket': source['bucket'], 'Q': source['Q'],
            'target_smiles': source['smiles'], 'target_q': source['q'].tolist(), 'target_h': source['h'].tolist(),
            'labels': labels.tolist(), 'strict_B': graph == source['truth'], 'baseline': baseline,
            'baseline_replay_equal': replay['atom_states'] == baseline['atom_states'],
            'methods': {'plain_kbest': entry}, 'seconds': time.perf_counter()-start}


def preflight(device):
    D.check_inputs(); C.selftest()
    selected = torch.load(DEV/'features.pt', weights_only=False, map_location='cpu')
    chosen = []
    for g in S.GROUPS:
        chosen.extend(sorted([r for r in selected if r['bucket'] == g], key=lambda r: r['id'])[:3])
    assert len(chosen) == 12
    catalogue = S.J.make_catalogue(torch.load(S.SOURCES, weights_only=False, map_location='cpu')['train'])
    mismatches = []; count = 0
    for seed in S.SEEDS:
        with gzip.open(DEV/f'evaluations/seed{seed}/predictions.json.gz', 'rt') as f:
            old = {r['name']: r for r in json.load(f)['records']}
        net, _ = S.T.load_joint(S.checkpoint(seed), device)
        for row in chosen:
            r = infer(net, row, catalogue, device); want = old[row['id']]
            flags = {k: r['baseline'][k] == want['baseline'][k] for k in ('labels','atom_states','native','constructible')}
            a, b = r['methods']['plain_kbest'], want['methods']['plain_kbest']
            flags['returned_indices'] = a['returned_indices'] == b['returned_indices']
            flags['raw_states'] = [x['atom_states'] for x in a['raw']] == [x['atom_states'] for x in b['raw']]
            flags['raw_identities'] = [x['smiles'] for x in a['raw']] == [x['smiles'] for x in b['raw']]
            if not all(flags.values()): mismatches.append({'seed': seed, 'name': row['id'], 'flags': flags})
            count += 1
        del net
        if str(device).startswith('cuda'): torch.cuda.empty_cache()
    out = {'scope': 'development-only implementation equivalence', 'checked_cases': count,
           'mismatches': mismatches, 'held_back_forward_passes': 0,
           'driver_sha256': S.sha(Path(__file__)), 'development_run_plan_sha256': S.sha(DEV/'RUN_PLAN.json'),
           'created_utc': datetime.now(timezone.utc).isoformat()}
    path = HERE/'PREFLIGHT.json'
    if path.exists(): raise ValueError('Preflight receipt exists: inspect/reuse, do not overwrite')
    S.save(path, out)
    assert not mismatches, mismatches
    print('development equivalence passed:', count, 'cases; held-back forward passes: 0', flush=True)


def freeze():
    D.check_inputs()
    before = json.loads((HERE/'PREFLIGHT.json').read_text())
    assert not before['mismatches'] and before['checked_cases'] == 36
    assert before['driver_sha256'] == S.sha(Path(__file__))
    audit = json.loads((DEV/'DATASET_AUDIT.json').read_text())
    assert not audit['conflicts']
    rows = torch.load(DEV/'held_back.pt', weights_only=False, map_location='cpu')
    population = dict(collections.Counter(r['bucket'] for r in rows))
    assert population == {'cation':100, 'anion':46, 'separated_neutral':34, 'uncharged_neutral':100}
    assert len({r['smiles'] for r in rows}) == len(rows) == 280
    assert len({r['skeleton'] for r in rows}) == 277
    paths = [Path(__file__), HERE/'README.md', HERE/'PREFLIGHT.json', DEV/'RUN_PLAN.json',
             DEV/'DATASET.json', DEV/'DATASET_AUDIT.json', DEV/'ANALYSIS.json',
             DEV/'held_back.pt', DEV/'held_back.sdf', DEV/'run.py', DEV/'joint_candidates.py']
    sha = dict(json.loads((DEV/'RUN_PLAN.json').read_text())['spec']['sha256'])
    sha.update({str(p.relative_to(ROOT)): S.sha(p) for p in paths})
    spec = {'status': 'prospective held-back estimation of a development-selected configuration',
            'population': population, 'n':280, 'seeds': S.SEEDS, 'encoder_fixed': True,
            'algorithm': 'plain_kbest only, identical to development; baseline+top31, max32 attempts, max8 canonical-unique buildable species',
            'Q': 'true total charge embedded and enforced', 'B': 'current model selected B fixed, no reference B',
            'primary': {'scope':'separated_neutral', 'n':34, 'endpoint':'native species coverage@8, mean over three fixed heads',
                        'CI':'95% pointwise connectivity-skeleton cluster bootstrap'},
            'secondary': ['baseline native','coverage@1/2/4','coverage@8-baseline','other groups','pooled','equal-weight four-stratum macro',
                          'returned counts','constructibility','wall time'],
            'bootstrap': {'reps':REPS, 'seed':20261102, 'primary_unit':'connectivity skeleton cluster, molecule-weighted ratio',
                          'sensitivity_unit':'molecule', 'three_heads':'average before resampling, head SD reported separately',
                          'training_uncertainty_included':False, 'intervals':'pointwise, not simultaneous'},
            'hypothesis_tests':0, 'Holm_family_size':0,
            'no_null_test_reason':'baseline inclusion guarantees nondecrease; a positive McNemar result would not establish mechanism',
            'cohort_filter_after_outcomes':False, 'seed_selection':False, 'early_stopping':False,
            'original_artifacts_and_families_modified':False,
            'nonclaims':['single-answer prediction accuracy gain','new learned architecture','calibrated molecular distribution',
                         'unique protonation from missing-H input','real bound-pose validation','physics validity'],
            'sha256':sha}
    spec = json.loads(json.dumps(spec)); path = HERE/'PROTOCOL.json'
    if path.exists(): assert json.loads(path.read_text())['spec'] == spec, 'Frozen protocol drift'
    else:
        assert not list(HERE.glob('evaluations/seed*/DONE.json')) and not (HERE/'features.pt').exists(), 'Held-back outputs existed before freezing'
        S.save(path, {'frozen_utc':datetime.now(timezone.utc).isoformat(), 'spec':spec})
    print('protocol SHA256:', S.sha(path), flush=True)
    return S.sha(path)


def check_inputs():
    for rel, want in json.loads((HERE/'PROTOCOL.json').read_text())['spec']['sha256'].items():
        assert S.sha(ROOT/rel) == want, rel


def features(phash, device):
    path = HERE/'features.pt'; receipt = HERE/'FEATURES.json'
    if path.exists():
        meta = json.loads(receipt.read_text())
        assert meta['protocol_sha256'] == phash and meta['sha256'] == S.sha(path)
        return torch.load(path, weights_only=False, map_location='cpu')
    check_inputs()
    start_path = HERE/'STARTED.json'
    if not start_path.exists():
        S.save(start_path, {'protocol_sha256':phash, 'first_forward_started_utc':datetime.now(timezone.utc).isoformat()})
    else: assert json.loads(start_path.read_text())['protocol_sha256'] == phash
    rows = torch.load(DEV/'held_back.pt', weights_only=False, map_location='cpu')
    encoder, args = S.P.load_backbone(S.ENCODER, device)
    geos = [S.P.geometry(r, 0., np.random.default_rng(11), args['cutoff']) for r in rows]
    out = []
    for i in range(0, len(rows), 8): out.extend(S.J.extract(encoder, geos[i:i+8], device))
    torch.save(out, path); S.save(receipt, {'protocol_sha256':phash, 'sha256':S.sha(path), 'n':len(out), 'cutoff':args['cutoff']})
    del encoder
    if str(device).startswith('cuda'): torch.cuda.empty_cache()
    return out


def cell(seed, rows, catalogue, device, phash):
    folder = HERE/f'evaluations/seed{seed}'; path = folder/'predictions.json.gz'; done = folder/'DONE.json'
    if done.exists():
        receipt = json.loads(done.read_text())
        assert receipt['protocol_sha256'] == phash and receipt['sha256'] == S.sha(path)
        print('verified completed head', seed, flush=True); return
    net, _ = S.T.load_joint(S.checkpoint(seed), device)
    records = []; start = time.perf_counter()
    for i, row in enumerate(rows):
        records.append(infer(net, row, catalogue, device))
        if (i+1)%25 == 0: print('held-back head', seed, i+1, '/', len(rows), flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as f: json.dump({'protocol_sha256':phash,'records':records}, f, separators=(',',':'), allow_nan=False)
    S.save(done, {'protocol_sha256':phash, 'sha256':S.sha(path), 'n':len(records),'seconds':time.perf_counter()-start})
    del net
    if str(device).startswith('cuda'): torch.cuda.empty_cache()


def estimate(matrix, clusters):
    matrix = np.asarray(matrix, dtype=float); mean_molecule = matrix.mean(0)
    n = len(mean_molecule); assert matrix.shape == (3, n) and len(clusters) == n
    rng = np.random.default_rng(20261102); ids = rng.integers(0, n, (REPS, n))
    molecule_draws = mean_molecule[ids].mean(1)
    keys = sorted(set(clusters)); index = {k:i for i,k in enumerate(keys)}
    counts = np.zeros(len(keys)); sums = np.zeros(len(keys))
    for value, cluster in zip(mean_molecule, clusters):
        counts[index[cluster]] += 1; sums[index[cluster]] += value
    rng = np.random.default_rng(20261102); ids = rng.integers(0, len(keys), (REPS, len(keys)))
    draws = sums[ids].sum(1)/counts[ids].sum(1)
    out = {'mean':float(matrix.mean()),'head_sd':float(matrix.mean(1).std(ddof=1)),
           'per_head':matrix.mean(1).tolist(), 'molecule_bootstrap_ci95':np.quantile(molecule_draws,[.025,.975]).tolist(),
           'cluster_bootstrap_ci95':np.quantile(draws,[.025,.975]).tolist(), 'n':n, 'n_skeleton_clusters':len(keys)}
    return out, draws


def estimator_selftest():
    value, draws = estimate(np.ones((3, 3)), ['a','a','b'])
    assert value['mean'] == 1 and np.all(draws == 1)
    zero, draws = estimate(np.zeros((3, 3)), ['a','a','b'])
    assert zero['mean'] == 0 and np.all(draws == 0)


def summarize(phash):
    estimator_selftest(); check_inputs()
    source = {r['id']:r for r in torch.load(DEV/'held_back.pt', weights_only=False, map_location='cpu')}
    feature_map = {r['id']:r for r in torch.load(HERE/'features.pt', weights_only=False, map_location='cpu')}
    dev_ids = {r['id'] for r in torch.load(DEV/'development.pt', weights_only=False, map_location='cpu')}
    data = {}; checked = 0; replay_mismatches = []
    for seed in S.SEEDS:
        folder = HERE/f'evaluations/seed{seed}'
        receipt = json.loads((folder/'DONE.json').read_text())
        assert receipt['protocol_sha256'] == phash and receipt['sha256'] == S.sha(folder/'predictions.json.gz')
        with gzip.open(folder/'predictions.json.gz','rt') as f: saved = json.load(f)
        assert saved['protocol_sha256'] == phash
        assert len(saved['records']) == len({r['name'] for r in saved['records']}) == 280
        data[seed] = {r['name']:r for r in saved['records']}
        assert set(data[seed]) == set(source) and not set(data[seed]) & dev_ids
        for r in saved['records']:
            feat = feature_map[r['name']]; target = source[r['name']]
            assert r['target_smiles'] == target['smiles'] and r['Q'] == target['Q']
            if not r['baseline_replay_equal']: replay_mismatches.append({'seed':seed,'name':r['name']})
            entry = r['methods']['plain_kbest']; expected = []; seen = set()
            assert len(entry['raw']) <= 32 and entry['raw'][0]['atom_states'] == r['baseline']['atom_states']
            for i,a in enumerate(entry['raw']):
                assert sum(s//5-2 for s in a['atom_states']) == r['Q']
                smi,good = D.build(feat['z'].tolist(),feat['pairs'].tolist(),r['labels'],a['atom_states'])
                assert (smi,bool(good)) == (a['smiles'],a['constructible'])
                assert a['native'] == bool(good and smi == target['smiles'])
                if good and smi not in seen and len(expected)<8: seen.add(smi); expected.append(i)
                checked += 1
            assert expected == entry['returned_indices']
            assert not r['baseline']['native'] or D.hit(r,'plain_kbest',1)
    out = {'status':'prospective held-back estimation, selected plain k-best only', 'protocol_sha256':phash,
           'primary':'separated_neutral/coverage8','cells':{}, 'equal_weight_stratum_macro':{},
           'verification':{'independent_graph_rebuilds':checked,'baseline_replay_mismatches':replay_mismatches,
                           'development_prediction_identity_hits':0,'hypothesis_tests':0,'old_families_modified':False}}
    macro_draws = {}; macro_rates = {}
    for scope in ('pooled', *S.GROUPS):
        ns = sorted(n for n,r in source.items() if scope == 'pooled' or r['bucket'] == scope)
        clusters = [source[n]['skeleton'] for n in ns]
        matrices = {'baseline_native':np.array([[int(data[s][n]['baseline']['native']) for n in ns] for s in S.SEEDS])}
        matrices.update({f'coverage{k}':np.array([[int(D.hit(data[s][n],'plain_kbest',k)) for n in ns] for s in S.SEEDS]) for k in (1,2,4,8)})
        matrices['coverage8_minus_baseline'] = matrices['coverage8']-matrices['baseline_native']
        cellout = {'n':len(ns),'n_skeleton_clusters':len(set(clusters)),'metrics':{},'counts_per_head':{}}
        for key,matrix in matrices.items():
            stats,draws = estimate(matrix,clusters); cellout['metrics'][key] = stats
            if scope != 'pooled':
                # Independent deterministic streams per stratum for stratified macro.
                offset = S.GROUPS.index(scope)+1
                rng = np.random.default_rng(20261102+offset)
                uniq = sorted(set(clusters)); lookup = {g:np.array([i for i,c in enumerate(clusters) if c==g]) for g in uniq}
                sums = np.array([matrix.mean(0)[lookup[g]].sum() for g in uniq])
                sizes = np.array([len(lookup[g]) for g in uniq]); ids = rng.integers(0,len(uniq),(REPS,len(uniq)))
                macro_draws.setdefault(key,[]).append(sums[ids].sum(1)/sizes[ids].sum(1))
                macro_rates.setdefault(key,[]).append(matrix.mean(1))
        for s in S.SEEDS:
            rs = [data[s][n] for n in ns]
            entries = [r['methods']['plain_kbest'] for r in rs]
            cellout['counts_per_head'][str(s)] = {'baseline_native':sum(r['baseline']['native'] for r in rs),
                **{f'coverage{k}':sum(D.hit(r,'plain_kbest',k) for r in rs) for k in (1,2,4,8)},
                'mean_returned':statistics.mean(len(e['returned_indices']) for e in entries),
                'any_constructible':sum(bool(e['returned_indices']) for e in entries),
                'baseline_constructible':sum(r['baseline']['constructible'] for r in rs),
                'raw_attempts_min_max':[min(len(e['raw']) for e in entries),max(len(e['raw']) for e in entries)],
                'median_proposal_seconds':statistics.median(e['seconds'] for e in entries),
                'median_full_seconds':statistics.median(r['seconds'] for r in rs)}
        out['cells'][scope] = cellout
    for key,draws in macro_draws.items():
        rates = np.stack(macro_rates[key]).mean(0)
        out['equal_weight_stratum_macro'][key] = {'mean':float(rates.mean()),'head_sd':float(rates.std(ddof=1)),
            'per_head':rates.tolist(),'stratified_cluster_bootstrap_ci95':np.quantile(np.stack(draws).mean(0),[.025,.975]).tolist()}
    check_inputs(); S.save(HERE/'SUMMARY.json',out)
    lines = ['# 联合(q,h)候选：冻结留存集估计', '',
             '方法在开发集选定，留存集首次运行前已冻结协议。不训练，不调整方法。主端点：34个电荷分离中性分子的native species coverage@8。', '',
             '| 组 | n | 单解native | coverage@1 | @2 | @4 | @8 | @8的95%骨架簇bootstrap区间 | @8-单解 pp |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for scope,cellout in out['cells'].items():
        m = cellout['metrics']; fmt = lambda k:f"{100*m[k]['mean']:.2f}±{100*m[k]['head_sd']:.2f}"
        lo,hi = m['coverage8']['cluster_bootstrap_ci95']
        lines.append('| '+ ' | '.join([scope,str(cellout['n']),fmt('baseline_native'),*[fmt(f'coverage{k}') for k in (1,2,4,8)],f'[{100*lo:.2f}, {100*hi:.2f}]',f"{100*m['coverage8_minus_baseline']['mean']:+.2f}"])+' |')
    lines += ['', '百分比均值±三个固定头的样本SD；bootstrap按分子骨架簇重采样并以分子计权，区间条件于这三个头。逐分子敏感性区间、分层macro、候选数及耗时见SUMMARY.json。',
              '没有假设检验、p值或新Holm族（大小0）；未变动既有检验族。所有次要区间为点态、非同时区间。',
              'baseline优先保留保证集合覆盖率不下降；因此增量并非独立的机制证据。coverage不等于自动选对唯一答案，不代表物理有效性或已学到校准的分子分布。',
              '本实验仍为CCD理想坐标与策展原子态，不是真实结合态质子化验证。',
              f'独立重构{checked}个赋值；baseline重放不一致{len(replay_mismatches)}项；开发集预测身份命中0。', '']
    (HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    primary = out['cells']['separated_neutral']['metrics']['coverage8']
    print('Held-back report saved; primary:',json.dumps(primary),flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('action',choices=['preflight','freeze','run','report']); ap.add_argument('--device',default='cuda')
    args = ap.parse_args(); torch.set_num_threads(4); RDLogger.DisableLog('rdApp.*')
    if args.action == 'preflight': preflight(args.device); return
    phash = freeze()
    if args.action == 'freeze': return
    if args.action == 'run':
        check_inputs(); C.selftest(); estimator_selftest(); rows = features(phash,args.device)
        catalogue = S.J.make_catalogue(torch.load(S.SOURCES,weights_only=False,map_location='cpu')['train'])
        for seed in S.SEEDS: cell(seed,rows,catalogue,args.device,phash)
    summarize(phash)


if __name__ == '__main__': main()
