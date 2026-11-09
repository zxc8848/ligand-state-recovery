"""Fixed frozen model; independent development; Q-constrained joint q/h candidates."""
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

import joint_candidates as C

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'experiments/charged_benchmark_20261102'))
import study as S
from verify_outputs import build

METHODS = ('plain_kbest', 'mass_stratified')


def freeze():
    original = json.loads((ROOT/'experiments/charged_benchmark_20261102/PROTOCOL.json').read_text())
    for rel, want in original['spec']['sha256'].items(): assert S.sha(ROOT/rel) == want, rel
    dataset = json.loads((HERE/'DATASET.json').read_text())
    assert dataset['split_counts']['development'].get('separated_neutral', 0) >= 10, 'Insufficient new separated-neutral development cases'
    paths = [Path(__file__), HERE/'joint_candidates.py', HERE/'prepare.py', HERE/'README.md',
             HERE/'DATASET.json', HERE/'PREPARATION_PLAN.json', HERE/'development.pt', HERE/'development.sdf',
             HERE/'held_back.pt', HERE/'held_back.sdf', HERE/'preflight_v1/RUN_PLAN.json',
             HERE/'preflight_v1/run.py.txt']
    spec = {'status': 'development experiment; new identities but not prospective confirmatory test',
            'runtime_amendment': 'v1 stopped before saving a completed head on an empty edge list: torch.tensor([]) inferred float. Explicit long dtype added; no sample, budget, model or search changes. v1 plan/source/features preserved.',
            'population': dataset['split_counts']['development'], 'held_back_inference': False,
            'heads': S.SEEDS, 'encoder': str(S.ENCODER.relative_to(ROOT)),
            'current_B_selection': 'unchanged study.infer neural_constrained arm, 48-beam local B search',
            'new_search': 'only q/h on fixed selected B; Q embedded, no atom states revealed',
            'methods': METHODS, 'raw_assignment_budget': 32, 'returned_species_budget': 8,
            'plain': 'baseline first + top31 exact global product scores under Q',
            'mass': 'baseline first + 11/10/10 per positive mass max(Q,0)+0/1/2, round robin',
            'return_rule': 'baseline first if constructible, then proposal order; canonical unique, at most8; no target use',
            'metrics': 'native coverage1/2/4/8, raw buildability, returned species count, full pipeline and proposal wall times',
            'statistics': 'per-head paired molecule bootstrap 2000 draws, seed20261102; no p-values',
            'other_families_modified': False,
            'sha256': {**original['spec']['sha256'], **{str(p.relative_to(ROOT)): S.sha(p) for p in paths}}}
    spec = json.loads(json.dumps(spec))
    path = HERE/'RUN_PLAN.json'
    if path.exists(): assert json.loads(path.read_text())['spec'] == spec, 'Config/input drift; use versioned amendment'
    else: S.save(path, {'created_utc': datetime.now(timezone.utc).isoformat(), 'spec': spec})
    return S.sha(path)


def check_inputs():
    for rel, want in json.loads((HERE/'RUN_PLAN.json').read_text())['spec']['sha256'].items():
        assert S.sha(ROOT/rel) == want, rel


def features(phash, device):
    path = HERE/'features.pt'; receipt = HERE/'FEATURES.json'
    if path.exists():
        meta = json.loads(receipt.read_text())
        assert meta['plan_sha256'] == phash and meta['sha256'] == S.sha(path)
        return torch.load(path, weights_only=False, map_location='cpu')
    rows = torch.load(HERE/'development.pt', weights_only=False, map_location='cpu')
    encoder, args = S.P.load_backbone(S.ENCODER, device)
    geos = [S.P.geometry(r, 0., np.random.default_rng(11), args['cutoff']) for r in rows]
    out = []
    for start in range(0, len(rows), 8):
        out.extend(S.J.extract(encoder, geos[start:start+8], device))
    torch.save(out, path)
    S.save(receipt, {'plan_sha256': phash, 'sha256': S.sha(path), 'n': len(out), 'cutoff': args['cutoff']})
    del encoder
    if str(device).startswith('cuda'): torch.cuda.empty_cache()
    return out


def enumerate_sets(row, labels, logits, baseline):
    result = {}
    for method in METHODS:
        start = time.perf_counter()
        raw = C.proposals(logits, row['Q'], baseline['atom_states'], method)
        accepted = []; seen = set()
        for i, item in enumerate(raw):
            states = item.pop('states')
            assert sum(s//5-2 for s in states) == row['Q']
            smi, good = S.J.build_species(row, labels, torch.tensor(states))
            item.update(atom_states=states, smiles=smi, constructible=bool(good))
            if good and smi not in seen and len(accepted) < 8:
                seen.add(smi); accepted.append(i)
        result[method] = {'raw': raw, 'returned_indices': accepted, 'seconds': time.perf_counter()-start}
    return result


@torch.no_grad()
def infer(net, source, catalogue, device):
    start = time.perf_counter()
    old = S.infer(net, source, catalogue, device)
    baseline = old['arms']['neural_constrained']
    row = S.blind(source); labels = torch.tensor(baseline['labels'], dtype=torch.long)
    batch = S.J.pack([row], device); bd = labels.to(device)
    _, neural = net(batch, bd, torch.full_like(batch['atarget'], S.J.AMASK))
    logits = neural + S.J.catalogue_logits(batch, bd, catalogue)
    replay = S.assign_and_select(row, [bd], [0.], logits, True)
    replay_equal = replay['atom_states'] == baseline['atom_states']
    # Preserve the actual baseline even if a GPU rounding difference arises.
    sets = enumerate_sets(row, labels, logits.cpu().numpy(), baseline)
    # Proposal generation and returned-set selection have completed without targets.
    for method in METHODS:
        for item in sets[method]['raw']:
            item['native'] = bool(item['constructible'] and item['smiles'] == source['smiles'])
    graph = {tuple(p): c for p, c in zip(row['pairs'].tolist(), labels.tolist()) if c}
    return {'name': source['id'], 'bucket': source['bucket'], 'Q': source['Q'],
            'target_smiles': source['smiles'], 'target_q': source['q'].tolist(), 'target_h': source['h'].tolist(),
            'labels': labels.tolist(), 'strict_B': graph == source['truth'],
            'baseline': baseline, 'baseline_replay_equal': replay_equal, 'methods': sets,
            'seconds': time.perf_counter()-start}


def cell(seed, rows, catalogue, device, phash):
    path = HERE/f'evaluations/seed{seed}/predictions.json.gz'; done = path.parent/'DONE.json'
    if done.exists():
        d = json.loads(done.read_text())
        assert d['plan_sha256'] == phash and d['sha256'] == S.sha(path)
        print('verified completed', seed, flush=True); return
    net, _ = S.T.load_joint(S.checkpoint(seed), device)
    records = []
    for i, row in enumerate(rows):
        records.append(infer(net, row, catalogue, device))
        if (i+1) % 25 == 0: print('head', seed, i+1, '/', len(rows), flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as f: json.dump({'plan_sha256': phash, 'records': records}, f, separators=(',', ':'), allow_nan=False)
    S.save(done, {'plan_sha256': phash, 'sha256': S.sha(path), 'n': len(records)})
    del net
    if str(device).startswith('cuda'): torch.cuda.empty_cache()


def hit(record, method, k):
    entry = record['methods'][method]
    return any(entry['raw'][i]['native'] for i in entry['returned_indices'][:k])


def summarize(phash, rows):
    features_by_name = {r['id']: r for r in rows}; data = {}; checked = 0; replays = []
    for seed in S.SEEDS:
        with gzip.open(HERE/f'evaluations/seed{seed}/predictions.json.gz', 'rt') as f: saved = json.load(f)
        assert saved['plan_sha256'] == phash
        data[seed] = sorted(saved['records'], key=lambda r: r['name'])
        assert {r['name'] for r in data[seed]} == set(features_by_name)
        for r in data[seed]:
            feat = features_by_name[r['name']]
            if not r['baseline_replay_equal']: replays.append({'seed': seed, 'name': r['name']})
            for method in METHODS:
                entry = r['methods'][method]; raw = entry['raw']
                assert len(raw) <= 32 and raw[0]['atom_states'] == r['baseline']['atom_states']
                expected = []; seen = set()
                for i, a in enumerate(raw):
                    assert sum(s//5-2 for s in a['atom_states']) == r['Q']
                    smi, good = build(feat['z'].tolist(), feat['pairs'].tolist(), r['labels'], a['atom_states'])
                    assert (smi, bool(good)) == (a['smiles'], a['constructible'])
                    assert a['native'] == bool(good and smi == r['target_smiles'])
                    if good and smi not in seen and len(expected) < 8:
                        seen.add(smi); expected.append(i)
                    checked += 1
                assert entry['returned_indices'] == expected
                assert not r['baseline']['native'] or hit(r, method, 1)
    out = {'scope': 'new-identity ideal CCD development, not confirmation', 'plan_sha256': phash,
           'cells': {}, 'verification': {'independent_builds': checked, 'baseline_replay_mismatches': replays,
                                        'held_back_inference': False}}
    for group in ('pooled', *S.GROUPS):
        out['cells'][group] = {}
        for seed, records in data.items():
            rs = [r for r in records if group == 'pooled' or r['bucket'] == group]
            if not rs: continue
            c = {'n': len(rs), 'baseline_native': statistics.mean(int(r['baseline']['native']) for r in rs),
                 'baseline_constructible': statistics.mean(int(r['baseline']['constructible']) for r in rs),
                 'methods': {}}
            for method in METHODS:
                c['methods'][method] = {f'coverage{k}': statistics.mean(int(hit(r, method, k)) for r in rs) for k in (1, 2, 4, 8)}
                c['methods'][method].update(mean_returned_species=statistics.mean(len(r['methods'][method]['returned_indices']) for r in rs),
                    any_buildable=statistics.mean(bool(r['methods'][method]['returned_indices']) for r in rs),
                    mean_raw_attempts=statistics.mean(len(r['methods'][method]['raw']) for r in rs),
                    raw_buildable_fraction=statistics.mean(statistics.mean(int(a['constructible']) for a in r['methods'][method]['raw']) for r in rs),
                    proposal_seconds_median=statistics.median(r['methods'][method]['seconds'] for r in rs))
            rng = np.random.default_rng(20261102); idx = rng.integers(0, len(rs), (2000, len(rs)))
            c['mass_minus_plain'] = {}
            for k in (1, 2, 4, 8):
                diff = np.array([int(hit(r, METHODS[1], k))-int(hit(r, METHODS[0], k)) for r in rs])
                c['mass_minus_plain'][str(k)] = {'delta': float(diff.mean()), 'ci95': np.quantile(diff[idx].mean(1), [.025, .975]).tolist(),
                                                'rescued': int((diff == 1).sum()), 'harmed': int((diff == -1).sum())}
            out['cells'][group][str(seed)] = c
    S.save(HERE/'SUMMARY.json', out)
    lines = ['# 联合(q,h)候选：独立身份开发集结果', '',
             '固定原模型、固定已选B；当前单解与两种候选策略比较。每种候选策略最多32次原始赋值、返回最多8个可构建且canonical唯一的species。开发结果，不是确认测试。', '',
             '| 组 | n | Single native | Plain coverage@8 | Mass-stratified coverage@8 | 差值(pp) | Plain / mass平均返回数 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for group, cells in out['cells'].items():
        if not cells: continue
        values = list(cells.values()); avg = lambda fn: statistics.mean(fn(c) for c in values)
        fmt = lambda fn: f'{100*avg(fn):.2f} ± {100*statistics.stdev(fn(c) for c in values):.2f}'
        lines.append(f"| {group} | {values[0]['n']} | {fmt(lambda c:c['baseline_native'])} | {fmt(lambda c:c['methods']['plain_kbest']['coverage8'])} | {fmt(lambda c:c['methods']['mass_stratified']['coverage8'])} | {100*avg(lambda c:c['mass_minus_plain']['8']['delta']):+.2f} | {avg(lambda c:c['methods']['plain_kbest']['mean_returned_species']):.2f} / {avg(lambda c:c['methods']['mass_stratified']['mean_returned_species']):.2f} |")
    lines += ['', '数值为百分比均值±三个头样本SD；同一encoder。逐种子分组、coverage@1/2/4/8、成对分子bootstrap、可构建率、proposal耗时见SUMMARY.json。',
              '扩大集合且保留baseline意味着覆盖率不下降是设计保证，不是模型质量提升的独立证据。关键是同预算mass-stratified与plain之差，以及其它组的覆盖率代价。',
              '所有输出只由模型预测B、Q和原子logits构造；无真值q/h参与候选生成。保留集合后才附加native标签。',
              f"独立构图核验{checked}个原始赋值；baseline重放差异{len(replays)}项。held-back样本未运行推理。", '']
    (HERE/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('saved report', HERE/'REPORT.md', 'verification', out['verification'], flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default='cuda'); args = ap.parse_args()
    torch.set_num_threads(4); RDLogger.DisableLog('rdApp.*'); C.selftest()
    phash = freeze(); rows = features(phash, args.device)
    sources = torch.load(S.SOURCES, weights_only=False, map_location='cpu'); catalogue = S.J.make_catalogue(sources['train'])
    for seed in S.SEEDS: cell(seed, rows, catalogue, args.device, phash)
    check_inputs(); summarize(phash, rows); check_inputs()


if __name__ == '__main__': main()
