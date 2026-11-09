"""Oracle-B forward diagnostic, post-outcome and non-confirmatory. No training."""
import argparse
import gzip
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from rdkit import RDLogger

import study as S
from verify_outputs import build

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE/'fixed_reference_b'
ARMS = (*S.ARMS, 'neural_only_unconstrained')
CONTEXTS = ('reference', 'selected_neural', 'selected_catalogue', 'selected_unconstrained')
SELECTED = dict(zip(ARMS, (CONTEXTS[1], CONTEXTS[2], CONTEXTS[3], CONTEXTS[1])))


def plan():
    # Includes old predictions, analysis code and all old frozen model inputs.
    old = json.loads((HERE/'PROTOCOL.json').read_text())
    hashes = dict(old['spec']['sha256'])
    hashes.update({str(p.relative_to(ROOT)): S.sha(p) for p in [Path(__file__), OUT/'README.md',
                  HERE/'PROTOCOL.json', HERE/'features.pt', HERE/'FEATURES.json',
                  HERE/'SUMMARY.json', HERE/'verify_outputs.py', OUT/'PLAN_FAILED_PREFLIGHT.json',
                  OUT/'failed_preflight_v1.py.txt']})
    for seed in S.SEEDS:
        for name in ('predictions.json.gz', 'DONE.json'):
            p = HERE/f'evaluations/seed{seed}/{name}'
            hashes[str(p.relative_to(ROOT))] = S.sha(p)
    for rel, digest in old['spec']['sha256'].items():
        assert S.sha(ROOT/rel) == digest, rel
    spec = {'status': 'post-outcome exploratory oracle-B forward diagnostic; not prospective validation',
            'preflight_history': 'v1 stopped before saving any head: overly strict duplicate graph GPU allclose assertion. v2 deduplicates graph contexts before forward; history preserved.',
            'population': 'all original 270 molecules; all three frozen heads; no exclusions by outcome',
            'Q': 'true Q embedded in every arm; no reference q/h in forward or decoding',
            'reference_context': 'full reference B over existing geometry-only candidate pairs; no q/h revelation',
            'selected_context': 'original selected B fixed separately per original arm; raw neural uses neural-constrained selected B',
            'selection': 'none; one fixed B per context; no candidate reranking',
            'arms': ARMS, 'neural_only_unconstrained': 'raw 25-state joint argmax, no catalogue, no sum constraint',
            'solver': 'same original greedy suffix sum-product; exact Q sum, not joint MAP',
            'network_diagnostics': 'also report marginal-q argmax from raw 25-state probabilities, separately from joint-state argmax',
            'metrics': 'native, constructibility, q/h exact, q macro-F1, nonzero-q micro-F1, q/h accuracy, sum constraint satisfaction, all-neutral rate',
            'contrasts': 'reference-B minus recomputed selected-B within each arm; reference-B neural vs catalogue',
            'uncertainty': 'sample SD over three heads; paired molecule bootstrap separately per head, 2000 draws, seed 20261102',
            'hypothesis_tests': 0, 'original_Holm_family_modified': False,
            'nonclaims': ['not deployable scores', 'not theoretical upper bound', 'not a new training or held-out validation set',
                          'cannot establish unique protonation from heavy coordinates', 'no direct architecture comparison'],
            'sha256': hashes}
    spec = json.loads(json.dumps(spec))
    path = OUT/'PLAN.json'
    if path.exists():
        assert json.loads(path.read_text())['spec'] == spec, 'Diagnostic code/input drift: use new version'
    else:
        S.save(path, {'created_utc': S.datetime.now(S.timezone.utc).isoformat(), 'spec': spec})
    return S.sha(path)


def verify_inputs():
    p = json.loads((OUT/'PLAN.json').read_text())
    for rel, digest in p['spec']['sha256'].items():
        assert S.sha(ROOT/rel) == digest, rel


def annotate(item, source, pairs):
    a = torch.tensor(item['atom_states'])
    graph = {tuple(e): c for e, c in zip(pairs.tolist(), item['labels']) if c}
    item.update(native=bool(item['constructible'] and item['smiles'] == source['smiles']),
                strict=graph == source['truth'], charge_sum_correct=item['qsum'] == source['Q'],
                q_exact=bool(torch.equal(a//5-2, source['q'])),
                h_exact=bool(torch.equal(a%5, source['h'])),
                all_neutral=bool((a//5-2 == 0).all()))
    return item


@torch.no_grad()
def infer(net, source, original, catalogue, device):
    row = S.blind(source)
    pairs = row['pairs']
    reference = torch.tensor([source['truth'].get(tuple(p), 0) for p in pairs.tolist()], dtype=torch.long)
    assert {tuple(p): c for p, c in zip(pairs.tolist(), reference.tolist()) if c} == source['truth']
    labels = [reference] + [torch.tensor(original['arms'][arm]['labels'], dtype=torch.long) for arm in S.ARMS]
    assert all(len(b) == len(pairs) for b in labels)
    unique = []; mapping = []
    for label in labels:
        match = next((i for i, u in enumerate(unique) if torch.equal(u, label)), None)
        if match is None:
            match = len(unique); unique.append(label)
        mapping.append(match)
    batch = S.J.pack([row]*len(unique), device)
    bd = torch.cat(unique).to(device)
    assert torch.all(batch['atarget'] == 10)  # blinded placeholder, not reference states
    _, neural = net(batch, bd, torch.full_like(batch['atarget'], S.J.AMASK))
    prior = S.J.catalogue_logits(batch, bd, catalogue)
    n = len(row['z'])
    neural_unique = neural.split(n); prior_unique = prior.split(n)
    neural_parts = [neural_unique[i] for i in mapping]
    prior_parts = [prior_unique[i] for i in mapping]
    # Reuse exact same logits for identical B, avoiding GPU scatter rounding
    # differences between duplicate graphs becoming a supposed context effect.
    output = {'name': source['id'], 'bucket': source['bucket'], 'Q': source['Q'],
              'arms': {}, 'raw_network': {}}
    for arm in ARMS:
        output['arms'][arm] = {}
        for mode, context in (('reference', 'reference'), ('selected', SELECTED[arm])):
            ci = CONTEXTS.index(context)
            logits = (prior_parts[ci] if arm == 'catalogue_constrained' else neural_parts[ci]
                      if arm == 'neural_only_unconstrained' else neural_parts[ci]+prior_parts[ci])
            constrained = arm in ('neural_constrained', 'catalogue_constrained')
            item = S.assign_and_select(row, [labels[ci].to(device)], [0.], logits, constrained)
            output['arms'][arm][mode] = item
    # q marginal vs joint-state mode are distinct; do not conflate them.
    for mode, ci in (('reference', 0), ('selected', 1)):
        probs = neural_parts[ci].softmax(-1).reshape(n, 5, 5).sum(-1).cpu()
        output['raw_network'][mode] = {'q_probabilities_minus2_to_plus2': probs.tolist(),
                                       'q_marginal_argmax': (probs.argmax(-1)-2).tolist()}
    # Ground-truth atom states are attached and scored only after predictions.
    output.update(target_q=source['q'].tolist(), target_h=source['h'].tolist(), target_smiles=source['smiles'])
    for arm in ARMS:
        for item in output['arms'][arm].values():
            annotate(item, source, pairs)
    output['reproduction_of_original_selected'] = {
        arm: {k: output['arms'][arm]['selected'][k] == original['arms'][arm][k]
              for k in ('labels', 'atom_states', 'native', 'constructible', 'q_exact', 'h_exact')}
        for arm in S.ARMS}
    return output


def cell(seed, rows, old, catalogue, device, phash):
    path = OUT/f'seed{seed}/predictions.json.gz'; done = path.parent/'DONE.json'
    if done.exists():
        meta = json.loads(done.read_text())
        assert meta['plan_sha256'] == phash and meta['sha256'] == S.sha(path)
        print('verified completed head', seed, flush=True)
        return
    net, _ = S.T.load_joint(S.checkpoint(seed), device)
    results = []; start = time.perf_counter()
    for i, row in enumerate(rows):
        results.append(infer(net, row, old[row['id']], catalogue, device))
        if (i+1) % 50 == 0:
            print('head', seed, i+1, '/', len(rows), flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        json.dump({'plan_sha256': phash, 'records': results}, f, separators=(',', ':'), allow_nan=False)
    S.save(done, {'plan_sha256': phash, 'sha256': S.sha(path), 'n': len(results),
                  'seconds': time.perf_counter()-start})
    del net
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()


def metric(records, arm, context):
    projected = [dict(r, arms={arm: r['arms'][arm][context]}) for r in records]
    result = S.metric(projected, arm)
    result['all_neutral_rate'] = statistics.mean(int(r['arms'][arm][context]['all_neutral']) for r in records)
    result['all_neutral_names'] = [r['name'] for r in records if r['arms'][arm][context]['all_neutral']]
    return result


def raw_metrics(records, mode):
    truth = np.array([q for r in records for q in r['target_q']])
    probs = np.array([p for r in records for p in r['raw_network'][mode]['q_probabilities_minus2_to_plus2']])
    pred = np.array([q for r in records for q in r['raw_network'][mode]['q_marginal_argmax']])
    assert np.allclose(probs.sum(1), 1., atol=1e-6)
    assert np.array_equal(pred, probs.argmax(1)-2)
    cm = np.zeros((5, 5), int)
    np.add.at(cm, (truth+2, pred+2), 1)
    classes = np.unique(truth)+2
    den = cm.sum(0)+cm.sum(1)
    f1 = np.divide(2*cm.diagonal(), den, out=np.zeros(5, float), where=den != 0)
    charged = truth != 0
    return {'q_marginal_macro_f1': float(f1[classes].mean()),
            'q_marginal_all_neutral_rate': statistics.mean(int(all(q == 0 for q in r['raw_network'][mode]['q_marginal_argmax'])) for r in records),
            'true_charged_site_count': int(charged.sum()),
            'true_charged_site_correct': int(((pred == truth) & charged).sum()),
            'mean_probability_of_reference_charge_at_charged_sites': float(probs[charged, truth[charged]+2].mean()) if charged.any() else None,
            'mean_probability_neutral_at_charged_sites': float(probs[charged, 2].mean()) if charged.any() else None,
            'q_confusion_matrix_minus2_to_plus2': cm.tolist()}


def contrast(records, arm):
    out = {}
    rng = np.random.default_rng(20261102)
    idx = rng.integers(0, len(records), (2000, len(records)))
    for key in ('native', 'q_exact', 'h_exact', 'all_neutral'):
        diffs = np.array([int(r['arms'][arm]['reference'][key])-int(r['arms'][arm]['selected'][key]) for r in records])
        out[key] = {'delta': float(diffs.mean()), 'pointwise_paired_molecule_bootstrap_ci95': np.quantile(diffs[idx].mean(1), [.025, .975]).tolist(),
                    'rescued': int((diffs == 1).sum()), 'harmed': int((diffs == -1).sum())}
    return out


def report(data, phash):
    out = {'scope': 'post-outcome oracle-B forward diagnostic', 'plan_sha256': phash, 'cells': {},
           'verification': {'rebuilt_graphs': 0, 'old_selected_mismatches': []}}
    features = {r['id']: r for r in torch.load(HERE/'features.pt', weights_only=False, map_location='cpu')}
    for seed, records in data.items():
        for r in records:
            feat = features[r['name']]
            for arm in ARMS:
                for mode, a in r['arms'][arm].items():
                    smi, good = build(feat['z'].tolist(), feat['pairs'].tolist(), a['labels'], a['atom_states'])
                    assert (smi, bool(good)) == (a['smiles'], a['constructible'])
                    assert a['native'] == bool(good and smi == r['target_smiles'])
                    assert a['qsum'] == sum(s//5-2 for s in a['atom_states'])
                    assert a['q_exact'] == ([s//5-2 for s in a['atom_states']] == r['target_q'])
                    assert a['h_exact'] == ([s%5 for s in a['atom_states']] == r['target_h'])
                    if arm in S.ARMS[:2]: assert a['qsum'] == r['Q']
                    out['verification']['rebuilt_graphs'] += 1
            for arm, flags in r['reproduction_of_original_selected'].items():
                if not all(flags.values()):
                    out['verification']['old_selected_mismatches'].append({'seed': seed, 'name': r['name'], 'arm': arm, 'flags': flags})
    for group in ('pooled', *S.GROUPS):
        cellout = {}
        for seed, records in data.items():
            rs = [r for r in records if group == 'pooled' or r['bucket'] == group]
            cellout[str(seed)] = {'n': len(rs), 'arms': {arm: {mode: metric(rs, arm, mode) for mode in ('selected', 'reference')} for arm in ARMS},
                                 'raw_network_marginal_q': {mode: raw_metrics(rs, mode) for mode in ('selected', 'reference')},
                                 'oracle_B_minus_fixed_selected_B': {arm: contrast(rs, arm) for arm in ARMS}}
        out['cells'][group] = cellout
    S.save(OUT/'SUMMARY.json', out)
    lines = ['# 固定参考键图后的原子态预测诊断', '',
             '这是事后、使用参考B的诊断，不是可部署预测成绩或理论上界。原模型、原协议、原30项Holm族不变；无训练、无新增假设检验。', '',
             '仅用缓存几何特征与真实总电荷Q；参考q/h始终mask。每个臂分别固定原先选中的B，再以参考B重新前向；没有重新排序B候选。', '',
             '## 重点：23个电荷分离中性分子（均值±三个头样本SD，%）', '',
             '| 模式 | 键图上下文 | Native | q向量exact | h向量exact | 电荷macro-F1 | 全中性比例 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for mode in ('selected', 'reference'):
            metrics = [v['arms'][arm][mode] for v in out['cells']['separated_neutral'].values()]
            vals = []
            for k in ('native', 'q_exact', 'h_exact', 'q_macro_f1_truth_present_classes', 'all_neutral_rate'):
                vv = [x[k] for x in metrics]
                vals.append(f'{100*statistics.mean(vv):.2f} ± {100*statistics.stdev(vv):.2f}')
            lines.append('| '+ ' | '.join([arm, mode, *vals])+' |')
    lines += ['', '## 原始神经网络：对h边缘化后的电荷预测（无catalogue、无电荷和约束）', '',
              '| 种子 | B上下文 | q marginal macro-F1 | 全中性分子数/23 | 真实带电位置预测正确数/48 | 真实带电位置：正确q概率 / 中性概率 |',
              '|---|---|---:|---:|---:|---:|']
    for seed, c in out['cells']['separated_neutral'].items():
        for mode, m in c['raw_network_marginal_q'].items():
            lines.append(f"| {seed} | {mode} | {100*m['q_marginal_macro_f1']:.2f}% | {round(23*m['q_marginal_all_neutral_rate'])}/23 | {m['true_charged_site_correct']}/48 | {100*m['mean_probability_of_reference_charge_at_charged_sites']:.2f}% / {100*m['mean_probability_neutral_at_charged_sites']:.2f}% |")
    lines += ['', '## 四分组参考B结果：native / 电荷macro-F1（3头均值，%）', '',
              '| 组 | neural+catalogue+Q | catalogue+Q | neural+catalogue无和约束 | raw neural无和约束 |',
              '|---|---:|---:|---:|---:|']
    for group in ('pooled', *S.GROUPS):
        vals = []
        for arm in ARMS:
            mm = [v['arms'][arm]['reference'] for v in out['cells'][group].values()]
            vals.append(f"{100*statistics.mean(x['native'] for x in mm):.2f} / {100*statistics.mean(x['q_macro_f1_truth_present_classes'] for x in mm):.2f}")
        lines.append('| '+ ' | '.join([group, *vals])+' |')
    lines += ['', '## 解释边界', '',
              '- selected上下文是各臂各自原选B，不是所有臂共用同一张预测图；reference上下文在各臂间完全一致。',
              '- raw neural行的selected B固定为原neural_constrained臂的B。原始联合(q,h)状态argmax与对h边缘化的q argmax并非同一个决策。',
              '- 原子索引q/h exact惩罚共振和对称等价位置交换；native不要求同一索引定域方式。',
              '- 参考B本身可能通过键级和离域表示泄露化学状态信息；因此不能将此表当成部署能力。',
              '- 零样本bootstrap差异不代表总体效应确定为零；所有区间仅为每个固定头下的逐分子探索性区间。',
              f"- 独立重构核验{out['verification']['rebuilt_graphs']}个选定输出；原selected输出不一致条目{len(out['verification']['old_selected_mismatches'])}个（明细见SUMMARY.json）。", '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Summary saved; verification:', out['verification'], flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default='cuda'); args = ap.parse_args()
    torch.set_num_threads(4); RDLogger.DisableLog('rdApp.*')
    S.selftest()
    phash = plan()
    rows = S.features(S.sha(HERE/'PROTOCOL.json'), args.device)
    assert len(rows) == 270
    sources = torch.load(S.SOURCES, weights_only=False, map_location='cpu')
    catalogue = S.J.make_catalogue(sources['train'])
    data = {}
    for seed in S.SEEDS:
        with gzip.open(HERE/f'evaluations/seed{seed}/predictions.json.gz', 'rt') as f:
            old = {r['name']: r for r in json.load(f)['records']}
        cell(seed, rows, old, catalogue, args.device, phash)
        with gzip.open(OUT/f'seed{seed}/predictions.json.gz', 'rt') as f:
            saved = json.load(f)
        assert saved['plan_sha256'] == phash
        data[seed] = saved['records']
    verify_inputs()
    report(data, phash)
    verify_inputs()


if __name__ == '__main__':
    main()
