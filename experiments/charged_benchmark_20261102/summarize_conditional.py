"""Independent arithmetic checks and case ledger for post-hoc charge analysis."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from rdkit import RDLogger

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE/'exploratory_conditional'
sys.path.insert(0, str(ROOT/'experiments/prob_mask_20261002'))
import joint as J

ARMS = ('neural_constrained', 'catalogue_constrained', 'neural_unconstrained')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    RDLogger.DisableLog('rdApp.*')
    plan = json.loads((OUT/'ANALYSIS_PLAN.json').read_text())
    d = json.loads((OUT/'SUMMARY.json').read_text())
    assert d['analysis_plan_sha256'] == sha(OUT/'ANALYSIS_PLAN.json')
    assert plan['analysis_sha256'] == sha(HERE/'conditional_analysis.py')
    for rel, want in plan['inputs_sha256'].items():
        assert sha(ROOT/rel) == want, rel
    feats = {r['id']: r for r in torch.load(HERE/'features.pt', weights_only=False, map_location='cpu')}
    rows = {r['id']: r for r in torch.load(HERE/'selected.pt', weights_only=False, map_location='cpu')}
    stats = {}; checked = 0; oracle_builds = 0
    case_lines = ['# 电荷分离中性组逐例账本（事后诊断）', '',
                  '每个种子包含同一批23个分子；不要视为69个独立分子。B/q/h均为原子索引严格一致，native允许规范SMILES等价。', '',
                  '| CCD | Seed | Native neural / catalogue / free | Neural B/q/h exact | Free all-neutral | Neural q错位原子 | Neural h错误原子 |',
                  '|---|---:|---|---|---|---|---|']
    for seed in (20261002, 20261003, 20261004):
        with gzip.open(HERE/f'evaluations/seed{seed}/predictions.json.gz', 'rt') as f:
            records = json.load(f)['records']
        byname = {r['name']: r for r in records}
        for group, cells in d['cells'].items():
            rs = [r for r in records if group == 'pooled' or r['bucket'] == group]
            S = [r for r in rs if sum(s//5-2 for s in r['arms'][ARMS[2]]['atom_states']) == r['Q']]
            saved = cells[str(seed)]
            assert set(saved['S_names']) == {r['name'] for r in S}
            for scope, chosen in (('all', rs), ('common_S', S)):
                for arm in ARMS:
                    for key in ('native', 'q_exact', 'charge_sum_correct'):
                        want = sum(r['arms'][arm][key] for r in chosen)/len(chosen) if chosen else None
                        got = saved[scope][arm][key]
                        assert got is None if want is None else abs(got-want) < 1e-12
                        checked += 1
                    truth = [q for r in chosen for q in r['target_q']]
                    pred = [s//5-2 for r in chosen for s in r['arms'][arm]['atom_states']]
                    if chosen:
                        f1s = []
                        for q in saved['truth_present_q_classes']:
                            tp = sum(t == q and p == q for t, p in zip(truth, pred))
                            den = truth.count(q)+pred.count(q)
                            f1s.append(2*tp/den if den else 0.)
                        assert abs(sum(f1s)/len(f1s)-saved[scope][arm]['q_macro_f1']) < 1e-12
                        checked += 1
        zrs = [r for r in records if r['bucket'] == 'separated_neutral']
        diag = d['separated_diagnostics'][str(seed)]
        ds = {r['name']: r for r in diag}
        pooled = d['cells']['pooled'][str(seed)]
        common = pooled['contrasts_constrained_minus_unconstrained']['common_S']
        stat = {'native_count_gain_all': sum(int(r['arms'][ARMS[0]]['native'])-int(r['arms'][ARMS[2]]['native']) for r in records),
                'native_count_gain_common_S': common['native_rescued']-common['native_harmed'],
                'separated_neutral': {}}
        for arm in ARMS:
            aa = [r['arms'][arm] for r in zrs]
            stat['separated_neutral'][arm] = {
                'n': len(aa),
                'all_atoms_predicted_neutral': sum(all(s//5-2 == 0 for s in a['atom_states']) for a in aa),
                'true_charged_sites_correct': sum((s//5-2) == q for r in zrs for s, q in zip(r['arms'][arm]['atom_states'], r['target_q']) if q),
                'true_charged_sites': sum(q != 0 for r in zrs for q in r['target_q']),
                'reference_context_unseen_charged_state_sites': sum(a['q'] != 0 and a['target_state_empirical_count'] == 0 for r in diag for a in r['reference_context_support']),
                'predicted_context_unseen_charged_state_sites': sum(a['q'] != 0 and a['target_state_empirical_count'] == 0 for r in diag for a in r['arms'][arm]['predicted_context_target_support']),
                'predicted_context_unseen_charge_sites': sum(a['q'] != 0 and a['target_q_empirical_count'] == 0 for r in diag for a in r['arms'][arm]['predicted_context_target_support'])}
        for r in zrs:
            item = ds[r['name']]; a = item['arms'][ARMS[0]]
            flags = [r['arms'][arm]['native'] for arm in ARMS]
            freezero = all(s//5-2 == 0 for s in r['arms'][ARMS[2]]['atom_states'])
            fmt = lambda values: ','.join(map(str, values)) or '-'
            case_lines.append('| '+ ' | '.join([item['component_id'], str(seed), '/'.join(str(int(v)) for v in flags),
                '/'.join(str(int(a[k])) for k in ('strict', 'q_exact', 'h_exact')), str(int(freezero)),
                fmt(a['q_error_atoms']), fmt(a['h_error_atoms'])])+' |')
            # Cross-check every oracle rebuild with the original builder path.
            row = rows[r['name']]; feat = feats[r['name']]
            refb = torch.tensor([row['truth'].get(tuple(p), 0) for p in feat['pairs'].tolist()])
            for arm in ARMS:
                pred = r['arms'][arm]
                for key, result in item['arms'][arm]['oracle_replacement_controls'].items():
                    _, bmode, _, qmode, _, hmode = key.split('_')
                    q = r['target_q'] if qmode == 'ref' else [s//5-2 for s in pred['atom_states']]
                    h = r['target_h'] if hmode == 'ref' else [s%5 for s in pred['atom_states']]
                    atoms = torch.tensor([(qi+2)*5+hi for qi, hi in zip(q, h)])
                    smi, good = J.build_species(feat, refb if bmode == 'ref' else torch.tensor(pred['labels']), atoms)
                    assert bool(good) == result['constructible']
                    assert bool(good and smi == r['target_smiles']) == result['native']
                    oracle_builds += 1
        stats[str(seed)] = stat
    gain_all = sum(x['native_count_gain_all'] for x in stats.values())
    gain_S = sum(x['native_count_gain_common_S'] for x in stats.values())
    lines = ['# 电荷约束与电荷分离中性组：探索性结论', '',
             '本分析在看到原实验结果之后新增，不属于预注册确认分析。原协议、模型、预测和30项Holm检验族均未修改；没有新增显著性检验。', '',
             '## 1. 电荷约束的收益主要来自何处', '',
             '| 种子 | 共同子集 S | S上 native：约束 / 无约束 | S上 q-vector exact：约束 / 无约束 | 全体native净增例数 / S内净增 |',
             '|---|---:|---:|---:|---:|']
    for seed, stat in stats.items():
        x = d['cells']['pooled'][seed]; c, u = x['common_S'][ARMS[0]], x['common_S'][ARMS[2]]
        lines.append(f"| {seed} | {x['S_n']}/{x['n']} | {100*c['native']:.2f}% / {100*u['native']:.2f}% | {100*c['q_exact']:.2f}% / {100*u['q_exact']:.2f}% | {stat['native_count_gain_all']} / {stat['native_count_gain_common_S']} |")
    lines += ['', f'三个头的结果合计有{gain_all}次native净增，其中{gain_all-gain_S}次发生在无约束输出总电荷错误的分子上。这个合计只用来描述收益来源，不把重复分子当独立样本做推断。', '',
              '结论：总电荷硬约束确实实现守恒，并改善端到端可用率；共同子集不支持它普遍改善电荷定位。全样本电荷macro-F1仍从63.67%提高到71.04%，这不是native定义的直接同义反复，但不能据此声称网络本身学会了更好的定位：比较的是两种解码。', '',
              '共同子集bootstrap是对已观察S重采样，无法消除选择偏差。零差异种子的[0,0]区间仅表示已观察配对没有差异，不是总体效应必然为零。', '',
              '## 2. 电荷分离中性组：中性化和耦合错误', '',
              '| 种子 | 全原子预测q=0：神经约束 / catalogue约束 / 无约束（分母23） | 真实带电位置预测正确：三臂（分母48） |',
              '|---|---:|---:|']
    for seed, stat in stats.items():
        z = stat['separated_neutral']
        lines.append('| '+seed+' | '+ ' / '.join(str(z[a]['all_atoms_predicted_neutral']) for a in ARMS)+' | '+ ' / '.join(str(z[a]['true_charged_sites_correct']) for a in ARMS)+' |')
    first = stats['20261002']['separated_neutral'][ARMS[0]]
    lines += ['', f"参考键图上下文中，48个真实带电原子的参考(q,h)全部曾在实际回退层级的训练catalogue中出现（缺失{first['reference_context_unseen_charged_state_sites']}个）。所以这批失败不能简单归因于catalogue完全没收录这些原子态。正伪计数也使经验零频状态并非硬不可行。", '',
              '但是换成预测键图后，参考电荷在部分局部上下文中的经验频次变为0（神经约束臂三个头分别11、12、10个带电原子位置）。这提示键图与原子态先验之间有耦合，不能把全部失败独立归为电荷头问题。', '',
              '在参考B和参考q都已给定时，仅保留模型预测h，神经约束臂也只有7/23能恢复native；保留预测q、给参考B和参考h则只有3/23、3/23、5/23。说明电荷与氢数都是短板。参考替换控制可降低原有正确率，因为正确native也可能依赖等价的键/电荷定域表示；它们不是单调修复或可相加的归因。', '',
              '## 3. 下一步边界', '',
              '- 论文中可保留“已知总电荷约束改善守恒与端到端身份恢复”，但删除“约束普遍提升电荷定位”的机制性推断。',
              '- 电荷分离中性组应作为明确失败模式，而不是只报Q=0总体成绩。该组只有23个不同分子，不能据此宣布普遍机制。',
              '- 如继续开发，优先在独立训练/开发数据上研究成对电荷状态与氢数的联合候选，而非继续增加同一总电荷约束强度；当前测试23例不得变成调参训练集。',
              '- 要验证神经头在正确键图上是否仍偏向中性，需新增“固定参考B后重新前向预测原子态”的诊断。本轮只替换已保存输出，没有重跑前向，不能替代该实验。',
              '- 真正确认实验须另找未读过预测的新样本；现有结果不能重新命名为预注册结果。', '',
              '详细分组、每种子bootstrap区间及8种真值替换控制见REPORT.md与SUMMARY.json，逐分子见CASES.md。', '']
    (OUT/'FINDINGS.md').write_text('\n'.join(lines), encoding='utf-8')
    (OUT/'CASES.md').write_text('\n'.join(case_lines)+'\n', encoding='utf-8')
    for rel, want in plan['inputs_sha256'].items():
        assert sha(ROOT/rel) == want, rel
    verification = {'scope': 'post-outcome exploratory analysis', 'arithmetic_checks': checked,
                    'oracle_builds_crosschecked': oracle_builds, 'original_inputs_unchanged': True,
                    'original_native_hypothesis_tests': len(json.loads((HERE/'SUMMARY.json').read_text())['tests']),
                    'analysis_plan_sha256': sha(OUT/'ANALYSIS_PLAN.json'), 'summary_sha256': sha(OUT/'SUMMARY.json'),
                    'verifier_sha256': sha(Path(__file__)), 'per_seed_diagnostics': stats}
    (OUT/'VERIFICATION.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
