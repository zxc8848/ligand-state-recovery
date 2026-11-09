"""Verify oracle-B outputs and audit charge-separated training coverage."""
import collections
import gzip
import json
from pathlib import Path

import torch
from rdkit import Chem, RDLogger

import fixed_reference_b as F
from prepare import classify
from verify_outputs import build

HERE = F.HERE
OUT = F.OUT


def main():
    RDLogger.DisableLog('rdApp.*')
    F.verify_inputs()
    summary = json.loads((OUT/'SUMMARY.json').read_text())
    plan_hash = F.S.sha(OUT/'PLAN.json')
    assert summary['plan_sha256'] == plan_hash
    sources = torch.load(F.S.SOURCES, weights_only=False, map_location='cpu')
    composition = {}
    for split, source_rows in sources.items():
        counts = collections.Counter()
        for row in source_rows:
            mol = Chem.MolFromSmiles(row['smiles'])
            assert mol is not None
            counts[classify(mol)] += 1
        composition[split] = dict(counts)
    rows = {r['id']: r for r in torch.load(HERE/'features.pt', weights_only=False, map_location='cpu')}
    checked = 0; per_seed = {}; cases = []
    for seed in F.S.SEEDS:
        folder = OUT/f'seed{seed}'
        done = json.loads((folder/'DONE.json').read_text())
        assert done['sha256'] == F.S.sha(folder/'predictions.json.gz') and done['plan_sha256'] == plan_hash
        with gzip.open(folder/'predictions.json.gz', 'rt') as f:
            payload = json.load(f)
        assert payload['plan_sha256'] == plan_hash
        data = payload['records']
        assert len(data) == len({r['name'] for r in data}) == 270
        assert {r['name'] for r in data} == set(rows)
        with gzip.open(HERE/f'evaluations/seed{seed}/predictions.json.gz', 'rt') as f:
            old = {r['name']: r for r in json.load(f)['records']}
        for r in data:
            source = rows[r['name']]
            assert r['target_q'] == source['q'].tolist() and r['target_h'] == source['h'].tolist()
            assert r['target_smiles'] == source['smiles'] and r['Q'] == source['Q']
            ref = [source['truth'].get(tuple(p), 0) for p in source['pairs'].tolist()]
            assert {tuple(p): c for p, c in zip(source['pairs'].tolist(), ref) if c} == source['truth']
            for arm in F.ARMS:
                for mode, result in r['arms'][arm].items():
                    assert result['labels'] == (ref if mode == 'reference' else old[r['name']]['arms'][
                        arm if arm in F.S.ARMS else 'neural_constrained']['labels'])
                    assert len(result['atom_states']) == len(source['z'])
                    assert all(0 <= s < 25 for s in result['atom_states'])
                    q = [s//5-2 for s in result['atom_states']]
                    h = [s%5 for s in result['atom_states']]
                    assert result['charge_sum_correct'] == (sum(q) == r['Q'])
                    assert result['q_exact'] == (q == r['target_q'])
                    assert result['h_exact'] == (h == r['target_h'])
                    assert result['all_neutral'] == all(v == 0 for v in q)
                    smi, good = build(source['z'].tolist(), source['pairs'].tolist(), result['labels'], result['atom_states'])
                    assert (smi, bool(good)) == (result['smiles'], result['constructible'])
                    assert result['native'] == bool(good and smi == r['target_smiles'])
                    checked += 1
        z = [r for r in data if r['bucket'] == 'separated_neutral']
        assert len(z) == 23
        per_seed[str(seed)] = {}
        for arm in F.ARMS:
            a = [r['arms'][arm]['reference'] for r in z]
            per_seed[str(seed)][arm] = {
                'native': sum(x['native'] for x in a), 'all_neutral': sum(x['all_neutral'] for x in a),
                'constructible': sum(x['constructible'] for x in a),
                'all_neutral_and_constructible': sum(x['all_neutral'] and x['constructible'] for x in a),
                'q_exact': sum(x['q_exact'] for x in a), 'h_exact': sum(x['h_exact'] for x in a)}
        for r in z:
            a = r['arms']['neural_constrained']['reference']
            cat = r['arms']['catalogue_constrained']['reference']
            raw = r['raw_network']['reference']
            cases.append({'name': r['name'], 'seed': seed, 'target_smiles': r['target_smiles'],
                          'neural_referenceB_native': a['native'], 'neural_referenceB_all_neutral': a['all_neutral'],
                          'neural_referenceB_constructible': a['constructible'],
                          'neural_referenceB_q_exact': a['q_exact'], 'neural_referenceB_h_exact': a['h_exact'],
                          'neural_catalogue_states_differ': a['atom_states'] != cat['atom_states'],
                          'q_errors': [{'atom': i, 'truth': t, 'pred': s//5-2} for i, (t, s) in enumerate(zip(r['target_q'], a['atom_states'])) if t != s//5-2],
                          'h_errors': [{'atom': i, 'truth': t, 'pred': s%5} for i, (t, s) in enumerate(zip(r['target_h'], a['atom_states'])) if t != s%5],
                          'raw_network_at_charged_sites': [{'atom': i, 'truth_q': q,
                            'pred_q_marginal': raw['q_marginal_argmax'][i],
                            'p_reference_q': raw['q_probabilities_minus2_to_plus2'][i][q+2],
                            'p_neutral': raw['q_probabilities_minus2_to_plus2'][i][2]}
                            for i, q in enumerate(r['target_q']) if q]})
    F.verify_inputs()
    assert not summary['verification']['old_selected_mismatches']
    assert len(json.loads((HERE/'SUMMARY.json').read_text())['tests']) == 30
    result = {'post_outcome_exploratory': True, 'plan_sha256': plan_hash,
              'summary_sha256': F.S.sha(OUT/'SUMMARY.json'), 'auditor_sha256': F.S.sha(Path(__file__)),
              'graphs_independently_rebuilt': checked, 'original_selected_outputs_all_reproduced': True,
              'all_frozen_inputs_unchanged': True, 'original_Holm_test_count': 30,
              'head_training_composition_by_current_group_definition': composition,
              'per_seed_separated_neutral_referenceB': per_seed, 'separated_cases': cases}
    F.S.save(OUT/'AUDIT.json', result)
    lines = ['# 固定参考B实验：解释与下一步', '',
             '本轮为事后探索性诊断；没有训练，没有改写原冻结实验或其30项检验族。', '',
             '## 实测结论', '',
             '1. 键图错误确有影响，但不是唯一障碍。23个电荷分离中性分子中，给出参考B后，神经+catalogue+Q臂的native从23.19%升至30.43%（7/23，三个头均相同）；q向量exact由15.94%升至34.78%，h向量exact仍为30.43%。', '',
             '2. 正确B下仍有15/23被该臂预测成全中性。Q=0只限制净电荷，不区分“全中性”与“正负电荷分离”。因此继续加强同一个总电荷和约束无法解决这个判别。', '',
             '3. 去掉catalogue和总电荷和约束后，原始神经网络在参考B下仍偏中性。对h边缘化后，48个真实带电位置仅预测正确6、14、9个；这些位置的平均中性概率分别83.68%、72.57%、80.09%。这排除了“偏中性完全由catalogue或错误B引起”的单一解释，但不证明网络完全没有有用信息。', '',
             '4. 同一参考B下，catalogue+Q也只有7/23 native；神经+catalogue+Q并未超过它。相反，在阳离子和阴离子组，神经原子态仍有增益，不能将此失败概括为整个电荷预测模块无用。', '',
             '## 训练覆盖核查（按与本次测试完全相同的分组函数）', '',
             '| 组 | Train | Validation |', '|---|---:|---:|']
    for group in ('uncharged_neutral', 'cation', 'anion', 'local_charge_separation', 'separated_neutral'):
        lines.append(f"| {group} | {composition['train'].get(group, 0)} | {composition['val'].get(group, 0)} |")
    lines += ['', '原来的500个内部带正负电荷训练分子中，451个是局部电荷分离；按最短正负电荷图距离≥3的定义，真正对应本组的只有49个，验证集仅4个。这个覆盖差异提供合理的改进靶点，但未经受控重训，不能声称已证明因果。这里统计的是原子态头训练集，不代表整个encoder预训练语料。', '',
              '## 不能过度解释', '',
              '- 参考B可能携带关于电荷/质子化的信息，故这些结果不是无模板部署能力，也不是严格上界。',
              '- 缺H时存在多种合法的(q,h)状态；匹配不到单个参考状态，不自动等同于化学无效。AUDIT.json另外区分全中性输出是否可构建，避免把合法替代物与构图失败混为一谈。',
              '- q向量exact会惩罚共振/对称等价电荷位置，不作为唯一化学正确性判据。',
              '- 此实验固定B，不重新排序候选；它不能回答候选集合是否覆盖参考状态。', '',
              '## 建议的下一步', '',
              '现在不必从头训练。若继续优化，最小值得做的开发实验是：在独立训练/验证来源上，把“局部电荷分离”与“远距离电荷分离”明确分开，先补充后者的验证覆盖，再比较联合(q,h)候选生成与当前单解。', '',
              '候选集应允许Q=0时同时存在全中性和电荷分离假设，而不是用规则强迫所有Q=0分子带正负电荷。要固定候选预算，并同时报告覆盖率、每分子候选数、可构建率和原有中性/带电组的代价。', '',
              '当前23例继续保留作已使用的诊断集，不再作为调参的确认集。任何进一步开发须在新的独立开发集完成，正式结论再用未读过预测的新测试集确认。', '',
              f'核验：{checked}个输出独立重构；原selected-B重计算结果与原保存结果全部一致；所有原冻结输入指纹不变。', '']
    (OUT/'FINDINGS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'separated_cases'}, indent=2))


if __name__ == '__main__':
    main()
