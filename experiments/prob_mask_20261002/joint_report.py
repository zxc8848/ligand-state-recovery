"""Report joint B/h/q validation and paired mask versus independent sampling."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE/'joint_v1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', default='evaluation.json')
    parser.add_argument('--extra-evaluation', nargs='*', default=[])
    args = parser.parse_args()
    ev = json.loads((RUN/args.evaluation).read_text(encoding='utf-8'))
    record_map = {(r['id'], r['sigma'], r['method'], r['rounds']): r for r in ev['records']}
    summary_map = {(r['sigma'], r['subset'], r['method'], r['rounds']): r for r in ev['summary']}
    for filename in args.extra_evaluation:
        extra = json.loads((RUN/filename).read_text(encoding='utf-8'))
        assert extra['checkpoint_epoch'] == ev['checkpoint_epoch']
        assert extra['args']['seed'] == ev['args']['seed']
        for r in extra['records']:
            key = (r['id'], r['sigma'], r['method'], r['rounds'])
            if key in record_map:
                assert record_map[key]['species_hits'] == r['species_hits'], 'Repeated control changed'
            record_map[key] = r
        for r in extra['summary']:
            summary_map[(r['sigma'], r['subset'], r['method'], r['rounds'])] = r
    ev['records'] = list(record_map.values()); ev['summary'] = list(summary_map.values())
    prep = json.loads((RUN/'prepare.json').read_text(encoding='utf-8'))
    training = json.loads((RUN/'training.json').read_text(encoding='utf-8'))
    lines = ['# 联合 B/h/q mask：实测记录', '',
             '开发验证实验；正式论文与 checkpoint 未更改。', '',
             '## 训练与输入', '',
             f"- 训练 {prep['stats']['train']['n']} 个分子；验证 {prep['stats']['val']['n']} 个分子，其中 GEOM core 300 个，电荷挑战集 250 个。",
             '- 输入只有 Z、R、给定总电荷 Q；h/q 标签及 reference SMILES 不作推理输入。',
             '- 原子 h/q 合并成 25 状态 token，与 5 类边 token 联合 masked completion。MASK 和 no-bond/h=0/q=0 不混用。',
             f"- 新模型 {training['parameters']:,} 参数；选中 epoch {ev['checkpoint_epoch']}；训练约 {training['history'][-1]['seconds']:.1f} s。",
             '- 每个候选都经相同的精确 sum(q)=Q 动态规划采样；这不是完整化学有效性约束，也不是完整图后验。',
             '- q=±2、h=4 没有训练标签支持，仅为预留词表；不能据此声称有这些状态的预测能力。',
             '- 新补全训练/验证按去 H、去电荷、去键级骨架隔离；未重新审计整个冻结 backbone 训练语料的身份重叠。因此这仍是开发验证，不是独立跨域测试。', '',
             '- B_then_HQ 是观察首轮错误后的开发改动：先采样完整 B，再让全部原子 (h,q) 在已采样 B 下补全；B_then_HQ_catalogue 再乘以同一训练局部状态先验。并非预注册验证。', '',
             '## 同预算 species 候选覆盖', '',
             'Species 按可 sanitize 的非立体 canonical SMILES 匹配；闭壳层有效性另外要求无自由基且连通。8 个原始候选，不按真值重排、不追加抽样。', '',
             '| σ (Å) | 子集 | 方法 | 候选数 | Species coverage | Strict B/h/q coverage | B coverage | 闭壳层连通有效率 | 有效 species 数/输入 |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in ev['summary']:
        if r['subset'] not in ['geom_core', 'charge_challenge', 'nH']:
            continue
        name = r['method']+(f"-{r['rounds']}r" if r['method'] == 'mask_joint' else '')
        lines.append(f"| {r['sigma']:.2f} | {r['subset']} (n={r['n']}) | {name} | {r['budget']} | {r['species_coverage']:.3f} | {r['joint_label_coverage']:.3f} | {r['graph_coverage']:.3f} | {r['closed_shell_connected_valid_fraction']:.3f} | {r['valid_unique_mean']:.2f} |")
    lines += ['', '## 多轮与独立 joint 采样的配对差值', '',
              '同 checkpoint、同坐标、同 Q、同样 8 候选和同一电荷 DP。按去 h/q/键级骨架进行配对 cluster bootstrap 95% 区间（3,000 次），未作多重比较修正；采样种子只有一个。', '',
              '| σ | 子集 | 对照 | Mask 轮数 | Species coverage Δ (pp, 95% 区间) |',
              '|---|---|---|---:|---:|']
    rng = np.random.default_rng(20261002); differences = []
    for sigma in ev['args']['eval_noises']:
        for subset in ['geom_core', 'charge_challenge', 'nH']:
            def belongs(r):
                return r['has_nH'] if subset == 'nH' else r['domain'] == ('geom_core' if subset == 'geom_core' else 'diverse_charge')
            for baseline in ['independent_joint', 'B_catalogue']:
                base = {r['id']: r for r in ev['records'] if r['sigma'] == sigma and r['method'] == baseline and belongs(r)}
                variants = [('mask_joint', n) for n in ev['args']['rounds']]
                variants += [(m, 1) for m in ['B_then_HQ', 'B_then_HQ_catalogue'] if any(r['method'] == m for r in ev['records'])]
                for method, rounds in variants:
                    recs = [r for r in ev['records'] if r['sigma'] == sigma and r['method'] == method and r['rounds'] == rounds and belongs(r)]
                    diff = np.array([int(any(r['species_hits']))-int(any(base[r['id']]['species_hits'])) for r in recs])
                    group_values = {}
                    for r, d in zip(recs, diff):
                        group_values.setdefault(r['skeleton'], []).append(d)
                    sums = np.array([sum(v) for v in group_values.values()]); sizes = np.array([len(v) for v in group_values.values()])
                    inds = rng.integers(0, len(sums), (3000, len(sums)))
                    boot = sums[inds].sum(1)/sizes[inds].sum(1)*100
                    lo, hi = np.quantile(boot, [.025, .975]); delta = diff.mean()*100
                    label = f'mask-{rounds}r' if method == 'mask_joint' else method
                    lines.append(f'| {sigma:.2f} | {subset} | {baseline} | {label} | {delta:+.1f} ({lo:+.1f}, {hi:+.1f}) |')
                    differences.append({'sigma': sigma, 'subset': subset, 'baseline': baseline, 'rounds': rounds,
                                        'method': method, 'n_clusters': len(sums), 'delta_pp': delta, 'paired_95ci_pp': [lo, hi]})
    lines += ['', '## 电荷／氢诊断', '',
              '以下 first-draw 指标不是 coverage；电荷准确率不等于总电荷守恒。总电荷 100% 是 DP 的硬保证。', '',
              '| σ | 子集 | Greedy species | H 每原子准确率 | Q 每原子准确率 | 非零电荷 macro F1 |',
              '|---|---|---:|---:|---:|---:|']
    for r in ev['summary']:
        if r['method'] != 'greedy_joint' or r['subset'] not in ['geom_core', 'charge_challenge']:
            continue
        f1 = r['first_charged_atom_macro_f1']
        lines.append(f"| {r['sigma']:.2f} | {r['subset']} | {r['species_coverage']:.3f} | {r['first_h_atom_accuracy']:.3f} | {r['first_q_atom_accuracy']:.3f} | {f'{f1:.3f}' if f1 is not None else 'N/A (无非零标签)'} |")
    lines += ['', '## 边界与时延', '',
              '- 相比第一轮 B-only 的数据配方多了 1,500 个电荷训练样本；不能把新旧模型差全部归因于 joint mask。主要因果对照是同模型的 independent_joint 与 mask_joint。',
              '- 单一 reference 没有穷尽合法 species。多出候选不意味着正确后验、多种真解或热力学稳定；模型仍可能退化到单一模式。',
              '- B_catalogue 只使用训练局部状态频率，带小的全支持伪计数与粗粒度回退；不是既有硬 catalogue/MILP 方法的精确复现。',
              '- 这里总电荷 Q 给定。未知 Q 或无条件电荷推断尚未测试。',
              '- decoder 时间包含 CPU 电荷 DP，按 batch=32 摊销；不含一次性几何编码、输入传输、RDKit 与启动，不与未实测扩散模型比较。', '',
              '| σ | 方法 | 平均解码 ms/输入候选集 |', '|---|---|---:|']
    for r in ev['summary']:
        if r['subset'] == 'all':
            name = r['method']+(f"-{r['rounds']}r" if r['method'] == 'mask_joint' else '')
            lines.append(f"| {r['sigma']:.2f} | {name} | {r['decode_ms_per_input_set_all']:.2f} |")
    example_records = [r for r in ev['records'] if r['method'] in ['B_then_HQ', 'B_then_HQ_catalogue'] and r['valid_unique'] > 1]
    # Examples selected by inferred candidate diversity, not by true recovery.
    examples = [{'id': r['id'], 'sigma': r['sigma'], 'domain': r['domain'], 'Q': r['Q'],
                 'reference': r['reference_smiles'], 'valid_candidates': list(dict.fromkeys(s for s in r['candidate_species'] if s is not None))}
                for r in example_records[:30]]
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (RUN/'paired_differences.json').write_text(json.dumps(differences, indent=2), encoding='utf-8')
    (RUN/'candidate_examples.json').write_text(json.dumps(examples, indent=2), encoding='utf-8')
    print('Wrote', RUN/'RESULTS.md')
    for r in ev['summary']:
        if r['subset'] in ['geom_core', 'charge_challenge', 'nH']:
            print(r['sigma'], r['subset'], r['method'], r['rounds'], 'species=', round(r['species_coverage'], 4))


if __name__ == '__main__':
    main()
