"""Three sampling seeds, fixed model/geometries, clustered paired intervals."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent/'joint_v1'


def main():
    files = ['block_evaluation.json', 'block_seed_20261003.json', 'block_seed_20261004.json']
    evs = [json.loads((HERE/f).read_text(encoding='utf-8')) for f in files]
    methods = ['independent_joint', 'B_catalogue', 'B_then_HQ', 'B_then_HQ_catalogue']
    rng = np.random.default_rng(20261002); output = []
    lines = ['# 联合状态：三个采样种子复核', '',
             '同一 epoch-13 checkpoint，同一 0.2 Å 扰动坐标，8 个原始候选。三个采样种子不是三个训练种子，不把 550 个分子当成 1,650 个独立样本。', '',
             '| 子集 | 方法 | 平均 species coverage | 采样种子 SD | 闭壳层连通有效率 | 有效 species 数/输入 |',
             '|---|---|---:|---:|---:|---:|']
    matrices = {}
    for subset in ['geom_core', 'charge_challenge', 'nH']:
        for method in methods:
            recsets = []
            for ev in evs:
                recs = sorted([r for r in ev['records'] if r['sigma'] == .2 and r['method'] == method and
                               (r['has_nH'] if subset == 'nH' else r['domain'] == ('geom_core' if subset == 'geom_core' else 'diverse_charge'))], key=lambda r: r['id'])
                recsets.append(recs)
            ids = [r['id'] for r in recsets[0]]
            assert all(ids == [r['id'] for r in rs] for rs in recsets)
            mat = np.array([[any(r['species_hits']) for r in rs] for rs in recsets], dtype=float)
            matrices[(subset, method)] = (mat, recsets[0])
            valid = np.mean([r['valid_count']/8 for rs in recsets for r in rs])
            unique = np.mean([r['valid_unique'] for rs in recsets for r in rs])
            row = {'subset': subset, 'method': method, 'n': len(ids), 'coverage_mean': float(mat.mean()),
                   'sampling_seed_std': float(mat.mean(1).std(ddof=1)),
                   'closed_shell_valid_fraction': float(valid), 'valid_unique_mean': float(unique)}
            output.append(row)
            lines.append(f"| {subset} (n={len(ids)}) | {method} | {mat.mean()*100:.1f}% | {row['sampling_seed_std']*100:.1f}pp | {valid*100:.1f}% | {unique:.2f} |")
    lines += ['', '## 同骨架配对差值', '',
              '先对同一分子的三个采样种子取平均，再按去 h/q/键级骨架进行 4,000 次配对 cluster bootstrap。区间未做多重比较校正，只是开发性描述。', '',
              '| 子集 | 方法 | 对照 | Δ species coverage (pp, 95% 区间) |', '|---|---|---|---:|']
    deltas = []
    for subset in ['geom_core', 'charge_challenge', 'nH']:
        for method in ['B_then_HQ', 'B_then_HQ_catalogue']:
            for baseline in ['independent_joint', 'B_catalogue']:
                mat, recs = matrices[(subset, method)]; base, _ = matrices[(subset, baseline)]
                diff = (mat-base).mean(0)
                groups = {}
                for r, d in zip(recs, diff):
                    groups.setdefault(r['skeleton'], []).append(d)
                sums = np.array([sum(v) for v in groups.values()]); sizes = np.array([len(v) for v in groups.values()])
                ix = rng.integers(0, len(sums), (4000, len(sums)))
                boot = sums[ix].sum(1)/sizes[ix].sum(1)*100
                lo, hi = np.quantile(boot, [.025, .975]); delta = diff.mean()*100
                deltas.append({'subset': subset, 'method': method, 'baseline': baseline, 'n_clusters': len(groups),
                               'delta_pp': float(delta), 'cluster_paired_95ci_pp': [float(lo), float(hi)]})
                lines.append(f'| {subset} | {method} | {baseline} | {delta:+.1f} ({lo:+.1f}, {hi:+.1f}) |')
    lines += ['', '## 结论边界', '',
              '- 多轮并行 mask 不是当前收益来源；先 B、再条件补全 (h,q) 的分块采样与局部状态先验应分开消融。',
              '- 已观察开发验证结果后选择了 block 方案；上述区间不等于预注册确认性显著性检验。',
              '- catalogue 是训练局部频率＋全支持小伪计数的软对照，不等价于历史硬 catalogue/MILP 方法。',
              '- Q 给定且由 DP 强制保持；未知 Q 未测试。没有训练标签支持的 q=±2、h=4 不能声称可泛化。',
              '- 骨架隔离仅针对这轮补全训练集。冻结 backbone 的完整训练语料重叠未重新审计。',
              '- 候选 sanitize/闭壳层有效不等于热力学稳定，reference coverage 也不是完整后验校准。', '',
              '逐分子输出见 block_evaluation.json 和 block_seed_*.json；完整单种子噪声曲线、负结果与时延见 RESULTS.md。', '']
    (HERE/'MULTISEED.md').write_text('\n'.join(lines), encoding='utf-8')
    (HERE/'aggregate.json').write_text(json.dumps({'sampling_seeds': [e['args']['seed'] for e in evs],
                                                'rows': output, 'paired_differences': deltas}, indent=2), encoding='utf-8')
    print('Wrote', HERE/'MULTISEED.md')
    for row in output:
        print(row['subset'], row['method'], 'coverage', round(row['coverage_mean'], 4))


if __name__ == '__main__':
    main()
