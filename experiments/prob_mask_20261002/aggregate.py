"""Three sampling seeds on identical validation geometries; NOT three training seeds."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    run = HERE/'pilot_v2'
    files = [run/'evaluation.json', run/'eval_seed_20261003.json', run/'eval_seed_20261004.json']
    evs = [json.loads(p.read_text(encoding='utf-8')) for p in files]
    sigma = .2
    keys = [(r['method'], r['rounds']) for r in evs[0]['summary'] if r['sigma'] == sigma]
    rng = np.random.default_rng(20261002)
    rows = []; matrices = {}
    for key in keys:
        perseed = []
        ids0 = None
        metrics = []
        for ev in evs:
            records = sorted([r for r in ev['records'] if r['sigma'] == sigma and (r['method'], r['rounds']) == key], key=lambda r: r['id'])
            ids = [r['id'] for r in records]
            if ids0 is None:
                ids0 = ids
            assert ids0 == ids
            perseed.append([r['strict_hit'] for r in records])
            metrics.append(next(r for r in ev['summary'] if r['sigma'] == sigma and (r['method'], r['rounds']) == key))
        matrix = np.array(perseed, dtype=float)
        matrices[key] = matrix
        rows.append({'method': key[0], 'rounds': key[1], 'mean_coverage': matrix.mean(),
                     'sampling_seed_std': matrix.mean(1).std(ddof=1),
                     'decoder_ms': np.mean([r['decoder_ms_per_molecule_set'] for r in metrics]),
                     'budget': metrics[0]['budget']})
    base = matrices[('decoder_independent', 0)]
    for r in rows:
        mat = matrices[(r['method'], r['rounds'])]
        diff = (mat-base).mean(0)
        boot = diff[rng.integers(0, len(diff), (4000, len(diff)))].mean(1)
        r['paired_difference'] = diff.mean()
        r['paired_bootstrap_95ci'] = np.quantile(boot, [.025, .975]).tolist()
    summary = {'n_molecules': base.shape[1], 'sampling_seeds': [e['arguments']['seed'] for e in evs],
               'noise_sigma': sigma, 'rows': rows,
               'note': 'One trained checkpoint, three sampling seeds, SAME 300 noisy geometries. Do not pool into n=900.'}
    (run/'aggregate.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = ['# 自主探索记录：概率 mask 键图候选生成', '',
             '日期：2026-10-02。论文、正式模型与原始数据未改动。', '',
             '## 已完成', '',
             '- 冻结修正方向后的 heavy-only 单阶段 BondNet，缓存几何表征。',
             '- 3,000 个训练身份，300 个验证身份；按原 GEOM 分组切分，排除 test，并剔除 train/val canonical identity 重叠。',
             '- 两个小模型各训练 12 epoch：基础 mask 补全头 148,181 参数；加局部状态统计的版本 158,933 参数。',
             '- 每个分子固定 8 个原始候选；比较独立采样、1/2/4/8 轮揭示、低置信度重新 mask。',
             '- 第二版在同一份坐标上复测 3 个**采样**种子，不是 3 个训练种子。',
             '- 无向边对称、邻边依赖、MASK 与 no-bond 区分、不同轮数完成、推理不读参考标签等测试通过。', '',
             '## 0.2 Å 噪声下的严格整图覆盖', '',
             '| 方法 | 候选数 | 平均覆盖率 | 采样种子 SD | 相对补全头独立采样 Δ（配对 bootstrap 95% 区间） | 解码 ms/输入候选集 |',
             '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        name = r['method']+(f"-{r['rounds']}r" if r['method'] in ['mask', 'remask'] else '')
        lo, hi = r['paired_bootstrap_95ci']
        lines.append(f"| {name} | {r['budget']} | {r['mean_coverage']*100:.1f}% | {r['sampling_seed_std']*100:.1f}pp | {r['paired_difference']*100:+.1f}pp ({lo*100:+.1f}, {hi*100:+.1f}) | {r['decoder_ms']:.2f} |")
    lines += ['', '## 如何解释', '',
              '1. 小型残差补全头确实提高了这批带噪验证分子的预测与候选覆盖。但原 checkpoint 训练噪声上限为 0.15 Å，本探针加入 0.2 Å，因此收益同时包含噪声域适配，不能归因于 mask 创新。',
              '2. 多轮 mask 必须和同一补全头的一次独立采样比；不能只与旧模型相比。单轮 mask 与独立采样在共同随机种子下逐候选完全一致，已由断言核验。',
              '3. 配对区间按 300 个分子做 bootstrap，先在同一分子的三个采样种子上取平均；不把重复坐标虚增为 900 个独立样本。这些区间是未做多重比较校正的探索性描述。',
              '4. 时延是 batch=32 的摊销 GPU 解码时间（8 个候选），不含几何编码/传输/RDKit，不能直接当单样本延迟，也未实测完整扩散对照。几何缓存 val=900 个输入共约 2.08 s。', '',
              '## 重要边界：一对多问题还没有完整解决', '',
              '- 当前输出只有 B；缺氢导致的不确定性常在 **(B,q,h)** 的联合状态，而非仅在键级。中性 GEOM 上的 noisy graph coverage 不是这个问题的最终验证。',
              '- 用真值重原子键图、固定中性原子和 RDKit 默认隐式氢，34/300 个分子仍不能恢复 reference species；这 34 个全含 `[nH]`。因此 species/validity 列只保留为默认补氢诊断，不作为新模型核心结果。',
              '- 所有新实验仅用开发验证集；不能直接写成独立测试证据。单一真值也不枚举全部合理 species，采样质量不等于校准后的后验概率。',
              '- 一次小规模 clean 旋转审计：16 个分子 hard graph 均未改变，最大单类概率差约 0.0143；不足以证明严格旋转不变性。', '',
              '## 建议的下一步（尚未实施）', '',
              '先把原子 H-count/charge 状态作为 mask token 与边状态联合补全，加入芳香 N 等明确歧义的最小测试集，再与同预算独立采样和现有 catalogue/MILP 候选生成比较。',
              '不建议现在扩大纯边 mask 训练规模，或把这组带噪改善直接当成论文的新核心贡献。', '',
              '复现命令与协议见 README.md；逐分子记录见 pilot_v2/evaluation.json 和两份 eval_seed JSON。', '']
    (HERE/'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote', HERE/'SUMMARY.md')
    for r in rows:
        print(r['method'], r['rounds'], 'coverage=', round(r['mean_coverage'], 4),
              'delta=', round(r['paired_difference'], 4),
              'paired95ci=', [round(x, 4) for x in r['paired_bootstrap_95ci']])


if __name__ == '__main__':
    main()
