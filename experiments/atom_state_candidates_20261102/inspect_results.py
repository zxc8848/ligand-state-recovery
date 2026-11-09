"""Additional descriptive analysis of the completed, fixed-budget development run."""
import collections
import gzip
import json
from pathlib import Path
import statistics

import numpy as np
import torch
from rdkit import Chem, RDLogger

import run as R

HERE = R.HERE


def main():
    RDLogger.DisableLog('rdApp.*'); R.check_inputs()
    planhash = R.S.sha(HERE/'RUN_PLAN.json')
    dataset = json.loads((HERE/'DATASET.json').read_text())
    audit = json.loads((HERE/'DATASET_AUDIT.json').read_text())
    assert not audit['conflicts'] and audit['dev_heldback_skeleton_intersection'] == 0
    features = {r['id']: r for r in torch.load(HERE/'features.pt', weights_only=False, map_location='cpu')}
    names = sorted(features)
    held = {r['id'] for r in torch.load(HERE/'held_back.pt', weights_only=False, map_location='cpu')}
    assert not set(names) & held
    data = {}; total_assignments = 0; outcomes = {}; pairs = []
    for seed in R.S.SEEDS:
        folder = HERE/f'evaluations/seed{seed}'
        receipt = json.loads((folder/'DONE.json').read_text())
        assert receipt['plan_sha256'] == planhash and receipt['sha256'] == R.S.sha(folder/'predictions.json.gz')
        with gzip.open(folder/'predictions.json.gz', 'rt') as f: payload = json.load(f)
        assert payload['plan_sha256'] == planhash
        rs = payload['records']; assert len(rs) == len({r['name'] for r in rs}) == len(names)
        data[seed] = {r['name']: r for r in rs}
        assert set(data[seed]) == set(names)
        for r in rs:
            feat = features[r['name']]
            assert r['target_smiles'] == feat['smiles'] and r['Q'] == feat['Q']
            for method in R.METHODS:
                entry = r['methods'][method]; expected = []; seen = set()
                for i, a in enumerate(entry['raw']):
                    assert len(a['atom_states']) == len(feat['z'])
                    assert all(0 <= x < 25 for x in a['atom_states'])
                    q = [x//5-2 for x in a['atom_states']]
                    assert sum(q) == r['Q']
                    if a['positive_mass_level'] is not None:
                        assert sum(max(v, 0) for v in q)-max(r['Q'], 0) == a['positive_mass_level']
                    # Recompute identity from stored predicted SMILES independently
                    # of the stored native flag (full graph rebuild is in run.py).
                    assert a['native'] == bool(a['constructible'] and a['smiles'] == feat['smiles'])
                    if a['constructible']:
                        mol = Chem.MolFromSmiles(a['smiles']); assert mol is not None
                        assert sum(x.GetFormalCharge() for x in mol.GetAtoms()) == r['Q']
                        if a['smiles'] not in seen and len(expected) < 8:
                            seen.add(a['smiles']); expected.append(i)
                    total_assignments += 1
                assert entry['returned_indices'] == expected and len(entry['raw']) <= 32
                if r['baseline']['native']: assert R.hit(r, method, 1)
            if r['bucket'] == 'separated_neutral':
                pairs.append({'seed': seed, 'name': r['name'], 'target_smiles': r['target_smiles'],
                              'baseline_native': r['baseline']['native'], 'strict_selected_B': r['strict_B'],
                              'plain_native8': R.hit(r, 'plain_kbest', 8),
                              'mass_native8': R.hit(r, 'mass_stratified', 8),
                              'plain_returned': len(r['methods']['plain_kbest']['returned_indices']),
                              'mass_returned': len(r['methods']['mass_stratified']['returned_indices'])})
    for group in ('pooled', *R.S.GROUPS):
        ns = [n for n in names if group == 'pooled' or data[R.S.SEEDS[0]][n]['bucket'] == group]
        if not ns: continue
        rng = np.random.default_rng(20261102)
        draws = rng.integers(0, len(ns), (2000, len(ns)))
        cell = {'n': len(ns), 'methods': {}, 'contrasts': {}, 'counts_per_seed': {}}
        for method in R.METHODS:
            hitarr = np.array([[int(R.hit(data[s][n], method, 8)) for n in ns] for s in R.S.SEEDS])
            cell['methods'][method] = {'coverage8_mean': float(hitarr.mean()), 'head_sd': float(hitarr.mean(1).std(ddof=1)),
                'returned_species_mean': statistics.mean(len(data[s][n]['methods'][method]['returned_indices']) for s in R.S.SEEDS for n in ns),
                'raw_attempts_min_max': [min(len(data[s][n]['methods'][method]['raw']) for s in R.S.SEEDS for n in ns),
                                         max(len(data[s][n]['methods'][method]['raw']) for s in R.S.SEEDS for n in ns)],
                'median_proposal_seconds': statistics.median(data[s][n]['methods'][method]['seconds'] for s in R.S.SEEDS for n in ns)}
        for label, control in (('mass_minus_plain', 'plain_kbest'), ('mass_minus_single', 'baseline')):
            diff = np.array([[int(R.hit(data[s][n], 'mass_stratified', 8)) -
                (int(data[s][n]['baseline']['native']) if control == 'baseline' else int(R.hit(data[s][n], control, 8)))
                for n in ns] for s in R.S.SEEDS])
            averaged = diff.mean(0)
            cell['contrasts'][label] = {'delta': float(averaged.mean()),
                'pointwise_fixed_three_heads_paired_molecule_bootstrap_ci95': np.quantile(averaged[draws].mean(1), [.025, .975]).tolist(),
                'rescued_per_head': (diff == 1).sum(1).tolist(), 'harmed_per_head': (diff == -1).sum(1).tolist(),
                'training_uncertainty_included_in_CI': False}
        for s in R.S.SEEDS:
            cell['counts_per_seed'][str(s)] = {'single': sum(data[s][n]['baseline']['native'] for n in ns),
                **{m: sum(R.hit(data[s][n], m, 8) for n in ns) for m in R.METHODS},
                'strict_selected_B': sum(data[s][n]['strict_B'] for n in ns)}
        outcomes[group] = cell
    R.check_inputs()
    out = {'status': 'descriptive development analysis; no new hypothesis tests', 'plan_sha256': planhash,
           'input_summary_sha256': R.S.sha(HERE/'SUMMARY.json'), 'data_audit_sha256': R.S.sha(HERE/'DATASET_AUDIT.json'),
           'analysis_script_sha256': R.S.sha(Path(__file__)), 'all_assignments_checked': total_assignments,
           'held_back_prediction_identity_hits': 0, 'cells': outcomes, 'separated_cases': pairs}
    R.S.save(HERE/'ANALYSIS.json', out)
    lines = ['# 固定模型联合(q,h)候选：分析账本', '',
             '这是新身份开发集上的预先固定预算试验，不是最终确认实验。没有训练或调参。两种候选算法共享32个原始赋值、8个返回species的上限；实际尝试次数可能少于上限，见ANALYSIS.json。', '',
             '| 组 | n | 单解正确数/种子 | Plain@8正确数/种子 | Mass@8正确数/种子 | Mass−Plain pp（逐分子区间，固定3头） |',
             '|---|---:|---|---|---|---:|']
    for group, c in outcomes.items():
        counts = list(c['counts_per_seed'].values())
        d = c['contrasts']['mass_minus_plain']; lo, hi = d['pointwise_fixed_three_heads_paired_molecule_bootstrap_ci95']
        lines.append('| '+ ' | '.join([group, str(c['n']), '/'.join(str(x['single']) for x in counts),
            '/'.join(str(x['plain_kbest']) for x in counts), '/'.join(str(x['mass_stratified']) for x in counts),
            f"{100*d['delta']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]"])+' |')
    lines += ['', '区间不包含训练随机性的总体不确定性；只是固定三个头、逐分子配对重采样的点态探索性区间，无多重比较校正，不作确认性显著性声明。不能把3×n当独立分子。', '',
              '## 保留的限制', '',
              '- 所有方法固定为当前模型选定的B，不能修正B候选本身没选中的连接/键级错误。strict B计数只作索引标签诊断，不当作native理论上限。',
              '- mass分层在所有组使用同一规则，不知道哪个分子来自电荷分离组；但它是启发式多样性分配，不是校准后验采样。',
              '- baseline优先保留使候选覆盖率相对单解不下降成为设计保证。应重点看mass相对同上限预算plain的增量以及其它组损失。',
              '- 可构建不等于物理合理、真实质子化或预测正确；这轮没有外部实验状态标签。',
              '- 280个held-back分子未推理（其中34个电荷分离中性）。在开发策略选定之前不读取其模型成绩。', '',
              '## 复现与核验', '',
              '运行run.py复用已完成头并核验全部原始赋值；运行inspect_results.py核验集合选择、总电荷、native标签并生成本账本。DATASET_AUDIT.json为独立完整排除检查。',
              f'本次检查{total_assignments}个赋值；模型输出中留存集身份命中0。', '']
    (HERE/'ANALYSIS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Analysis saved; assignments checked:', total_assignments, '; held-back identity hits: 0', flush=True)


if __name__ == '__main__': main()
