"""Diff rescored development verdicts; never reads locked result values."""
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NEW = ROOT/'work/evaluation_fix_v3/development'
OLD = HERE/'joint_v6/unified'


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    conditions = json.loads((NEW/'unified_resonance_summary.json').read_text())['conditions']
    changes = []
    for condition in conditions:
        for name in ('old', 'knodle', 'new_seed20261002', 'new_seed20261003', 'new_seed20261004'):
            op, np = OLD/condition/f'{name}.json', NEW/'unified'/condition/f'{name}.json'
            old = json.loads(op.read_text())
            new = json.loads(np.read_text())
            om = {r['name']: r for r in old['per_molecule']}
            nm = {r['name']: r for r in new['per_molecule']}
            if set(om) != set(nm): raise ValueError('Old/new scorer populations differ')
            changed = [{'name': n, 'old': om[n]['unified_resonance'], 'new': nm[n]['unified_resonance']}
                       for n in om if om[n]['unified_resonance'] != nm[n]['unified_resonance']]
            changes.append({'condition': condition, 'system': name, 'n': len(nm), 'changed': changed})
    (NEW/'scorer_changes.json').write_text(json.dumps(changes, indent=1), encoding='utf-8')
    lines = ['# v3 开发集重评分与差异清单', '',
             '模型与预测不变，只修复评分定义和统计输入。旧结果保留；正式锁定集未运行。', '',
             '| 条件 | Knodle | 旧模型 | 新模型 mean±SD | 新−旧 pp |',
             '|---|---:|---:|---:|---:|']
    for c, d in conditions.items():
        v = d['new_per_seed']
        lines.append(f'| {c} | {d["knodle"]:.4f} | {d["old"]:.4f} | '
                     f'{st.mean(v):.4f}±{st.stdev(v):.4f} | {100*(st.mean(v)-d["old"]):+.2f} |')
    lines += ['', '## 逐分子评分改变（v3 相对 v2）', '',
              '| 条件 | 系统 | 由错变对 | 由对变错 |', '|---|---|---:|---:|']
    for d in changes:
        if d['changed']:
            gained = sum(r['new'] for r in d['changed'])
            lines.append(f'| {d["condition"]} | {d["system"]} | {gained} | {len(d["changed"])-gained} |')
    lines += ['', '全部样本 ID 和方向见同目录 `scorer_changes.json`。',
              '配对率、McNemar 和两套 CI 现都来自同一版本逐分子 flags；',
              'Holm 见 `holm_development.json`（完整 108 个回顾性检验），不借显著性删减比较项。']
    (NEW/'RESCORE_V3.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__': main()
