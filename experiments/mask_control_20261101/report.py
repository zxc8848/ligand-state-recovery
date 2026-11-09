"""Human-readable report for the completed frozen supplementary study."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def fmt(v): return f'{100*v["mean"]:.2f} +/- {100*v["sd"]:.2f}'


def main():
    state=read(HERE/'STATUS.json')
    if state['stage']!='COMPLETE': raise ValueError(f'Study not complete: {state}')
    summary=read(HERE/'SUMMARY.json'); timing=read(HERE/'TIMING.json')
    protocol=read(HERE/'PROTOCOL.json')
    changed=[p for p,digest in protocol['source_sha256'].items() if sha(ROOT/p)!=digest]
    if changed: raise ValueError(f'Frozen supplementary input/source changed: {changed}')
    cells=list((HERE/'evaluations').rglob('evaluation.json'))
    if len(cells)!=27: raise ValueError(f'Need 27 complete cells, found {len(cells)}')
    caches={c:sha(HERE/'eval_features'/(c+'.pt')) for c in ('pose_clean','pose_noise010','ccd_ideal')}
    for cell in cells:
        meta=read(cell)
        if meta['feature_cache_sha256']!=caches[meta['condition']]:
            raise ValueError(f'Evaluation feature bytes differ: {cell}')
    check=subprocess.run([sys.executable,'-B',str(ROOT/'experiments/prob_mask_20261002/frozen_manifest.py')],
                         cwd=ROOT,check=True,text=True,capture_output=True)
    training=[]
    for seed in protocol['seeds']:
        pair=[]
        for arm in ('ordinary','masked'):
            directory=HERE/'training'/f'seed{seed}'/arm
            done=read(directory/'COMPLETE.json');hist=read(directory/'training.json')
            if sha(directory/'best.pt')!=done['best_checkpoint_sha256']: raise ValueError('Checkpoint changed')
            if done['protocol_sha256']!=sha(HERE/'PROTOCOL.json'): raise ValueError('Training protocol mismatch')
            selected=next(x for x in hist['history'] if x['epoch']==done['selected_epoch'])
            training.append({'seed':seed,'arm':arm,'selected_epoch':selected['epoch'],
                             'validation_CE':selected['val_bond_CE']+selected['val_atom_CE'],
                             'seconds':hist['history'][-1]['seconds']})
            pair.append((done['initial_state_sha256'],done['data_order_sha256']))
        if pair[0]!=pair[1]: raise ValueError('Initial states or data order differ')
    verification={'supplementary_source_and_inputs_unchanged':True,
                  'original_frozen_manifest':check.stdout.strip(), 'training':training,
                  'all_six_best_checkpoints_match_completed_hashes':True,
                  'all_27_cells_use_identical_feature_bytes_per_condition':True,
                  'paired_initial_states_and_data_orders_match':True,
                  'note':'No original locked inference was run by this study.'}
    (HERE/'FINAL_VERIFICATION.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    metrics=['native','strict','unified_resonance','smiles_species']
    names={'ordinary_marginal':'普通联合分类 + 同预算搜索',
           'masked_marginal':'mask 训练 + 相同边际搜索',
           'masked_conditional':'mask 训练 + 条件原子打分'}
    conds={'pose_clean':'真实 pose，clean','pose_noise010':'真实 pose，sigma=0.10','ccd_ideal':'CCD ideal'}
    lines=['# 配对 mask 训练对照研究（2026-11-01）','',
           '这是新增的开发性实验，不替换原稿模型或原锁定集结果。所有数值为三个训练种子的均值 +/- SD（%）。',
           '普通模型与 mask 模型同架构、同参数量、同随机初始权重、同数据顺序和优化步数；只训练轻量头，PaiNN 冻结。',
           '两组均从零训练，没有使用曾做过 mask 预训练的边子网，因此不是已发布热启动配方的完整反事实复现。','',
           '## 1. 完整结果','',
           '| 条件 | 配置 | 原生 B/h/q 身份 | 严格图 | 共振等价图 | 电荷盲图 SMILES |',
           '|---|---|---:|---:|---:|---:|']
    for c in conds:
        for mode in names:
            cell=summary['cells'][f'{c}/{mode}']
            lines.append('| '+' | '.join([conds[c],names[mode]]+[fmt(cell[e]) for e in metrics])+' |')
    lines+=['','原生身份与电荷盲图 SMILES 是不同端点，不能互相替代。所有组都使用同一个训练来源 catalogue、总电荷求解器和可构建性过滤，不使用 delocalized 输出改写。',
            '不同模型的候选内容可以不同，但候选算法/预算完全相同。mask 边际与条件读出则共享完全相同的候选图及边分数。','',
            '## 2. 原生身份的预先规定对照','',
            '| 条件 | 对照 | 平均差值 pp | 身份簇区间（种子平均）pp | 各种子 Holm p（72 项） |',
            '|---|---|---:|---|---|']
    for contrast in summary['contrasts']:
        if contrast['endpoint']!='native': continue
        ps=[t['holm_p_72'] for t in summary['tests'] if all(t[k]==contrast[k] for k in ('condition','old','new','endpoint'))]
        ci=contrast['seed_averaged_identity_cluster_ci95']
        label='训练效应（共同边际读出）' if contrast['old']=='ordinary_marginal' else '条件读出增量'
        lines.append(f'| {conds[contrast["condition"]]} | {label} | {100*contrast["mean_delta"]:+.2f} | '
                     f'[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] | '+', '.join(f'{p:.3g}' for p in ps)+' |')
    lines+=['','区间是在固定已训练模型下按身份簇重采样；三种子的训练变异另以 SD 报告。区间不等于覆盖整个模型训练随机性的置信区间。',
            '未显著不等于证明完全无效；不能从该有限预算实验推广到所有 mask 方法。新 72 项校正与原 78/108 项家族分开，不覆盖旧文件。','',
            '## 3. 保守无已知来源重叠子集（描述性）','',
            '| 条件 | n | 配置 | 原生 B/h/q | 严格图 | 共振等价图 | 图 SMILES |',
            '|---|---:|---|---:|---:|---:|---:|']
    for c in conds:
        for mode in names:
            v=summary['cells'][f'{c}/{mode}']['no_known_corpus_hit_sensitivity']
            lines.append('| '+' | '.join([conds[c],str(v['n']),names[mode]]+[fmt(v['metrics'][e]) for e in metrics])+' |')
    charge_profile={}
    for c in conds:
        rows=torch.load(HERE/'eval_features'/(c+'.pt'),map_location='cpu',weights_only=False)['rows']
        groups={'net_neutral_uncharged_atoms':[],'net_neutral_internal_charge':[],'net_charged':[]}
        for r in rows:
            group='net_charged' if r['Q'] else 'net_neutral_internal_charge' if bool((r['q']!=0).any()) else 'net_neutral_uncharged_atoms'
            groups[group].append(r['id'])
        charge_profile[c]={}
        for group,group_names in groups.items():
            charge_profile[c][group]={'n':len(group_names),'models':{}}
            for mode in names:
                rates=[]
                for seed in protocol['seeds']:
                    v=read(HERE/'evaluations'/f'seed{seed}'/mode/c/'verdicts_selected.json')
                    byname={r['name']:r for r in v['per_molecule']}
                    if group_names: rates.append(statistics.mean(byname[n]['native'] for n in group_names))
                charge_profile[c][group]['models'][mode]=({'mean':statistics.mean(rates),'sd':statistics.stdev(rates),'per_seed':rates} if rates else None)
    (HERE/'CHARGE_PROFILE.json').write_text(json.dumps(charge_profile,indent=2),encoding='utf-8')
    lines+=['','### 探索性电荷分层（clean pose，无新增显著性检验）','',
            '| 参考组 | n | 普通联合分类 | mask 边际 | mask 条件 |',
            '|---|---:|---:|---:|---:|']
    for group,v in charge_profile['pose_clean'].items():
        vals=[fmt(v['models'][mode]) if v['models'][mode] is not None else '--' for mode in names]
        lines.append('| '+' | '.join([group,str(v['n'])]+vals)+' |')
    lines+=['','分层仅依据参考状态：净电荷零不等于每个原子的形式电荷都为零。样本很少的组不能支持稳定的稀有电荷结论。完整三条件分层见 CHARGE_PROFILE.json。']
    audit=read(HERE/'ENCODER_SPLIT_IDENTITY_AUDIT.json')
    lines+=['','## 4. 必须修正的原稿事实','',
            '- 补全训练源实际是 3,000 GEOM + 1,500 Diverse 带电分子，共 13,500 个几何样本，不是纯 GEOM。',
            f'- 原冻结编码器缓存的实际划分审计：开发 pose 中有 {len(audit["conditions"]["pose_clean"]["train_hit_names"])} / 488 个身份在 train；'
            f'CCD 控制中有 {len(audit["conditions"]["ccd_ideal"]["train_hit_names"])} / 1000。开发全集不能再称训练身份无重叠。',
            '- 原缓存候选半径是 3.0 A，编码器径向 cutoff 为 2.5 A；当前补全缓存/推理候选图是 2.5 A，且 probe.geometry 不做 degree-15 截断。应分别描述。',
            '- 打分实际对 neural logits + catalogue log-prior 做 log-softmax。论文公式需保留依赖候选 B 的归一化项，或直接定义归一化后的乘积专家分布。',
            '- 当前解码器要求提供总电荷 Q；不能在没有缺失-Q实验的情况下把该输入称为可选。',
            '- 编码器训练身份审计依赖当前保存的原训练缓存；checkpoint 没有记录训练缓存哈希，因此报告中保留这项来源限制。',
            '- 锁定的 491 个参考结构对保存的完整来源排除集合，在两种 canonical key 下均为零命中。这里只核对参考身份，没有重新做锁定模型预测或评分。',
            '- 原稿锁定集的已有成绩没有被本实验替换；不能把新模型的开发结果贴到原锁定结果上。','',
            '## 5. 耗时（同一空闲 GPU，含失败样本）','',
            '| 配置 | 中位数 ms/分子 | P95 ms | 计时次数 | 不可构建次数 |',
            '|---|---:|---:|---:|---:|']
    for name,v in timing['summary'].items():
        lines.append(f'| {name} | {v["median_ms"]:.2f} | {v["p95_ms"]:.2f} | {v["timed_predictions"]} | {v["failed_predictions_including_repeats"]} |')
    lines+=['',f'设备：{timing["gpu"]}；PyTorch {timing["torch_version"]}。固定开发集前 50 个分子，每配置 5 个 warm-up，正式重复 3 次；CUDA 同步。',
            '计入候选图、编码、补全、catalogue/Q 求解、RDKit 构建与排序；不计载入权重、构建 catalogue 和参考评分。不能与未在同硬件重测的工具耗时作直接比较。','',
            '## 6. 已有锁定预测的错误转移审计','',
            '详见 HISTORICAL_ERROR_TRANSITIONS.json：三个种子分别修好 19/14/11 个图，破坏 5/4/7 个图；新增和修复的边错误都逐分子记录，包含假阳性。',
            '这支持整分子收益与局部错误数不下降可以同时存在，但不是错误预算守恒，也不是算法在分子之间转移错误。','',
            '## 7. 完整性与复现','',
            f'- {check.stdout.strip()}',
            '- 六个 best.pt 的 SHA256、两组初始状态和数据顺序全部核对通过。',
            '- 新协议与输入字节匹配；模型、评分和原稿冻结文件未更改。',
            '- 主输出：SUMMARY.json、RESULTS.md、TIMING.json、FINAL_VERIFICATION.json。',
            '- 运行：`D:\\minconda\\python.exe -B -u experiments\\mask_control_20261101\\study.py run --device cuda`。',
            '- 生成本报告：`D:\\minconda\\python.exe -B experiments\\mask_control_20261101\\report.py`。']
    lines+=['','## 8. 研究判断与建议','',
            '本组固定预算、匹配冷启动对照没有为 mask 训练提供稳定的正向证据。clean pose 的原生身份只增加约 0.48 pp，身份簇区间跨零；CCD 增加约 0.10 pp；加噪 pose 平均下降约 1.78 pp，且一个种子出现 Holm 校正后显著退化。',
            '额外条件原子打分的原生身份增量分别约 -0.34 / +0.14 / +0.07 pp，没有正向比较通过新 72 项校正。保守来源排除子集呈相同总体格局。',
            '因此不建议继续把 mask 训练或条件打分本身作为论文的核心创新依据。较简单的普通联合分类器加同预算候选搜索已经有竞争力。研究重点应转向真实无氢输入下的完整状态恢复、候选构建过滤及诚实的输入/评价边界。',
            '这不证明所有 mask 方法都无效，也不是原热启动发布配方的完整因果消融。不要据此直接替换原稿权重或把新模型的开发成绩冒充锁定集结果。',
            '投稿前优先纠正开发集身份无重叠、补全训练来源、候选半径/邻居上限和打分公式等事实描述。模型变复杂不能补救这些描述问题。']
    (HERE/'RESEARCH_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('REPORT COMPLETE:',HERE/'RESEARCH_REPORT.md')


if __name__=='__main__': main()
