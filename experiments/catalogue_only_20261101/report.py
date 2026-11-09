"""Human-readable report; no changes to frozen selector or evaluation artifacts."""
import gzip
import json
from pathlib import Path
import statistics

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


summary = read(HERE / "SUMMARY.json")
seeds = (20261002, 20261003, 20261004)
conditions = ("pose_clean", "pose_noise010", "ccd_ideal")
arms = ("neural_catalogue", "catalogue_only")
labels = {"pose_clean": "真实 pose，clean", "pose_noise010": "真实 pose，σ=0.10",
          "ccd_ideal": "CCD ideal"}
metrics = ("native", "strict", "unified_resonance", "smiles_species")


def number(value):
    return f'{100*value["mean"]:.2f} ± {100*value["sd"]:.2f}'


lines = [
    "# 神经原子状态项与 catalogue-only 对照",
    "",
    "原主模型的三个种子；只做开发集，不改论文、原锁定结果或原统计家族。",
    "两组逐分子共享候选标签、候选顺序和边分数；catalogue-only 将神经原子 logits 置零，",
    "保留训练来源 catalogue、总电荷约束求解、归一化原子概率、可构建过滤与排序。",
    "键模型仍然是神经模型，因此这不是完全非神经基线，也不是取消原子头训练的重训消融。",
    "所有百分数是三个原始 head 种子的均值 ± 样本 SD。",
    "",
    "## 1. 主要结果",
    "",
    "| 条件 | 配置 | Native 身份 | 严格图 | 共振等价图 | 电荷盲图 SMILES |",
    "|---|---|---:|---:|---:|---:|",
]
for condition in conditions:
    for arm in arms:
        cell = summary["cells"][f"{condition}/{arm}"]
        lines.append("| " + " | ".join([labels[condition], arm] +
                     [number(cell[m]) for m in metrics]) + " |")

lines += ["", "## 2. Native 配对差值", "",
          "正数表示加入神经原子 logits 更好。区间按身份簇重采样，条件于这三个已训练模型。",
          "它不覆盖所有训练随机性；训练变异另以 SD 报告。",
          "",
          "| 条件 | 神经项增量 pp | 身份簇 95% CI pp | 三种子 Holm p（36项） |",
          "|---|---:|---|---|"]
for c in summary["contrasts"]:
    if c["endpoint"] != "native":
        continue
    tests = [t for t in summary["tests"] if t["condition"] == c["condition"] and t["endpoint"] == "native"]
    lo, hi = [100*x for x in c["identity_cluster_ci95"]]
    lines.append(f'| {labels[c["condition"]]} | {100*c["mean_neural_minus_catalogue"]:+.2f} | '
                 f'[{lo:+.2f}, {hi:+.2f}] | ' +
                 ", ".join(f'{t["holm_p_36"]:.5g}' for t in tests) + " |")
lines += ["", "36 项校正仅属于本次开发性对照，不替换原 78/108 项或 mask 研究的 72 项家族。",
          "", "## 3. 保守来源排除子集（描述性）", "",
          "| 条件 | n | 神经项＋catalogue Native | catalogue-only Native |",
          "|---|---:|---:|---:|"]
for condition in conditions:
    cells = [summary["cells"][f"{condition}/{arm}"]["source_excluded"] for arm in arms]
    lines.append(f'| {labels[condition]} | {cells[0]["n"]} | ' +
                 " | ".join(number(c["metrics"]["native"]) for c in cells) + " |")

lines += ["", "## 4. 建图成功与逐原子索引一致性（描述性）", "",
          "逐索引 q/h 一致性会惩罚共振等价原子交换，不能代替 Native 身份。",
          "",
          "| 条件 | 配置 | 可构建 | 全分子 q 逐索引全对 | h 逐索引全对 | (q,h) 逐索引全对 |",
          "|---|---|---:|---:|---:|---:|"]
for condition in conditions:
    for arm in arms:
        cell = summary["cells"][f"{condition}/{arm}"]
        lines.append("| " + " | ".join([labels[condition], arm] +
                     [number(cell[m]) for m in ("constructible", "q_exact", "h_exact", "atom_state_exact")]) + " |")

lines += ["", "## 5. 电荷分层与选图变化（探索性，无新增检验）", "",
          "| 条件 | 参考电荷组 | n | 神经项＋catalogue Native | catalogue-only Native |",
          "|---|---|---:|---:|---:|"]
strata_report = {}
selection_report = {}
historical_report = {}
for condition in conditions:
    blob = torch.load(ROOT/"experiments/mask_control_20261101/eval_features"/f"{condition}.pt",
                      weights_only=False, map_location="cpu")
    grouped = {"neutral_atoms": [], "internal_charge": [], "net_charged": []}
    for row in blob["rows"]:
        group = "net_charged" if row["Q"] else ("internal_charge" if (row["atom_target"]//5 != 2).any() else "neutral_atoms")
        grouped[group].append(row["id"])
    flags = {}
    for seed in seeds:
        for arm in arms:
            v = read(HERE/"evaluations_v3"/f"seed{seed}"/condition/f"verdicts_{arm}.json")
            flags[seed, arm] = {r["name"]: r for r in v["per_molecule"]}
    strata_report[condition] = {}
    for group, names in grouped.items():
        if not names:
            continue
        vals = {a: [statistics.mean(flags[s,a][n]["native"] for n in names) for s in seeds] for a in arms}
        strata_report[condition][group] = {"n": len(names), "per_seed": vals}
        formatted = [f'{100*statistics.mean(vals[a]):.2f} ± {100*statistics.stdev(vals[a]):.2f}' for a in arms]
        lines.append("| " + " | ".join([labels[condition], group, str(len(names))] + formatted) + " |")
    transitions = []
    historical = []
    for seed in seeds:
        with gzip.open(HERE/"evaluations_v3"/f"seed{seed}"/condition/"predictions.json.gz",
                       "rt", encoding="utf-8") as handle:
            records = json.load(handle)["records"]
        counts = {"seed": seed, "n": len(records), "same_selected_B": 0,
                  "neural_native_gain_same_B": 0, "neural_native_loss_same_B": 0,
                  "neural_native_gain_changed_B": 0, "neural_native_loss_changed_B": 0}
        audit = {"seed": seed, "n": len(records)}
        for key in ("same_selected_B", "same_selected_buildability",
                    "same_selected_native", "same_unordered_candidate_outcomes"):
            audit["mismatches_"+key] = sum(not r["historical_audit"][key] for r in records)
        score_diffs = [r["historical_audit"]["max_unordered_score_difference"] for r in records]
        audit["max_score_difference_where_signatures_match"] = max(
            (v for v in score_diffs if v is not None), default=None)
        historical.append(audit)
        for r in records:
            n, c = [r[a] for a in arms]
            same = n["labels"] == c["labels"]
            counts["same_selected_B"] += same
            suffix = "same_B" if same else "changed_B"
            counts[f"neural_native_gain_{suffix}"] += n["native"] and not c["native"]
            counts[f"neural_native_loss_{suffix}"] += c["native"] and not n["native"]
        transitions.append(counts)
    selection_report[condition] = transitions
    historical_report[condition] = historical

lines += ["", "小样本带电分层不能支持稳定的稀有电荷结论。Native 是完整分子身份，不是原子电荷准确率。",
          "选图变化的逐种子计数保存于 DIAGNOSTICS.json；相同最终 B 下的 Native 差异来自构建原子状态。",
          "不同 B 下的结果同时涉及状态分配、可构建过滤和重排，不作单一机制归因。",
          "",
          "## 6. 完整性与边界",
          "",
          "- 只运行开发集；无新训练、无锁定推理、无论文结果替换。",
          "- 历史输出一致性见各单元 COMPLETE.json 与预测内 historical_audit；不据此筛样本。",
          "- 旧档未保存完整候选向量，且数值不可逐值复现；对照使用本次同一次计算的完整模型伙伴。",
          "- 预检差异、协议版本和已完成单元迁移见 PREFLIGHT_DIAGNOSTIC.md 与迁移收据。",
          "- 两个对照分支的候选向量和边分数逐字相同，并随预测一起保存。",
          "- 不使用 delocalized 改写。Native 身份比较非立体 canonical SMILES，非逐索引 (B,h,q) 完全一致。",
          "- 这是推理时原子-logit 消融，不能证明原子监督在训练阶段可有可无。",
          "- FINAL_VERIFICATION.json 记录原 162 个冻结文件和本次输入/协议核验。"]
lines += ["", "## 7. 研究判断", "",
          "该配对对照支持：在当前固定候选搜索/电荷求解框架下，推理时的神经原子状态项有实质价值。",
          "这不等于所有 mask 训练都有效，也不等于该模块优于所有可能的化学规则或原子态枚举方法。",
          "",
          "之前 C−B 阶梯只改变是否将原子概率分数用于候选排序；B 档本来就用神经原子状态建图。",
          "因此 C−B 增量较小，不能解释成整个原子头没有作用。本次同时移除它对状态分配与评分的贡献，",
          "测量的是不同的组件效应；也不能把所有 Native 收益单独归为电荷准确率提升。",
          "",
          "Native 身份与图 SMILES 不是同一端点：加噪 pose 和 CCD 的电荷盲图 SMILES 均未随 Native 同步改善。",
          "净带电与内部电荷分离的分层样本很少；clean pose 的内部电荷分离组反而退化，不能声称所有电荷子类均获益。",
          ""]
for condition in conditions:
    tests = [t for t in summary["tests"] if t["condition"] == condition and t["endpoint"] == "native"]
    count = sum(t["holm_p_36"] < .05 for t in tests)
    lines.append(f'- {labels[condition]}：Native 的三个逐种子对照中 {count}/3 通过 36 项 Holm 校正。')
verification = read(HERE/"OUTPUT_VERIFICATION.json")
total_builds = sum(c["selected_native_rebuild_checks"] for c in verification["cells"])
total_states = sum(c["candidate_charge_and_score_checks"] for c in verification["cells"])
lines += ["", f"独立核验通过：{total_builds:,} 次选中分子重建，"
          f"{total_states:,} 次候选电荷/分数组合检查；所有固定总体均保留。",
          "论文文件尚未修改；这些是新增的开发性证据，不是新的锁定集成绩。"]
save = {"strata": strata_report, "selection_transitions": selection_report,
        "historical_archive_audit": historical_report}
(HERE/"DIAGNOSTICS.json").write_text(json.dumps(save, indent=2, ensure_ascii=False), encoding="utf-8")
(HERE/"RESEARCH_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
print("Wrote RESEARCH_REPORT.md and DIAGNOSTICS.json")
