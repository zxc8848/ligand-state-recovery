# 冻结配置 v1（2026-10-04）—— **已作废，勿据此动锁定集**

> **接替者：`results/FROZEN_CONFIG_V2_2026-10-04.md`（2026-10-04 晚）。**
> 四处接口错误已全部修完并逐项验证，`verify_frozen_config.py` 现在核验的是 v2，
> `run_locked_eval.ps1` 的 `BONDNET_LOCKED_EVAL_INTERFACE_FIXED` 硬闸已拆除。
> 本文件保留为历史记录，**不要**据它跑任何东西。

> **2026-10-04 作废声明。** 外部审查发现、并经 `REVIEW_RESPONSE_2026-10-04.md`
> 独立验证：本文件第 4.1 节把原子态求解描述为"精确后缀 DP"并把整条方法称为
> "约束局部 MAP"，而 `constrained_atoms(greedy=True)` 先把 h 边缘化、再对 q 做
> sum-product 后缀表上的逐位贪心——**不是联合 MAP，且依赖原子顺序**（实测 2 原子
> 9.2% 次优、3 原子 14% 顺序翻转）。第 7 节的评估口径也写错了：
> `evaluate_species_recovery.py` 对双方都把形式电荷置零、氢数交给 RDKit 重推，
> 所以三个端点**都没有评价模型输出的 q 与 h**；Knodle 用的还是另一个等价定义。
>
> 因此这份文件不能作为冻结依据，`scripts/run_locked_eval.ps1` 已加硬闸。
> 修完 `REVIEW_RESPONSE_2026-10-04.md` 第 6 节四件事后重新冻结，届时
> `verify_frozen_config.py` 需补上接口与 scorer 的断言。
>
> 锁定测试集本身未被评估、sha256 不变，仍然可用。


动用锁定测试集 `data/pdbbind_pose_locked500.sdf` 之前，这份文件把**每一个会影响
预测的选择**写死。写死的含义是：在锁定集上读到任何数字之后，这份文件里的任何一项
都不得再改；若必须改，锁定集就作废，必须用新种子重抽一套。

本文件的数值全部从代码与 checkpoint 里逐项抄出，不凭记忆。核验方式见第 10 节。

---

## 1. 数据与划分

| 项 | 值 |
|---|---|
| 冻结 backbone 训练语料 | GEOM-DRUGS，`data/geom_drugs_all_random1_rel10.sdf`（269,739 分子） |
| backbone 划分键 | `geom_mol_idx`，`val_split = 0.1` |
| 补全网特征语料 | 带电子集 1,000 训练 / 150 验证；中性内部集 500 训练 / 100 验证 |
| 补全网特征噪声档 | σ ∈ {0.0, 0.1, 0.2} Å（`--noises`） |
| 输入模式 | heavy-only，`explicit_h = False`；输入只有 (Z, R, Q) |
| 开发集 | 488 PDBBind pose（`data/pdbbind_ccd_pose500.sdf` 的可用子集） |
| 理想化对照 | 1,000 CCD heavy-only（`data/ccd_heavy_leakagefree_manifest.csv`） |
| **锁定测试集** | **500 PDBBind pose，sha256 `fe032b5100e4b066faa4dbbd6a1b50ae6fbf7a8be73dbaa1476cc276abe160d2`** |

锁定集的排除口径：按 stereo 与 non-isomeric 两个规范 SMILES 键，排除 Diverse-80k、
GEOM-DRUGS、488 开发集及其 605 个 PDB entry / 587 个 CCD component。验证结果
0 命中。身份集：`data/locked_testset_excluded_identities.json`（323,532 / 321,280 键）。

## 2. 冻结的几何编码器（不再训练，不再调）

`checkpoints/revision_v3_one_stage_heavy_seed42/best_e2e.pt`，epoch 45。

| 项 | 值 |
|---|---|
| backbone | PaiNN，`num_interactions = 4`，`hidden_size = 360` |
| 截断半径 | `cutoff = 2.5` Å（`h_cutoff` 同） |
| 边嵌入维度 | `edge_embedding_size = 20` |
| 向量范数上限 | `vector_norm_limit = 3.0` |
| dropout | 0.1（训练时；推理 `dropout = 0`） |
| 训练噪声 | `noise_min = 0.0`，`noise_max = 0.15` Å，`noise_clean_prob = 0.0`，无 warmup |
| 训练超参 | lr 1e-4，50 epochs，batch 128，weight decay 1e-5，grad clip 1.0，seed 42 |
| epoch 选择噪声档 | `selection_noise_levels = [0.0, 0.1]` |
| 参数量 | 6,443,825 |

**注意：`noise_max = 0.15` 是第 3 节噪声表混淆的来源**——旧两阶段 Stage-2 只见过
0–0.1，Knodle 零噪声。σ=0 的 pose 行与干净 CCD 行是唯一不受此影响的两个条件。

## 3. 补全网 JointNet（唯一被训练的部件）

`experiments/prob_mask_20261002/joint.py` 的 `JointNet`。

| 项 | 值 |
|---|---|
| width / layers | 96 / 2 |
| 边解码器 | `probe.MaskedBondDecoder(input_dim=360, width=96, layers=3, context=True)` |
| 嵌入 | `zemb` 119×96、`aemb` 26×96、`bemb` 6×96；电荷特征 (Q, \|Q\|, Q/n) → 2 层 MLP |
| 边块输入 | `[h, n_i+n_j, \|n_i−n_j\|]`，3·96 → 96 |
| 节点块输入 | `[n, agg/deg, 键类 one-hot 计数(6)]`，2·96+6 → 96 |
| 输出头 | 键 5 类（零初始化）、原子 25 类 = (q+2)·5 + h |
| 参数量 | **388,499** |
| 热启动 | `pilot_v2/decoder_best.pt` 载入 `net.edge`，`strict=True` |
| 优化器 | AdamW，lr 2e-4，weight decay 1e-4，grad clip 2.0 |
| 训练 | 18 epochs，batch 32 |
| epoch 选择 | all-MASK 键 CE + 联合原子 CE 之和最小 |
| **训练种子** | **20261002 / 20261003 / 20261004，三套权重全部保留并全部上报** |
| 已发表权重 | `joint_v1/best.pt`，seed 20261002，epoch 13，sum CE 0.11626 |
| 三种子复制 | `joint_v5/seed<N>/best.pt`（配方逐字相同，只变 seed 与 outdir） |

### 3.1 训练期掩码与污染（`probe.masks` + `joint.corrupt_states`）

| 项 | 值 |
|---|---|
| 键掩码率 | 每分子 rate ~ U(0.2, 1.0)，20% 概率置 1.0 |
| 熵加权 | 逐边概率 clamp(rate·(0.5 + 归一化熵), 0.05, 1.0) |
| 原子入射掩码 | 30% 概率额外掩掉 15% 原子所连的全部边 |
| 键污染 | 未掩码边 5% 概率随机换成 0–4 中一类 |
| 原子掩码率 | 每分子 rate ~ U(0.2, 1.0)，30% 概率置 1.0 |
| 原子-星形联动 | 40% 概率把被掩原子所连的边一并置 MASK |
| 原子污染 | 未掩码原子 3% 概率随机替换 |

## 4. 推理：约束局部 MAP 解码（`pose_localmap.py`）

| 项 | 值 |
|---|---|
| 推理噪声 | `--sigma 0`（已扰动坐标条件下自身不再加噪） |
| 采样种子 | `probe.seed_all(11)`，`np.random.default_rng(11)` |
| 标记比例 | 最低置信 `fraction = 0.10` 的候选边 |
| 标记上限 | `max_flagged = 8` |
| 每边候选标签 | `topk_per_edge = 2`（边缘分布 top-2） |
| 束宽 | `beam = 48`，按 Σ log P(B) 剪枝 |
| 打分 | log P(B) + log P(h,q \| B) + catalogue 先验 |
| 硬过滤 | `build_species` 可建图（RDKit 可 sanitize、无自由基、单连通），优先级高于分数 |
| 原子态求解 | `constrained_atoms(..., greedy=True)` |
| batch | 16 |
| 新评估集的参考边表 | `--reference_from_sdf`：分子集与参考键表都取自 SDF 本身 |

`--reference_from_sdf` 是为锁定集新加的：488 开发集有归档逐边 dump 可以借边表，
锁定集没有。**已在 488 上双路验证**：同一 checkpoint、同一分子，两条路径给出的
1,171 条共享边在边集、`true` 标签、元素、预测上**零差异**（分子集差 2 个仅因
`--limit` 截断点不同）。


不含任何参考信息：目标函数与过滤器都不读参考标签、参考 SMILES 或真值键表。

### 4.1 电荷约束（`joint.constrained_atoms`）

批量后缀 DP，float64，逐行重缩放，**只强制 Σq = Q**（q ∈ {−2..+2}，25 类联合
(q,h) 词表）。不强制价态可行性——这一点是干净-CCD 劣势的候选解释之一，但已被数据
否掉（见 `NOISE_CCD_KNODLE_2026-10-04.md` 第 6 节），**不要**在论文里当作机制。

### 4.2 catalogue 软先验（`joint.make_catalogue` / `catalogue_logits`）

三级回退，全部由**训练划分**统计得到：

1. exact：(中心 Z, 排序后的 (邻居 Z, 键类) 多重集)
2. coarse：(中心 Z, 排序后的键类多重集)
3. by_z：中心 Z

伪计数 `1e-4`，归一化后取 log 加到原子 logits 上。只用预测邻接，不读任何原子目标值。

## 5. 输出规则：delocalized 改写（`pose_delocalize.py`）

末端氧基团若其**预测**多重集不等于训练 canonical 多重集，则整组改写为 class-5。
资格判定只用预测邻接。`min_support = 5`。表由 `joint_v1/sources.pt` 的**训练划分**统计得到，三个训练种子共用同一张表（`joint_v5` 用的就是 `joint_v1/features.pt`）。

| 中心 / 末端氧数 | 训练 canonical 多重集 | 纯度 |
|---|---|---:|
| C, 2 | (1,2) | 279/280 |
| N, 2 | (1,2) | 453/461 |
| P, 2 | (1,2) | 11/11 |
| P, 3 | (1,1,2) | 10/10 |
| **S, 2** | **(2,2)** | **673/673** |
| S, 3 | (1,2,2) | 21/21 |

S,2 的 canonical 是 (2,2) 这一条必须保留：早期把 (1,2) 当成普适 canonical 的朴素
规则使 species 从 0.7684 掉到 0.6496。

在 488 开发集上该规则只改 15 个分子、53 条边。

## 6. 被试过并**排除**在冻结配置之外的方案

每一项都重训或重解码过，效果都在噪声内或为负。论文应报告这些是阴性结果，而不是
悄悄不提——它们正是"解码侧耦合搜索是起作用的机制"这一主张的支撑。

| 方案 | 结果 | 账本 |
|---|---|---|
| motif 掩码 | resonance 0.8320 vs 基线 0.8381 | `MOTIF_MASK_RETRAIN_2026-10-03.md` |
| 化学感知污染 | 0.8361 vs 0.8381 | 同上 |
| 两者同时 | 0.8238 vs 0.8381 | 同上 |
| t 嵌入（tcond） | localmap species 0.8340，与基线相同 | `T_CONDITIONING_2026-10-04.md` |
| 余弦吸收态调度 + t（diffusion） | T=1 时 0.8402，在噪声内 | 同上 |
| 多步解码 T=2/4/8 | 基线单调下降 −1.4pp | 同上 |
| 学习型 verifier（both） | SMILES 0.7869 vs 0.7848，+1 个分子 | `VERIFIER_AND_DELOCALIZED_2026-10-03.md` |

**verifier 不进冻结配置**：它在论文口径下的收益是 1 个分子，且 `target=species`
变体在原生指标上 0.8525、在论文的 charge-blind scorer 下只有 0.5881/0.4570——
同一份预测。教训已记录：选择器必须在最终上报的端点上训练。

## 7. 评估口径（锁定集上不得更改）

| 端点 | 定义 | 脚本 |
|---|---|---|
| strict | 逐边与文件标签完全一致 | `archive/.../evaluate_species_recovery.py` |
| resonance-aware | `classify`，接受 class-5、芳香、末端氧基团的 {1,2} 置换编码 | 同上 |
| canonical-SMILES | 电荷归零后重建的规范 SMILES 相等 | 同上 |
| typed recall | 逐边分类召回 | `joint_pose/compare_old_new.py` |

统计口径：exact McNemar（配对）+ molecule bootstrap 95% CI（2,000 次重采样，
分子为单位）+ 骨架簇 bootstrap。**所有数字按三个训练种子报均值 ± SD，不报 best-of-3。**

对照方法：旧两阶段 + MILP（`archive/legacy_two_stage_20261003`）、Knodle
（`external/knodle/Knodle`，固定二进制）。两者都必须在**与新方法逐位相同的坐标**上
评估。

## 8. 随机种子清单

| 用途 | 种子 |
|---|---|
| backbone 训练 | 42（已冻结，不重训） |
| 补全网训练 | 20261002 / 20261003 / 20261004 |
| 推理采样 / 掩码 | 11（`pose_localmap.py` 固定） |
| bootstrap | 11（`compare_old_new.py`） |
| 锁定集抽样 | 20261004 |

## 9. 锁定集一次性协议

1. 这份文件定稿后不再改动。
2. 在锁定集上，用**三个训练种子**各跑一次 greedy / localmap / localmap+deloc，
   外加旧两阶段与 Knodle，全部在同一份坐标上。唯一允许的入口是
   `scripts\run_locked_eval.ps1`：它先跑本文件的核验，再比对 sha256，最后检查
   `results\LOCKED_EVAL_DONE.json` 是否已存在（存在即拒绝重跑）。
   Knodle 的二进制是静态链接 ELF x86-64 Linux 可执行文件（2019 年编译、stripped、
   无需任何库），Windows 上跑不了，那一段用 `scripts/run_knodle_locked.sh` 在任意
   x86-64 Linux 上跑（已在 10 个开发集 PDB 上实测：10/10 产出，约 0.09 s/分子），
   回来再 `-Only knodle-score`。Knodle 的输入是 CONECT-free PDB，不含参考键表；
   Knodle 失败的分子**故意**不补任何东西，adapt 阶段按错误计分。
3. 读到数字之后**不得**调任何超参、不得换选择器、不得改 delocalized 表。
4. 开发集数字照常进论文，标注为 development；锁定集数字标注为 confirmatory。
5. 若第 3 条被违反，锁定集降级为开发集，必须换种子重抽一套新的。

## 10. 核验

本文件每个数值的来源：

| 节 | 来源 |
|---|---|
| 1 | `data/pdbbind_pose_locked500_freeze.json`、`joint.py` 的 argparse 默认值 |
| 2 | `torch.load(best_e2e.pt)['args']` 实读 |
| 3 | `joint.py:212` `JointNet.__init__`、`joint_seeds_train.py`、`joint_v1/best.pt` 的 `args` 与参数量 |
| 3.1 | `probe.py:247` `masks`、`joint.py:259` `corrupt_states` |
| 4 | `pose_localmap.py` 的 argparse 默认值 |
| 4.1 | `joint.py:322` `constrained_atoms` |
| 4.2 | `joint.py:384` `make_catalogue` / `joint.py:401` `catalogue_logits` |
| 5 | `joint_v6/joint_v1/pdbbind_sigma0p00/pose_delocalize.json` 的 `training_canonical_multisets` |
| 6 | 对应账本的表格 |

重跑核验：

```powershell
python experiments\prob_mask_20261002\verify_frozen_config.py
```

通过后写出 `results/frozen_config_verification.json`，内含本文件的 sha256。
`run_locked_eval.ps1` 会先跑它；不通过就不会碰锁定集。

| 产物 | 内容 |
|---|---|
| `experiments/prob_mask_20261002/verify_frozen_config.py` | 本文件的逐项核验（75 项） |
| `scripts/run_locked_eval.ps1` | 一次性确认评估的唯一入口，带三道闸 |
| `scripts/run_knodle_locked.sh` | Knodle 那一段（Linux），输入输出都在 `work/paper2_locked_eval/` |
| `results/frozen_config_verification.json` | 核验通过记录 + 本文件 sha256 |
