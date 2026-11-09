# BondNet 固定配置 v3：评分与运行接口修复

本版本替代 v2 的执行接口、等价定义、统计和消融解释；v1/v2 保留为历史。
模型结构、训练、权重、候选邻域、catalogue、原子状态选择算法均不改动。
这不是新模型，也不是性能调参。旧表是旧评分定义下的历史结果，不覆盖。

## 1. 锁定集状态与研究边界

锁定输入为 `data/pdbbind_pose_locked500.sdf`，500 条；SHA256：
`fe032b5100e4b066faa4dbbd6a1b50ae6fbf7a8be73dbaa1476cc276abe160d2`。
划分 shuffle seed=20261004；构造细节沿用 v2 §1。

**检测到 locked_eval 已有部分输出，不能认证“从未评测”。** 本次修复不读取这些
结果、不删除、不覆盖；正式驱动遇到没有匹配 STARTED manifest 的历史输出会拒绝
执行。任何恢复必须先审查历史启动/暴露情况，不能仅因 DONE 不存在声称测试集未使用。
若模型/超参数因已知测试结果改变，必须另行设计新的独立确认实验；本修复不自动换集。

## 2. 不变的模型与解码配置

- 几何 backbone：`checkpoints/revision_v3_one_stage_heavy_seed42/best_e2e.pt`。
  PaiNN，hidden=360，4 layers，参数 6,443,825；训练配置沿用 v2 §2。
- JointNet：388,499 参数；种子 1002/1003/1004，对应
  `joint_v5/seed20261002/3/4/best.pt`；训练和 warm start 沿用 v2 §3。
- Catalogue 仅取 `joint_v1/sources.pt` 的 train split；不读评测标签作推理输入。
- 输入 `(Z,R,Q)`；边 no-bond/single/double/triple/aromatic，原子状态
  `state=(q+2)*5+h`，q=-2..2，h=0..4；class-5 是图后处理的 delocalized 约定。
- 邻域：fraction=0.10，max_flagged=8，topk_per_edge=2，beam=48；不扫参数。
- 原子解码默认原来的 sum-product suffix + sequential greedy；不是精确联合 MAP。
  Q 约束仍硬性保留；精确 MAP 可选实验不进入主配置。
- 排序 C：lexicographic `(good, sum log P(B)+ascore)`；ascore 为
  catalogue-adjusted atom logits 经 log_softmax 后选中状态的和。
- D 的训练多重集、min_support=5 均沿用 v2 §5，不改写原子状态。

## 3. 两个端点与统一 verdict

原生完整 `(B,h,q)` 身份：按序列化的预测状态独立重建，要求可建图且规范
non-isomeric SMILES 等于带 q/H 的参考。输出候选的 B、atom_states、SMILES 和
selected_indices，`native_ladder_stats.py` 重新核验，而非只信 correctness 布尔值。
无可比的历史基线原子状态；D 没有该端点，不制造 D 的原生数字。

三方法 head-to-head 为图恢复：strict、统一 resonance、historical charge-blind
canonical-SMILES，后者双方 q 归零、H 重推，不证明预测 q/H 正确。

版本 `bondnet.graph-verdicts.v3`；等价定义
`terminal-oxy-v2+validated-aromatic-components-v3`。先执行历史末端氧对齐：
相同多重集置换；两个末端 O 的 {1,2} 接受全 class-5；含双键的末端 O 组接受
工具的全 class-4 表示。后两者门控并不完全相同，不宣称任意 class-4/5 都等价。

环归一化只对 RDKit 确认芳香的完整芳香键连通系统。预测必须全芳香或全单/双键；
固定参考 q/H 做表示合法性核验，sanitize 后该系统仍芳香才归一。接受两个方向和
合法替代 Kekulé；拒绝非芳香环、部分芳香和非法累积双键；融合体系整体核验。
这些参考信息仅用于评估等价，不进入预测器。legacy classify 只作诊断。

## 4. 分母与统计

固定参考 SDF 名单；预先允许的元素沿用 v2，元素排除名单必须记录。坏参考/重复
名字/参考缺坐标/不完整参考边表均报错；缺失预测保留在分母中记为失败。方法与种子
的名字集合必须完全相同。locked 名单期待 n=500。不得用集合交集过滤失败样本。

所有均值、配对差、exact McNemar 和区间都来自上述同一逐分子 verdict。
bootstrap=2000、seed=11；逐分子与 identity-cluster CI 并列。按 non-isomeric
身份聚类，不声称做了 scaffold bootstrap。三个种子报均值±SD，不取最佳。

正式确认族预先固定 78 个检验：3 seeds × 4 selectors × 3 graph endpoints ×
2 comparators =72；另加 3 seeds ×2 native A→B/B→C contrasts =6。
fullgraph 为主；refedges 仅历史可比性/探索性分析，不进入确认族、不独立报显著性。
主流程先输出 42-test provisional Holm；Knodle 完成后必须用完整 78-test Holm。
开发集本次重评分的 6 conditions ×3 seeds ×2 comparators ×3 endpoints =108
个检验是回顾性探索族，与正式确认族区分，不能借结果删减比较项。

## 5. 组件解释

- A：单个 greedy B，其 q/H 由同一 conditional model/catalogue/Q 解码补全。
- B：同一 beam 内按 `(good,bscore)`，即首个可建图的最高边分数候选。
- C：同一 beam、同一补全和过滤，改按 `(good,bscore+ascore)`。
- D：C 的逐边输出表示改写，图端点可报，原生端点空缺。

A→B 是搜索+过滤总收益；同一池 search-only vs filtered 才隔离过滤。
B→C 只隔离原子态分数的排序收益，不是引入原子模型/catalogue/Q 的消融。
A 不过滤，B/C 共用过滤，不能写“四档同一硬过滤”。

## 6. 执行保障

`FROZEN_MANIFEST_V3_2026-10-04.json` 固定代码、检查点、catalogue 源数据、锁定
文件与本配置的 SHA256。正式入口开发集端到端冒烟、化学反例、缺失样本、名单
不匹配和统计一致性均须通过；源码正则只是补充，不能代替运行测试。

驱动先预检全部依赖和固定三种子，再于第一次真实执行前持久写 STARTED；
失败留下 FAILED。Resume 只接受相同 manifest；重复固定推理/评分本身不是泄漏，
但观察测试结果改配置是泄漏。历史部分输出缺少 v3 provenance 时拒绝自动恢复。
本次验证只跑开发集，不运行正式锁定评估。
