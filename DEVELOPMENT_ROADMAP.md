# SSR 成熟化开发路线图

## 当前实现状态

截至本轮开发，阶段 0、阶段 1A 和阶段 2 的核心代码已经落地：

- 阶段 0：已新增配置文件、seed、完整 checkpoint、固定 eval pair list、环境信息保存和 smoke tests。
- 阶段 1A：已新增同结构增强视图、`SimCLRDataset` 和标准 SimCLR loss。
- 阶段 2：已接入 `pair` / `simclr` dataset mode，以及 `nt_xent`、`simclr`、`bce_similarity`、`triplet_margin` loss 选择。

尚未完成的是阶段 1B 的 temporal positive、阶段 1C 的 physical positive、hard negative mining 的高级策略，以及阶段 4 之后的系统评估。

阶段 3 的 xyz 可计算部分也已经落地：

- `element_shell` 特征模式：元素 one-hot + Li 标记 + 最近 Li 距离 + 第一壳层标记。
- `center_on_li`：以 Li 质心为中心平移坐标。
- `shell_radius`：可选 Li-centered crop。
- `compute_physical_descriptors.py`：导出配位数和 Li 距离统计。

阶段 3 中仍需要额外数据的部分包括 partial charge、force-field atom type、molecule id、真实拓扑边、周期性 box/PBC。

阶段 4 和阶段 5 的基础版本已经落地：

- `evaluation.py`：统一评估函数，支持 pair ROC-AUC、signature probe、coordination probe 和 composition/dummy baseline。
- `scripts/run_evaluation_suite.py`：输出 `metrics.json`、`metrics.csv`、`embeddings.npy`、`metadata.csv`。
- `scripts/run_experiment.py`：统一训练、评估、描述符导出，并保存 `commands.json`。
- `scripts/make_tiny_xyz.py`：生成 CPU smoke test 用的小样本 xyz 数据。

后续仍建议补充 RDF/SOAP baseline、真实物性下游任务、UMAP/t-SNE 可视化和 CI。

## 目标

将当前 SSR 从“基于 signature 的 EGNN 对比学习原型”逐步升级为一个可复现、可评估、能服务溶剂化结构分析和结构-性质建模的研究平台。

成熟版本应满足：

- 学到的 embedding 主要反映局部溶剂化几何和物理状态，而不是只反映组成 signature。
- 训练任务、损失函数、数据增强和采样策略相互一致。
- 有明确 baseline、消融实验和下游任务验证。
- 训练、评估、配置、日志、checkpoint 和测试流程可复现。

## 总体原则

1. 先修正任务定义，再优化模型复杂度。
2. 先证明 embedding 有物理意义，再扩展大规模训练。
3. 每个阶段都要有可运行命令、可量化指标和可复现实验记录。
4. 不把 signature ROC-AUC 作为唯一成功标准，它只能作为辅助指标。

## 阶段 0：稳定当前代码基线

### 目标

让当前代码能够稳定训练、评估和复现实验，作为后续改造的可靠起点。

### 开发任务

- [ ] 增加 `requirements.txt` 或 `pyproject.toml` 并锁定主要依赖版本。
- [ ] 增加统一配置文件，例如 `configs/default.yaml`。
- [ ] 增加 `--seed`，统一设置 `random`、`numpy`、`torch`、DataLoader worker seed。
- [ ] checkpoint 保存完整状态：
  - model state
  - optimizer state
  - epoch
  - best metric
  - config/args
- [ ] 增加确定性 eval pair list，避免每次评估随机变化。
- [ ] 增加 smoke tests：
  - DataLoader 能读取最小 `.xyz` 数据。
  - Model forward 输出 shape 正确。
  - Loss 在全正、混合正负、无正样本下行为明确。
  - Eval 能加载 checkpoint。

### 验收标准

- `python -m py_compile *.py` 通过。
- 最小测试数据上可以完成 1 个 epoch 训练和一次评估。
- 相同 seed 下 eval 结果一致。
- checkpoint 可恢复训练。

## 阶段 1：重构自监督任务

### 目标

把训练目标从“同组成 signature 拉近”升级为“真实结构相似性拉近”。

### 推荐路线 A：同结构增强视图 + SimCLR

这是最建议优先实现的版本，因为定义清晰、实现简单、物理上稳健。

开发任务：

- [ ] 为每个结构生成两个 augmented views。
- [ ] 实现 3D 合法增强：
  - 随机旋转
  - 随机平移
  - 小幅坐标噪声
  - 可选 atom dropout
  - 可选局部坐标 jitter
- [ ] 实现标准 SimCLR / NT-Xent：
  - 每个 anchor 有且只有一个 positive。
  - batch 内其他样本作为 negatives。
  - 不再依赖随机 pair label。
- [ ] 保留 signature 作为评估标签，而不是默认训练标签。

验收标准：

- batch 内每个样本都有明确 positive。
- loss 不会因为无正样本产生 NaN。
- 在相同结构增强视图上 cosine similarity 明显高于随机结构。

### 推荐路线 B：时间邻近 positive

适合有 MD trajectory 信息时使用。

开发任务：

- [ ] 从文件名或元数据中解析 trajectory id 和 frame index。
- [ ] 定义时间窗口，例如 `|t1 - t2| <= k` 为 positive。
- [ ] 远时间片或不同 trajectory 作为 negative。
- [ ] 支持 hard negative：组成相同但时间/结构状态不同。

验收标准：

- 同一轨迹邻近结构 embedding 更接近。
- embedding 能平滑反映 trajectory 演化，而不是只按 signature 分组。

### 推荐路线 C：物理相似性 positive

适合有配位数、RDF、距离分布或专家标签时使用。

开发任务：

- [ ] 计算每个结构的物理描述符：
  - Li 配位数
  - Li-O / Li-F / Li-N 距离统计
  - 第一溶剂化壳层组成
  - RDF bins
- [ ] 根据描述符距离定义 soft positive。
- [ ] 实现 supervised contrastive 或 soft contrastive loss。
- [ ] 支持 hard negative mining。

验收标准：

- embedding 距离与物理描述符距离正相关。
- 聚类结果能对应可解释的溶剂化状态。

## 阶段 2：损失函数与采样体系成熟化

### 目标

让采样策略、标签定义和损失函数严格一致。

### 开发任务

- [ ] 增加多种 dataset mode：
  - `pair`
  - `simclr`
  - `temporal`
  - `physical`
- [ ] 增加多种 loss：
  - `nt_xent`
  - `supervised_contrastive`
  - `bce_similarity`
  - `triplet_margin`
- [ ] 增加 hard negative mining：
  - 同 signature 不同配位状态
  - 不同 signature 但 RDF 相似
  - embedding 空间中距离过近的负样本
- [ ] 增加 batch sampler，保证每个 batch 有足够正样本和负样本。

### 验收标准

- 每种 mode/loss 有单元测试。
- 训练日志中记录正负样本比例、hard negative 比例、有效 positive 数量。
- 不同 loss 的结果可在统一 eval pipeline 下比较。

## 阶段 3：加入物理先验和更强结构表示

### 目标

让模型显式利用电解液/溶剂化体系中的关键物理信息。

### 开发任务

- [ ] 增加 atom-level 特征：
  - 元素 one-hot
  - force-field atom type
  - partial charge
  - molecule type
  - molecule id
- [ ] 增加 edge-level 特征：
  - 距离
  - 是否在 cutoff 内
  - 是否属于 Li-X 关键配位边
  - 分子内/分子间边类型
- [ ] 支持周期性边界条件：
  - box 信息读取
  - minimum image convention
  - cutoff neighbor graph
- [ ] 支持以 Li 为中心截取局部壳层。
- [ ] 增加 molecule-level pooling 和 ion-centered pooling。

### 验收标准

- 可在同一数据集上比较：
  - 仅元素 + 坐标
  - 加 molecule type
  - 加 charge
  - 加显式配位边
  - 加 PBC
- 加入物理先验后，下游任务指标有稳定提升或可解释性改善。

## 阶段 4：建立严谨评估体系

### 目标

证明 SSR embedding 不是只学组成，而是对结构和物性任务有价值。

### 必做评估

- [ ] Signature classification：辅助观察组成可分性。
- [ ] Coordination number prediction：验证局部配位信息。
- [ ] Solvation shell composition prediction：验证壳层结构。
- [ ] RDF/state clustering consistency：验证结构状态表达。
- [ ] Few-shot downstream prediction：验证预训练迁移价值。

### 推荐下游任务

根据数据可用性选择：

- 扩散系数预测
- 电导率预测
- Li+ 迁移数预测
- 溶剂化自由能预测
- 去溶剂化能预测
- 界面反应前态识别
- 异常结构检测

### Baseline

必须至少比较：

- composition-only baseline
- coordination number / RDF descriptor
- SOAP 或 ACSF
- MLP on handcrafted features
- 非等变 GNN
- EGNN without contrastive pretraining
- 随机初始化 encoder

可选比较：

- SchNet
- DimeNet / DimeNet++
- PaiNN
- NequIP / Allegro 类等变模型

### 消融实验

- [ ] 无坐标，仅元素/组成。
- [ ] 无物理先验。
- [ ] 无数据增强。
- [ ] 不同 positive 定义。
- [ ] 不同 loss。
- [ ] 不同 cutoff / neighbor 数。
- [ ] 不同 pooling 策略。

### 验收标准

- SSR 在至少一个结构任务和一个下游任务上显著优于 composition-only baseline。
- SSR 至少与 RDF/CN baseline 持平，最好在少样本场景有优势。
- 聚类或可视化结果能对应可解释的溶剂化状态。

## 阶段 5：实验管理与工程化

### 目标

让实验可追踪、可复现、可比较。

### 开发任务

- [ ] 引入配置管理：
  - YAML config
  - CLI override
  - config 自动保存到 log_dir
- [ ] 标准化输出目录：
  - checkpoints
  - tensorboard logs
  - metrics JSON/CSV
  - embeddings
  - plots
- [ ] 增加实验记录：
  - git commit hash
  - data version
  - config hash
  - random seed
  - environment info
- [ ] 增加 embedding export：
  - `.npy`
  - `.csv`
  - metadata JSON/CSV
- [ ] 增加可视化脚本：
  - UMAP/t-SNE
  - similarity heatmap
  - clustering report
- [ ] 增加 CI 或本地测试命令。

### 验收标准

- 任一实验可以通过 config 复现。
- 所有评估结果保存为机器可读格式。
- embedding 可以独立导出并用于外部分析。

## 阶段 6：面向应用的扩展

### 目标

将 SSR 从方法验证推进到具体科学或工程应用。

### 应用方向 A：MD 轨迹分析工具

功能：

- 批量读取 trajectory snapshots。
- 导出 embedding。
- 自动聚类溶剂化状态。
- 标记代表性结构。
- 识别稀有状态和状态转移。

验收：

- 能在一条真实 MD 轨迹上输出可解释状态图。
- 聚类中心结构能被领域规则解释。

### 应用方向 B：电解液配方预筛选

功能：

- 对不同盐/溶剂/浓度体系生成结构 embedding。
- 与扩散、电导、迁移数等指标关联。
- 支持少样本 property predictor。

验收：

- 在 held-out 配方上预测趋势优于 composition baseline。

### 应用方向 C：结构检索与异常检测

功能：

- 给定目标结构，检索相似溶剂化构型。
- 找出偏离常见状态的 rare events。
- 支持界面反应或分解前态筛选。

验收：

- 检索结果在关键配位几何上相似。
- 异常结构能被距离、配位或专家规则验证。

## 推荐开发顺序

1. 阶段 0：稳定代码基线。
2. 阶段 1A：实现同结构增强视图 + 标准 SimCLR。
3. 阶段 4 的最小评估：CN/RDF/signature baseline。
4. 阶段 1B/1C：加入时间邻近或物理相似性 positive。
5. 阶段 3：加入物理先验。
6. 阶段 5：完善实验管理。
7. 阶段 6：选择一个应用方向做端到端验证。

## 近期最小可执行任务清单

建议下一轮开发优先完成：

1. 新增 `augment.py`，实现旋转、平移、坐标噪声。
2. 新增 `SimCLRDataset`，每个结构返回两个 augmented views。
3. 重写 `nt_xent` 为标准 SimCLR loss。
4. 增加固定 eval pair list。
5. 新增最小测试数据和 `tests/test_smoke.py`.
6. 新增 `configs/default.yaml`.
7. 新增 `scripts/export_embeddings.py`.

完成后，SSR 才能从“能跑的原型”进入“可系统验证的方法平台”。
