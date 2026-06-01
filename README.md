# SSR: Solvation Structure Representation

SSR 是一个用于溶剂化结构表征学习的研究原型。当前实现使用 `.xyz` 原子结构文件作为输入，通过 EGNN 编码 3D 原子坐标和元素特征，并用对比学习训练结构 embedding。

## 技术路线

当前流程：

1. 从 `.xyz` 文件读取原子元素和 3D 坐标。
2. 默认使用同一结构的两种增强视图构造 SimCLR positive pair。
3. 用 EGNN 编码每个结构，经过投影头得到归一化 embedding。
4. 用标准 SimCLR / NT-Xent 损失训练。
5. 用固定 pair list 评估 signature 正负样本 cosine similarity 和 ROC-AUC。

仍保留 `pair` mode，用于兼容基于 signature 的正负样本训练。更成熟的路线应继续加入时间邻近、配位数、RDF 或局部溶剂化壳层状态等物理相似性定义。

更完整的技术评估见 [TECHNICAL_ROUTE_ASSESSMENT.md](TECHNICAL_ROUTE_ASSESSMENT.md)，代码层面的改进记录见 [IMPROVEMENT_SUGGESTIONS.md](IMPROVEMENT_SUGGESTIONS.md)。

## 文件结构

- `augment.py`：结构增强，包括随机旋转、平移、坐标噪声和 atom dropout。
- `physics.py`：从 xyz 坐标计算 Li 壳层、配位和距离相关物理先验。
- `dataloader.py`：读取 `.xyz` 文件，支持 `pair`、`simclr` 和 `temporal` 数据模式。
- `model.py`：定义 encoder、投影头和多种 contrastive loss。
- `utils.py`：seed、配置、checkpoint、环境信息工具。
- `egnn_torch_c.py`：本项目使用的 EGNN 实现。
- `train.py`：统一训练入口，自动支持单卡和 DDP。
- `ddp_train.py`：兼容 wrapper，内部调用 `train.py`。
- `eval.py`：固定 pair list 的 checkpoint 评估入口。
- `configs/default.yaml`：默认训练配置。
- `scripts/export_embeddings.py`：导出 embedding 和 metadata。
- `scripts/compute_physical_descriptors.py`：从 xyz 导出配位数和 Li 距离统计。
- `scripts/run_evaluation_suite.py`：运行 pair ROC、signature probe、coordination probe 和 baseline 对比。
- `scripts/run_experiment.py`：统一训练、评估、embedding 导出、描述符导出、可视化和命令记录。
- `scripts/visualize_embeddings.py`：导出 PCA embedding scatter 和 similarity heatmap SVG。
- `scripts/run_local_checks.sh`：本地编译和 smoke test 命令。
- `scripts/make_tiny_xyz.py`：生成 CPU smoke test 用的小样本 xyz 数据。
- `tests/test_smoke.py`：最小 smoke tests。
- `ratio_embedding/ratio_data_generation.ipynb`：ratio embedding 数据生成实验 notebook。

## 环境依赖

建议使用独立 Python 环境。核心依赖包括：

```bash
pip install -r requirements.txt
```

如果 `rdkit` 通过 pip 安装失败，可改用 conda：

```bash
conda install -c conda-forge rdkit
```

## 数据格式

训练数据目录可以直接包含 `.xyz` 文件，也可以按电解液配方 id 分目录递归组织。推荐后者：

```text
data/
  formula_001/
    Frame100_Li_2DMC_2EC_2EMC_id1030.xyz
  formula_002/
    Frame100_Li_2DMC_2EC_2EMC_id1030.xyz
```

不同配方目录中允许出现相同 `.xyz` 文件名；代码会使用完整路径区分样本，并在导出的 metadata 中记录 `formulation_id`。文件名需要包含 signature，例如：

```text
Frame100_Li_2DMC_2EC_2EMC_id1030.xyz
```

当前 signature 解析规则会提取：

```text
Li_2DMC_2EC_2EMC
```

`.xyz` 文件格式示例：

```text
12
signature: Li_2DMC_2EC_2EMC
Li 0.000 0.000 0.000
O  1.923 0.102 0.031
C  2.615 0.812 0.004
```

注意：

- 数据目录必须至少包含两个不同 signature，否则无法构造负样本。
- 每个原子行至少需要 4 列：元素、x、y、z。
- 当前默认元素 one-hot 支持常见电解液/电池界面元素：`H Li B C N O F Na Mg Al Si P S Cl K Ca Br I`。
- 默认 `atom_phys_shell` 特征 = 元素 one-hot + 归一化原子物性 + Li shell 特征。原子物性包括 atomic number、atomic mass、Pauling electronegativity、covalent radius、vdW radius、group、period 和 valence electrons。

## 单卡训练

默认推荐使用 `simclr` mode：

```bash
python train.py \
  --config configs/default.yaml \
  --data_dir /path/to/xyz_data \
  --log_dir ./runs/ssr_exp1 \
  --mode simclr \
  --loss simclr \
  --feature_mode atom_phys_shell \
  --center_on_li
```

兼容旧的 signature pair 训练：

```bash
python train.py \
  --data_dir /path/to/xyz_data \
  --log_dir ./runs/ssr_pair \
  --mode pair \
  --loss bce_similarity
```

时间邻近训练：

```bash
python train.py \
  --data_dir /path/to/xyz_data \
  --log_dir ./runs/ssr_temporal \
  --mode temporal \
  --loss bce_similarity \
  --temporal_positive_window 5 \
  --temporal_negative_min_gap 50 \
  --feature_mode atom_phys_shell \
  --center_on_li
```

`temporal` mode 要求每个 `.xyz` 能解析出 formulation id、trajectory id、中心 Li id 和 frame index。`formulation_id` 默认来自 `.xyz` 所在的配方目录名。推荐在第二行 metadata 中显式写入：

```text
signature: Li_2DMC_2EC_2EMC trajectory: TrajA center_id: 1030 frame: 100
```

或在文件名中使用：

```text
TrajA_Frame100_Li_2DMC_2EC_2EMC_id1030.xyz
```

其中 `id1030` 表示中心 Li 离子的 id。temporal positive 定义为同一 formulation、同一 trajectory、同一中心 Li id 内满足：

```text
temporal_min_lag <= |frame_i - frame_j| <= temporal_positive_window
```

negative 优先从同 signature 的远时间片中采样：

```text
|frame_i - frame_j| >= temporal_negative_min_gap
```

如果同 signature 远时间片不存在，则退到不同 trajectory 的样本。这样可以降低模型只学习 composition/signature 差异的风险。

训练会保存：

```text
./runs/ssr_exp1/best.pt
./runs/ssr_exp1/config.json
./runs/ssr_exp1/env.json
```

如需排查梯度，可开启：

```bash
python train.py --data_dir /path/to/xyz_data --debug_grad
```

## 多卡训练

```bash
torchrun --nproc_per_node=4 ddp_train.py \
  --config configs/default.yaml \
  --data_dir /path/to/xyz_data \
  --log_dir ./runs/ssr_ddp \
  --batch_size 32 \
  --num_workers 4
```

`train.py` 和 `ddp_train.py` 都会根据 `WORLD_SIZE` 和 `LOCAL_RANK` 启用 DDP，并使用 `DistributedSampler`。

## 评估

```bash
python eval.py \
  --data_dir /path/to/xyz_data \
  --ckpt ./runs/ssr_exp1/best.pt \
  --batch_size 64 \
  --seed 42
```

输出包括：

- 正样本平均 cosine similarity
- 负样本平均 cosine similarity
- ROC-AUC

评估使用固定 pair list，由 `--seed` 和 `--max_pairs_per_anchor` 控制，便于复现实验。

## 评估套件

阶段 4 的评估套件会输出：

- pair ROC-AUC 和正负 pair 平均 cosine similarity
- SSR embedding 的 signature linear probe
- composition-only signature baseline
- SSR embedding 的 coordination number regression probe
- composition-only coordination baseline
- RDF descriptor 和 ACSF-like radial descriptor baseline
- random / untrained EGNN encoder baseline
- solvation shell state classification
- RDF/state clustering consistency
- 可选 few-shot downstream probe
- dummy baseline

```bash
python scripts/run_evaluation_suite.py \
  --data_dir /path/to/xyz_data \
  --ckpt ./runs/ssr_exp1/best.pt \
  --out_dir ./runs/ssr_exp1/evaluation \
  --device cpu \
  --feature_mode atom_phys_shell \
  --center_on_li
```

如果有下游标签 CSV，可额外传入：

```bash
python scripts/run_evaluation_suite.py \
  --data_dir /path/to/xyz_data \
  --ckpt ./runs/ssr_exp1/checkpoints/best.pt \
  --out_dir ./runs/ssr_exp1/metrics \
  --downstream_csv ./labels.csv \
  --downstream_target conductivity \
  --device cpu \
  --feature_mode atom_phys_shell \
  --center_on_li
```

下游 CSV 可用 `path`、`filename` 或 `signature` 匹配样本。

输出：

- `metrics.json`
- `metrics.csv`
- `embeddings.npy`
- `embeddings.csv`
- `metadata.csv`
- `rdf_descriptors.npy`
- `acsf_like_descriptors.npy`
- `run_metadata.json`

## 端到端实验

阶段 5 的实验入口会记录命令、训练 checkpoint、评估结果和物理描述符：

```bash
python scripts/run_experiment.py \
  --data_dir /path/to/xyz_data \
  --log_dir ./runs/ssr_exp1 \
  --config configs/default.yaml \
  --device cpu
```

生成 CPU 小样本数据并快速测试：

```bash
python scripts/make_tiny_xyz.py --out_dir ./tmp/tiny_xyz
python scripts/run_experiment.py \
  --data_dir ./tmp/tiny_xyz \
  --log_dir ./runs/tiny_cpu \
  --epochs 1 \
  --batch_size 2 \
  --device cpu \
  --num_workers 0
```

标准输出目录：

- `checkpoints/`：`best.pt`、`config.json`、`env.json`
- `metrics/`：机器可读评估指标、metadata、RDF/ACSF-like 描述符
- `embeddings/`：独立 embedding `.npy` / `.csv` 和 metadata
- `plots/`：PCA scatter SVG 和 cosine similarity heatmap SVG

## 导出 Embedding

```bash
python scripts/export_embeddings.py \
  --data_dir /path/to/xyz_data \
  --ckpt ./runs/ssr_exp1/best.pt \
  --out_dir ./embeddings/ssr_exp1 \
  --feature_mode atom_phys_shell \
  --center_on_li
```

输出：

- `embeddings.npy`
- `metadata.csv`
- `embeddings.csv`
- `metadata.json`

## 物理描述符

普通 xyz 文件已经足够计算一部分重要物理量：

- Li 配位数，基于 `--li_cutoff`。
- 每个原子到最近 Li 的距离。
- 是否位于 Li 第一溶剂化壳层。
- 以 Li 为中心的平移归一化。
- 给定半径内的 Li-centered crop。

导出描述符：

```bash
python scripts/compute_physical_descriptors.py \
  --data_dir /path/to/xyz_data \
  --out_csv ./descriptors.csv \
  --li_cutoff 2.5
```

仅靠普通 xyz 通常不能可靠获得：

- force-field atom type
- partial charge
- molecule id / residue id
- 分子内键拓扑
- 周期性 box 和 PBC minimum image 距离

这些需要额外的拓扑文件、力场参数、轨迹元数据或带 lattice/box 信息的扩展 xyz。

## 测试

```bash
python -m py_compile *.py scripts/*.py
python -m pytest tests/test_smoke.py -q
```

或：

```bash
scripts/run_local_checks.sh
```

## 当前限制

- 默认 SimCLR positive 已升级为同结构增强视图；时间邻近 positive 已实现为 `temporal` mode；物理相似性 positive 仍未实现。
- `pair` mode 中正负样本仍由 signature 定义，可能更偏向组成分类。
- 已加入常见电解液元素 identity、原子物性编码和 xyz 可计算的 Li 壳层特征；周期性边界条件、分子身份、部分电荷、显式拓扑边仍需要额外数据。
- 已有最小 smoke tests，但仍缺少完整单元测试和真实数据回归测试。
- 已有 RDF、coordination number、ACSF-like radial descriptor、composition baseline；真实 SOAP、非等变 GNN、SchNet/DimeNet/PaiNN 等 baseline 仍待补充。

## 建议下一步

1. 实现 physical mode：基于配位数/RDF/距离分布构造 soft positive。
2. 补充真实数据回归测试，覆盖 B/Si 和多中心 Li trajectory 数据。
3. 增加真实下游验证任务，例如扩散/电导/粘度/迁移数预测。
4. 补充真实 SOAP、非等变 GNN、SchNet/DimeNet/PaiNN 等 baseline。
