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
- `dataloader.py`：读取 `.xyz` 文件，支持 `pair` 和 `simclr` 数据模式。
- `model.py`：定义 encoder、投影头和多种 contrastive loss。
- `utils.py`：seed、配置、checkpoint、环境信息工具。
- `egnn_torch_c.py`：本项目使用的 EGNN 实现。
- `train.py`：统一训练入口，自动支持单卡和 DDP。
- `ddp_train.py`：兼容 wrapper，内部调用 `train.py`。
- `eval.py`：固定 pair list 的 checkpoint 评估入口。
- `configs/default.yaml`：默认训练配置。
- `scripts/export_embeddings.py`：导出 embedding 和 metadata。
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

训练数据目录应包含 `.xyz` 文件。文件名需要包含 signature，例如：

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
- 当前默认元素 one-hot 支持：`H C N O F Li P S Cl Br`。

## 单卡训练

默认推荐使用 `simclr` mode：

```bash
python train.py \
  --config configs/default.yaml \
  --data_dir /path/to/xyz_data \
  --log_dir ./runs/ssr_exp1 \
  --mode simclr \
  --loss simclr
```

兼容旧的 signature pair 训练：

```bash
python train.py \
  --data_dir /path/to/xyz_data \
  --log_dir ./runs/ssr_pair \
  --mode pair \
  --loss bce_similarity
```

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

## 导出 Embedding

```bash
python scripts/export_embeddings.py \
  --data_dir /path/to/xyz_data \
  --ckpt ./runs/ssr_exp1/best.pt \
  --out_dir ./embeddings/ssr_exp1
```

输出：

- `embeddings.npy`
- `metadata.csv`

## 测试

```bash
python -m pytest tests/test_smoke.py -q
```

## 当前限制

- 默认 SimCLR positive 已升级为同结构增强视图；但时间邻近和物理相似性 positive 仍未实现。
- `pair` mode 中正负样本仍由 signature 定义，可能更偏向组成分类。
- 尚未加入周期性边界条件、分子身份、部分电荷、显式配位边等物理先验。
- 已有最小 smoke tests，但仍缺少完整单元测试和真实数据回归测试。
- 需要与 RDF、coordination number、SOAP、SchNet/DimeNet/PaiNN 等 baseline 做系统比较。

## 建议下一步

1. 实现 temporal mode：同一轨迹短时间窗口作为 positive。
2. 实现 physical mode：基于配位数/RDF/距离分布构造 soft positive。
3. 增加下游验证任务，例如配位数预测、RDF 状态分类、扩散/电导相关指标预测。
4. 建立 RDF/CN/SOAP/composition baseline。
