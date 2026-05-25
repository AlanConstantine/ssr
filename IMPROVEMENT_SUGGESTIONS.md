# SSR 代码学习与改善建议

## 当前代码概览

该目录实现的是一个基于 EGNN 的溶剂化结构对比学习流程：

- `dataloader.py`：读取 `.xyz` 结构文件，按溶剂组成签名构造正负样本对，并输出 padded batch。
- `model.py`：封装 `EGNN_NetworkC`、投影头和 `nt_xent` 对比损失。
- `egnn_torch_c.py`：自定义 EGNN 网络实现，支持坐标更新、邻居选择和可选全局注意力。
- `train.py`：单卡训练入口，包含 TensorBoard、early stopping 和 best checkpoint 保存。
- `ddp_train.py`：分布式训练入口，但目前与 dataloader 接口存在不一致。
- `eval.py`：加载 checkpoint 后计算正负样本 cosine similarity 和 ROC-AUC。
- `ratio_embedding/ratio_data_generation.ipynb`：用于 ratio embedding 数据生成的实验 notebook。

## 高优先级问题

### 1. 修复 DDP 数据加载入口

`ddp_train.py` 调用了：

```python
get_dataloader(..., sampler='distributed', rank=..., world_size=...)
```

但 `dataloader.py` 中的 `get_dataloader` 不接受这些参数，真正支持 DDP 参数的是 `get_dataloader_ddp`。这会导致 DDP 训练启动时报 `TypeError`。

建议：

- 在 `ddp_train.py` 中导入并调用 `get_dataloader_ddp`。
- 或者合并 `get_dataloader` 和 `get_dataloader_ddp`，用一个统一函数根据 `distributed=True/False` 返回 DataLoader。
- DDP DataLoader 也必须显式传入 `collate_fn=_collate_fn`，否则默认 collate 无法处理 `SolvationStructure` 对象。

### 2. 修复 `get_dataloader_ddp` 当前返回逻辑

`get_dataloader_ddp` 在 `sampler == 'distributed'` 分支中构造了 `dl`，但没有 `return dl`。此外，代码尝试执行：

```python
dl.sampler = data_sampler
```

`DataLoader.sampler` 通常不应该在构造后再赋值；构造时传入 sampler 即可。

建议：

```python
return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=data_sampler,
    collate_fn=_collate_fn,
    pin_memory=True,
    drop_last=True,
)
```

### 3. 修复评估脚本模型构造

`SolvContrastive.__init__` 当前签名是：

```python
def __init__(self, encoder: SolvEncoder, dim, proj_dim: int = 128):
```

但 `eval.py` 中调用为：

```python
model = SolvContrastive(encoder, proj_dim=128)
```

这会在运行评估时缺少必需参数 `dim`。

建议改为：

```python
model = SolvContrastive(encoder, dim=128, proj_dim=128)
```

或者给 `SolvContrastive` 的 `dim` 设置默认值，并确保训练和评估保持一致。

### 4. 重新审视 `nt_xent` 与数据采样方式

当前 DataLoader 会随机返回正样本或负样本，`nt_xent` 只保留 `labels == 1` 的 pair 参与损失：

```python
keep = torch.cat([labels, labels]).bool()
return loss[keep].mean()
```

这意味着显式采到的负样本不会作为正对参与，但仍可能出现在 batch 的 denominator 中。若一个 batch 中没有正样本，`loss[keep].mean()` 会产生 NaN。

建议：

- 保证每个 batch 至少包含一个正样本，或在 loss 中处理无正样本场景。
- 更标准的 SimCLR/NT-Xent 做法是每个样本都构造两种 view，每个 anchor 都有确定 positive，batch 内其他样本作为 negatives。
- 如果目标是二分类式结构相似性训练，可考虑改为 `BCEWithLogitsLoss`，让负样本显式参与目标函数。

### 5. 统一训练入口，减少 `train.py` 与 `ddp_train.py` 重复

两个训练脚本大量重复，且行为不完全一致，后续修复时容易产生分叉。

建议：

- 保留一个 `train.py`，通过 `torchrun` 环境变量自动判断是否启用 DDP。
- 将公共逻辑拆成 `build_dataloader`、`build_model`、`train_one_epoch`、`evaluate_or_log`。
- 删除或降级 `ddp_train.py` 为简单 wrapper，避免两套训练逻辑长期漂移。

## 中优先级问题

### 6. 为数据格式建立显式校验

`SolvationStructure._load` 默认 `.xyz` 第二行包含 signature，并默认每个原子行至少有 4 列。当前缺少异常提示，数据格式稍有不同会报出难以定位的错误。

建议：

- 校验文件至少有两行。
- 校验原子行列数和坐标可解析性。
- 当文件名 signature 与第二行 signature 不一致时给出 warning 或统一使用一种来源。
- 对空数据集、只有一个 signature、没有负样本候选等情况给出清晰错误。

### 7. 缓存结构读取，降低 I/O 开销

`ContrastiveDataset.__getitem__` 每次都会重新读取 `.xyz` 并构造 RDKit Atom。训练时同一文件会被多次读取，I/O 和对象构造开销较高。

建议：

- 在 Dataset 初始化时预加载所有结构，适合数据量不大时使用。
- 或使用 LRU cache 缓存 `SolvationStructure`。
- 对大数据集可预处理为 `.pt`、`.npz` 或 LMDB，训练时直接加载 tensor。

### 8. 增加可复现配置

当前训练参数散落在代码和 argparse 中，缺少随机种子、模型参数配置和依赖声明。

建议新增：

- `requirements.txt` 或 `pyproject.toml`，记录 `torch`、`rdkit`、`einops`、`tqdm`、`scikit-learn`、`tensorboard` 等依赖。
- `configs/default.yaml`，统一保存模型维度、层数、邻居数、学习率、batch size、路径等。
- `--seed` 参数，设置 `random`、`numpy`、`torch` 和 DataLoader worker seed。
- checkpoint 中保存 `args`、epoch、optimizer state、best loss，而不仅是 model state dict。

### 9. 增加最小测试集

目前没有自动化测试。建议增加 `tests/`：

- `test_dataloader.py`：用临时 `.xyz` 文件验证 signature 解析、正负样本生成和 batch shape。
- `test_model.py`：验证 forward 输出 shape、mask pooling 不产生 NaN。
- `test_loss.py`：验证 `nt_xent` 在全正、混合正负、无正样本场景下行为明确。
- `test_eval.py`：至少覆盖模型构造和 checkpoint 加载。

### 10. 修正自测代码

`model.py` 的 `__main__` 中：

```python
feats = torch.randint(0, Ft, (B, N))
```

但实际 `SolvEncoder` 使用 `nn.Linear(feat_dim, dim)`，期望输入是 `(B, N, feat_dim)` 的 float 特征。这个自测会因维度不匹配而失败。

建议改为：

```python
feats = torch.randn(B, N, Ft)
```

同时 `num_tokens` 对当前 Linear embedding 已经不再代表 token vocabulary，可考虑重命名或移除。

## 低优先级与工程质量建议

### 11. 清理未使用代码和重复定义

- `egnn_torch_c.py` 中 `exists` 定义了两次。
- 多个文件中存在未使用 import，例如 `time`、部分 DDP import。
- `dataloader.py` 中 `os`、`numpy`、`tqdm` 当前未使用。

建议配置 `ruff` 或 `flake8` 自动检查。

### 12. 梯度全零检查改为可选 debug

训练循环每一步遍历所有参数并打印全零梯度，会显著拖慢训练，也会污染多卡日志。

建议：

- 增加 `--debug_grad` 开关。
- 或每 N step 检查一次。
- DDP 下仅 rank 0 打印。

### 13. 明确保存目录创建逻辑

训练中直接保存：

```python
Path(args.log_dir) / 'best.pt'
```

如果目录不存在可能失败。

建议在创建 `SummaryWriter` 前执行：

```python
Path(args.log_dir).mkdir(parents=True, exist_ok=True)
```

### 14. 改善评估可靠性

当前评估仍使用随机 pair sampling，所以同一个 checkpoint 每次评估结果可能不同。

建议：

- 固定评估 seed。
- 构建确定性的 evaluation pair list。
- 分开报告 positive pair、hard negative、random negative。
- 保存评估结果到 JSON/CSV，便于实验对比。

### 15. 补充项目 README

建议新增 `README.md`，至少包含：

- 项目目标和方法简介。
- 输入 `.xyz` 格式示例。
- 安装依赖。
- 单卡训练命令。
- 多卡训练命令。
- 评估命令。
- checkpoint 与 TensorBoard 输出说明。

## 建议执行顺序

1. 先修复 DDP DataLoader、`eval.py` 模型构造、`model.py` 自测输入这三类会直接导致运行失败的问题。
2. 给 `nt_xent` 增加无正样本保护，并决定训练目标到底是 SimCLR 风格对比学习还是 pair 分类。
3. 增加最小 README、依赖文件和 smoke tests，保证后续修改不会破坏基础流程。
4. 再做性能优化，包括结构缓存、预处理 tensor、DDP 日志和 checkpoint 完整保存。
5. 最后整理配置系统和实验记录格式，提升复现实验与横向对比效率。

## 本次静态检查

已执行：

```bash
python -m py_compile dataloader.py model.py train.py ddp_train.py eval.py egnn_torch_c.py
```

结果：语法编译通过。上述问题主要是接口不一致、运行时行为和训练目标设计层面的风险。
