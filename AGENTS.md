# AGENTS.md

## Project Overview

SSR (Solvation Structure Representation) is a research prototype for learning embeddings from solvated atomic structures. The current pipeline reads `.xyz` files, encodes element features and 3D coordinates with an EGNN, and trains contrastive embeddings.

The preferred default path is SimCLR-style self-supervision: two augmented views of the same structure are positives, while other samples in the batch act as negatives. The older signature-pair mode is kept for compatibility and evaluation, but signature labels should not be treated as the main training objective.

## Current Status

- Stage 0 basics are in place: config loading, seed handling, richer checkpoints, fixed eval pair lists, environment metadata, and smoke tests.
- Stage 1A is implemented: same-structure augmented views, `SimCLRDataset`, and standard SimCLR / NT-Xent loss.
- Stage 2 basics are implemented: `pair` and `simclr` modes plus `nt_xent`, `simclr`, `bce_similarity`, and `triplet_margin` losses.
- Stage 3 xyz-derived physical features are partially implemented: `element_shell`, Li-centering, optional Li shell cropping, and physical descriptor export.
- Stage 4/5 foundation is present: evaluation suite, embedding export, physical descriptor export, tiny xyz generation, and an end-to-end experiment runner.

Planned but not complete: temporal positives, physical descriptor based positives, hard negative mining, RDF/SOAP baselines, real downstream property tasks, visualization, and CI.

## Important Files

- `README.md`: user-facing usage, data format, training, evaluation, and experiment commands.
- `DEVELOPMENT_ROADMAP.md`: staged maturity plan and current progress.
- `configs/default.yaml`: default experiment configuration.
- `dataloader.py`: xyz parsing, signature parsing, `pair` and `simclr` datasets, collate functions.
- `augment.py`: 3D augmentations for SimCLR views.
- `physics.py`: Li-centered features, shell features, centering, and cropping helpers.
- `model.py`: EGNN encoder wrapper, projection head, and contrastive losses.
- `train.py`: single-GPU / CPU / DDP training entrypoint.
- `eval.py`: fixed pair-list checkpoint evaluation.
- `evaluation.py`: shared evaluation suite logic.
- `scripts/run_experiment.py`: train + evaluation suite + descriptor export runner.
- `scripts/run_evaluation_suite.py`: metrics, embeddings, and metadata export.
- `scripts/export_embeddings.py`: checkpoint embedding export.
- `scripts/compute_physical_descriptors.py`: xyz physical descriptor export.
- `scripts/make_tiny_xyz.py`: small CPU test dataset generator.
- `tests/test_smoke.py`: minimal regression coverage.

## Development Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run smoke tests:

```bash
pytest tests/test_smoke.py
```

Compile-check Python files:

```bash
python -m py_compile *.py scripts/*.py
```

Generate tiny xyz data:

```bash
python scripts/make_tiny_xyz.py --out_dir ./tmp/tiny_xyz
```

Run a CPU end-to-end experiment:

```bash
python scripts/run_experiment.py \
  --data_dir ./tmp/tiny_xyz \
  --log_dir ./runs/tiny_cpu \
  --epochs 1 \
  --batch_size 2 \
  --device cpu
```

## Coding Guidelines

- Prefer existing project patterns and keep changes research-reproducible.
- Keep `simclr` as the default training mode unless a task explicitly targets pair compatibility.
- Use `PhysicalFeatureConfig` and `feature_dim_for_mode` when changing feature modes; do not hard-code derived feature dimensions.
- Keep training, evaluation, and export feature settings aligned, especially `feature_mode`, `center_on_li`, `li_cutoff`, and `shell_radius`.
- Preserve deterministic behavior where possible: respect `--seed`, fixed pair lists, and DataLoader worker seeding.
- Add or update smoke tests when touching dataloading, model shapes, loss behavior, checkpoint loading, or evaluation outputs.
- Avoid introducing dependencies unless they are needed for a concrete experiment path and are added to `requirements.txt`.

## Data Assumptions

- Training data directories contain `.xyz` files.
- Each xyz file has at least a first atom-count line, a second metadata/signature line, and atom rows with `element x y z`.
- Filename signatures are expected to look like `Frame100_Li_2DMC_2EC_2EMC_id1030.xyz`; the parsed signature is `Li_2DMC_2EC_2EMC`.
- Default element one-hot order is `H C N O F Li P S Cl Br`.
- Pair mode needs at least two different signatures unless a fixed `pair_list` is provided.

## Evaluation Notes

The pair ROC-AUC metric is useful for compatibility, but it can reward composition clustering. Prefer reporting it alongside:

- signature linear probe
- composition-only signature baseline
- coordination-number regression probe
- composition-only coordination baseline
- dummy baselines

The long-term goal is an embedding that reflects local solvation geometry and physical state, not only composition signature.
