from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    mean_absolute_error,
    normalized_mutual_info_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

from dataloader import ATOM_PROPERTY_DIM, ELEMENTS, ContrastiveDataset, SolvationStructure, _collate_fn, build_fixed_pair_list, get_dataloader
from model import SolvContrastive, SolvEncoder
from physics import PhysicalFeatureConfig, feature_dim_for_mode, nearest_li_distance
from utils import load_model_state, set_seed


@dataclass
class EvalConfig:
    data_dir: str
    ckpt: str
    batch_size: int = 64
    seed: int = 42
    max_pairs_per_anchor: int = 2
    feat_dim: int = len(ELEMENTS)
    feature_mode: str = 'atom_phys_shell'
    li_cutoff: float = 2.5
    center_on_li: bool = True
    shell_radius: float | None = None
    dim: int = 128
    depth: int = 4
    num_nearest_neighbors: int = 12
    rdf_bins: int = 12
    rdf_max_distance: float = 6.0
    downstream_csv: str | None = None
    downstream_target: str | None = None
    few_shot_sizes: tuple[int, ...] = (2, 4, 8, 16)


def compute_coordination_number(struct: SolvationStructure, li_cutoff: float) -> int:
    min_dist = nearest_li_distance(struct.coords, struct.symbols)
    non_li = torch.tensor([sym != 'Li' for sym in struct.symbols], dtype=torch.bool)
    finite_non_li = min_dist[non_li & torch.isfinite(min_dist)]
    return int((finite_non_li <= li_cutoff).sum().item())


def shell_composition(struct: SolvationStructure, li_cutoff: float) -> dict[str, float]:
    min_dist = nearest_li_distance(struct.coords, struct.symbols)
    counts: Counter[str] = Counter()
    for sym, dist in zip(struct.symbols, min_dist.tolist()):
        if sym != 'Li' and np.isfinite(dist) and dist <= li_cutoff:
            counts[sym] += 1
    return {key: float(value) for key, value in counts.items()}


def rdf_descriptor(
    struct: SolvationStructure,
    bins: int = 12,
    max_distance: float = 6.0,
) -> np.ndarray:
    li_indices = [idx for idx, sym in enumerate(struct.symbols) if sym == 'Li']
    if not li_indices:
        return np.zeros(bins, dtype=np.float32)
    li_coords = struct.coords[torch.tensor(li_indices, dtype=torch.long)]
    non_li = torch.tensor([sym != 'Li' for sym in struct.symbols], dtype=torch.bool)
    if not torch.any(non_li):
        return np.zeros(bins, dtype=np.float32)
    dists = torch.cdist(struct.coords[non_li], li_coords).min(dim=1).values.cpu().numpy()
    hist, _ = np.histogram(dists, bins=bins, range=(0.0, max_distance))
    total = max(float(hist.sum()), 1.0)
    return (hist.astype(np.float32) / total)


def acsf_like_descriptor(
    struct: SolvationStructure,
    bins: int = 12,
    max_distance: float = 6.0,
) -> np.ndarray:
    """Small dependency-free radial descriptor used as an ACSF/SOAP-class baseline."""
    elements = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br']
    min_dist = nearest_li_distance(struct.coords, struct.symbols)
    rows = []
    for element in elements:
        mask = torch.tensor([sym == element for sym in struct.symbols], dtype=torch.bool)
        vals = min_dist[mask & torch.isfinite(min_dist)].cpu().numpy()
        hist, _ = np.histogram(vals, bins=bins, range=(0.0, max_distance))
        total = max(float(hist.sum()), 1.0)
        rows.append(hist.astype(np.float32) / total)
    return np.concatenate(rows, axis=0)


def composition_vector(signature: str) -> dict[str, float]:
    # Li_2DMC_2EC -> {'DMC': 2, 'EC': 2}; Li-only fallback stays all zeros.
    import re
    return {name: float(count) for count, name in re.findall(r'_(\d+)([A-Za-z]+)', signature)}


def matrix_from_dicts(rows: list[dict[str, float]]) -> tuple[np.ndarray, list[str]]:
    keys = sorted({key for row in rows for key in row})
    mat = np.zeros((len(rows), len(keys)), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, key in enumerate(keys):
            mat[i, j] = row.get(key, 0.0)
    return mat, keys


def load_metadata_and_descriptors(
    data_dir: str,
    li_cutoff: float,
    rdf_bins: int = 12,
    rdf_max_distance: float = 6.0,
) -> dict[str, Any]:
    ds = ContrastiveDataset(Path(data_dir), pair_list=[])
    signatures = []
    cn = []
    comp_rows = []
    shell_rows = []
    rdf_rows = []
    acsf_rows = []
    shell_labels = []
    paths = []
    for path, signature in zip(ds.paths, ds.signatures):
        struct = SolvationStructure(path)
        paths.append(str(path))
        signatures.append(signature)
        cn_value = compute_coordination_number(struct, li_cutoff)
        shell = shell_composition(struct, li_cutoff)
        cn.append(cn_value)
        comp_rows.append(composition_vector(signature))
        shell_rows.append(shell)
        shell_labels.append(f'cn={cn_value};' + ';'.join(f'{k}{int(v)}' for k, v in sorted(shell.items())))
        rdf_rows.append(rdf_descriptor(struct, bins=rdf_bins, max_distance=rdf_max_distance))
        acsf_rows.append(acsf_like_descriptor(struct, bins=rdf_bins, max_distance=rdf_max_distance))
    comp, comp_keys = matrix_from_dicts(comp_rows)
    shell_mat, shell_keys = matrix_from_dicts(shell_rows)
    return {
        'paths': paths,
        'signatures': np.asarray(signatures),
        'coordination': np.asarray(cn, dtype=np.float32),
        'composition': comp,
        'composition_keys': comp_keys,
        'shell_composition': shell_mat,
        'shell_composition_keys': shell_keys,
        'shell_state': np.asarray(shell_labels),
        'rdf': np.vstack(rdf_rows).astype(np.float32),
        'acsf_like': np.vstack(acsf_rows).astype(np.float32),
    }


def export_embeddings(cfg: EvalConfig, device: torch.device, load_checkpoint: bool = True) -> np.ndarray:
    feat_dim = cfg.feat_dim
    if cfg.feature_mode.startswith('atom_phys') and feat_dim == len(ELEMENTS):
        feat_dim = len(ELEMENTS) + ATOM_PROPERTY_DIM
    physical_config = PhysicalFeatureConfig(
        mode=cfg.feature_mode,
        li_cutoff=cfg.li_cutoff,
        center_on_li=cfg.center_on_li,
        shell_radius=cfg.shell_radius,
    )
    ds = ContrastiveDataset(Path(cfg.data_dir), physical_config=physical_config, pair_list=[])
    loader = torch.utils.data.DataLoader(
        list(range(len(ds.paths))),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )
    encoder = SolvEncoder(
        feat_dim=feature_dim_for_mode(feat_dim, cfg.feature_mode),
        dim=cfg.dim,
        depth=cfg.depth,
        num_nearest_neighbors=cfg.num_nearest_neighbors,
    )
    model = SolvContrastive(encoder, dim=cfg.dim, proj_dim=cfg.dim)
    if load_checkpoint:
        model.load_state_dict(load_model_state(cfg.ckpt, map_location='cpu'))
    model.to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for indices in loader:
            structs = [SolvationStructure(ds.paths[int(i)], physical_config=physical_config) for i in indices]
            batch = _collate_fn([(s, s, torch.tensor(1.0)) for s in structs])
            feats = batch['a_feats'].to(device).float()
            coords = batch['a_coords'].to(device).float()
            mask = batch['a_mask'].to(device)
            embeddings.append(model(feats, coords, mask).cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def pair_auc(cfg: EvalConfig, device: torch.device) -> dict[str, float]:
    feat_dim = cfg.feat_dim
    if cfg.feature_mode.startswith('atom_phys') and feat_dim == len(ELEMENTS):
        feat_dim = len(ELEMENTS) + ATOM_PROPERTY_DIM
    pair_list = build_fixed_pair_list(cfg.data_dir, cfg.max_pairs_per_anchor, cfg.seed)
    physical_config = PhysicalFeatureConfig(
        mode=cfg.feature_mode,
        li_cutoff=cfg.li_cutoff,
        center_on_li=cfg.center_on_li,
        shell_radius=cfg.shell_radius,
    )
    dl = get_dataloader(
        cfg.data_dir,
        batch_size=cfg.batch_size,
        pair_list=pair_list,
        mode='pair',
        physical_config=physical_config,
        seed=cfg.seed,
        num_workers=0,
    )
    encoder = SolvEncoder(feature_dim_for_mode(feat_dim, cfg.feature_mode), cfg.dim, cfg.depth, cfg.num_nearest_neighbors)
    model = SolvContrastive(encoder, dim=cfg.dim, proj_dim=cfg.dim)
    model.load_state_dict(load_model_state(cfg.ckpt, map_location='cpu'))
    model.to(device)
    model.eval()
    sims, labels = [], []
    with torch.no_grad():
        for batch in dl:
            za = model(batch['a_feats'].to(device).float(), batch['a_coords'].to(device).float(), batch['a_mask'].to(device))
            zo = model(batch['o_feats'].to(device).float(), batch['o_coords'].to(device).float(), batch['o_mask'].to(device))
            sims.append(torch.nn.functional.cosine_similarity(za, zo, dim=-1).cpu().numpy())
            labels.append(batch['labels'].numpy())
    sims_arr = np.concatenate(sims)
    labels_arr = np.concatenate(labels)
    if len(np.unique(labels_arr)) < 2:
        return {'pair_auc': float('nan'), 'pair_pos_sim': float('nan'), 'pair_neg_sim': float('nan')}
    return {
        'pair_auc': float(roc_auc_score(labels_arr, sims_arr)),
        'pair_pos_sim': float(sims_arr[labels_arr == 1].mean()),
        'pair_neg_sim': float(sims_arr[labels_arr == 0].mean()),
    }


def classification_probe(x: np.ndarray, y: np.ndarray, seed: int) -> dict[str, float]:
    if x.shape[1] == 0 or len(np.unique(y)) < 2 or len(y) < 4:
        return {'accuracy': float('nan'), 'majority_accuracy': float('nan')}
    labels = LabelEncoder().fit_transform(y)
    stratify = labels if min(Counter(labels).values()) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, labels, test_size=0.4, random_state=seed, stratify=stratify
    )
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(x_train, y_train)
    return {
        'accuracy': float(accuracy_score(y_test, pred)),
        'majority_accuracy': float(accuracy_score(y_test, dummy.predict(x_test))),
    }


def regression_probe(x: np.ndarray, y: np.ndarray, seed: int) -> dict[str, float]:
    if x.shape[1] == 0 or len(y) < 4 or np.allclose(y, y[0]):
        return {'mae': float('nan'), 'r2': float('nan'), 'dummy_mae': float('nan')}
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=seed)
    reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    reg.fit(x_train, y_train)
    pred = reg.predict(x_test)
    dummy = DummyRegressor(strategy='mean')
    dummy.fit(x_train, y_train)
    return {
        'mae': float(mean_absolute_error(y_test, pred)),
        'r2': float(r2_score(y_test, pred)),
        'dummy_mae': float(mean_absolute_error(y_test, dummy.predict(x_test))),
    }


def clustering_report(x: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, float]:
    unique = np.unique(labels)
    if len(unique) < 2 or len(labels) < len(unique):
        return {'ari': float('nan'), 'nmi': float('nan')}
    n_clusters = min(len(unique), len(labels))
    pred = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(
        StandardScaler().fit_transform(x)
    )
    encoded = LabelEncoder().fit_transform(labels)
    return {
        'ari': float(adjusted_rand_score(encoded, pred)),
        'nmi': float(normalized_mutual_info_score(encoded, pred)),
    }


def read_downstream_table(path: str, target: str) -> tuple[dict[str, float], bool]:
    import csv

    rows: dict[str, float] = {}
    is_numeric = True
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if target not in (reader.fieldnames or []):
            raise ValueError(f'downstream target {target!r} not found in {path}')
        for row in reader:
            key = row.get('path') or row.get('filename') or row.get('signature')
            value = row[target]
            if key is None or value == '':
                continue
            try:
                parsed: float | str = float(value)
            except ValueError:
                parsed = value
                is_numeric = False
            rows[str(key)] = parsed
    return rows, is_numeric


def align_downstream(meta: dict[str, Any], path: str, target: str) -> tuple[np.ndarray, np.ndarray, bool]:
    values, is_numeric = read_downstream_table(path, target)
    keep = []
    y = []
    for idx, sample_path in enumerate(meta['paths']):
        filename = Path(sample_path).name
        signature = str(meta['signatures'][idx])
        for key in (sample_path, filename, signature):
            if key in values:
                keep.append(idx)
                y.append(values[key])
                break
    if is_numeric:
        return np.asarray(keep, dtype=np.int64), np.asarray(y, dtype=np.float32), True
    return np.asarray(keep, dtype=np.int64), np.asarray(y), False


def few_shot_probe(
    x: np.ndarray,
    y: np.ndarray,
    is_regression: bool,
    seed: int,
    train_sizes: tuple[int, ...],
) -> dict[str, float]:
    if len(y) < 4:
        return {}
    rng = np.random.default_rng(seed)
    metrics: dict[str, float] = {}
    for size in train_sizes:
        if size >= len(y) or size < 1:
            continue
        train_idx = rng.choice(len(y), size=size, replace=False)
        test_idx = np.asarray([idx for idx in range(len(y)) if idx not in set(train_idx.tolist())])
        if is_regression:
            if np.allclose(y[train_idx], y[train_idx][0]):
                continue
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(x[train_idx], y[train_idx])
            pred = model.predict(x[test_idx])
            metrics[f'few_shot_{size}_mae'] = float(mean_absolute_error(y[test_idx], pred))
        else:
            labels = LabelEncoder().fit_transform(y)
            if len(np.unique(labels[train_idx])) < 2:
                continue
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            model.fit(x[train_idx], labels[train_idx])
            pred = model.predict(x[test_idx])
            metrics[f'few_shot_{size}_accuracy'] = float(accuracy_score(labels[test_idx], pred))
    return metrics


def run_evaluation_suite(cfg: EvalConfig, device: torch.device) -> dict[str, Any]:
    set_seed(cfg.seed)
    meta = load_metadata_and_descriptors(
        cfg.data_dir,
        cfg.li_cutoff,
        rdf_bins=cfg.rdf_bins,
        rdf_max_distance=cfg.rdf_max_distance,
    )
    embeddings = export_embeddings(cfg, device)
    random_encoder_embeddings = export_embeddings(cfg, device, load_checkpoint=False)
    signature_embed = classification_probe(embeddings, meta['signatures'], cfg.seed)
    signature_comp = classification_probe(meta['composition'], meta['signatures'], cfg.seed)
    signature_random = classification_probe(random_encoder_embeddings, meta['signatures'], cfg.seed)
    cn_embed = regression_probe(embeddings, meta['coordination'], cfg.seed)
    cn_comp = regression_probe(meta['composition'], meta['coordination'], cfg.seed)
    cn_random = regression_probe(random_encoder_embeddings, meta['coordination'], cfg.seed)
    cn_rdf = regression_probe(meta['rdf'], meta['coordination'], cfg.seed)
    cn_acsf = regression_probe(meta['acsf_like'], meta['coordination'], cfg.seed)
    shell_embed = classification_probe(embeddings, meta['shell_state'], cfg.seed)
    shell_comp = classification_probe(meta['composition'], meta['shell_state'], cfg.seed)
    shell_random = classification_probe(random_encoder_embeddings, meta['shell_state'], cfg.seed)
    shell_rdf = classification_probe(meta['rdf'], meta['shell_state'], cfg.seed)
    cluster_embed = clustering_report(embeddings, meta['shell_state'], cfg.seed)
    cluster_rdf = clustering_report(meta['rdf'], meta['shell_state'], cfg.seed)
    metrics = {
        **pair_auc(cfg, device),
        'signature_probe_accuracy': signature_embed['accuracy'],
        'signature_majority_accuracy': signature_embed['majority_accuracy'],
        'signature_composition_accuracy': signature_comp['accuracy'],
        'signature_random_encoder_accuracy': signature_random['accuracy'],
        'coordination_probe_mae': cn_embed['mae'],
        'coordination_probe_r2': cn_embed['r2'],
        'coordination_dummy_mae': cn_embed['dummy_mae'],
        'coordination_composition_mae': cn_comp['mae'],
        'coordination_random_encoder_mae': cn_random['mae'],
        'coordination_rdf_mae': cn_rdf['mae'],
        'coordination_acsf_like_mae': cn_acsf['mae'],
        'shell_state_probe_accuracy': shell_embed['accuracy'],
        'shell_state_composition_accuracy': shell_comp['accuracy'],
        'shell_state_random_encoder_accuracy': shell_random['accuracy'],
        'shell_state_rdf_accuracy': shell_rdf['accuracy'],
        'shell_state_majority_accuracy': shell_embed['majority_accuracy'],
        'embedding_shell_cluster_ari': cluster_embed['ari'],
        'embedding_shell_cluster_nmi': cluster_embed['nmi'],
        'rdf_shell_cluster_ari': cluster_rdf['ari'],
        'rdf_shell_cluster_nmi': cluster_rdf['nmi'],
        'num_samples': int(len(meta['paths'])),
        'num_signatures': int(len(np.unique(meta['signatures']))),
        'num_shell_states': int(len(np.unique(meta['shell_state']))),
    }
    downstream: dict[str, Any] = {}
    if cfg.downstream_csv and cfg.downstream_target:
        indices, y, is_regression = align_downstream(meta, cfg.downstream_csv, cfg.downstream_target)
        if len(indices) > 0:
            downstream = {
                'target': cfg.downstream_target,
                'num_samples': int(len(indices)),
                'task': 'regression' if is_regression else 'classification',
            }
            downstream_metrics = few_shot_probe(
                embeddings[indices],
                y,
                is_regression,
                cfg.seed,
                cfg.few_shot_sizes,
            )
            metrics.update({f'downstream_{k}': v for k, v in downstream_metrics.items()})
            downstream['metrics'] = downstream_metrics
        else:
            downstream = {'target': cfg.downstream_target, 'num_samples': 0, 'task': 'unmatched'}
    return {'metrics': metrics, 'metadata': meta, 'embeddings': embeddings, 'downstream': downstream}
