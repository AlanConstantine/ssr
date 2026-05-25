from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

from dataloader import ContrastiveDataset, SolvationStructure, _collate_fn, build_fixed_pair_list, get_dataloader
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
    feat_dim: int = 10
    feature_mode: str = 'element_shell'
    li_cutoff: float = 2.5
    center_on_li: bool = True
    shell_radius: float | None = None
    dim: int = 128
    depth: int = 4
    num_nearest_neighbors: int = 12


def compute_coordination_number(struct: SolvationStructure, li_cutoff: float) -> int:
    min_dist = nearest_li_distance(struct.coords, struct.symbols)
    non_li = torch.tensor([sym != 'Li' for sym in struct.symbols], dtype=torch.bool)
    finite_non_li = min_dist[non_li & torch.isfinite(min_dist)]
    return int((finite_non_li <= li_cutoff).sum().item())


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


def load_metadata_and_descriptors(data_dir: str, li_cutoff: float) -> dict[str, Any]:
    ds = ContrastiveDataset(Path(data_dir), pair_list=[])
    signatures = []
    cn = []
    comp_rows = []
    paths = []
    for path, signature in zip(ds.paths, ds.signatures):
        struct = SolvationStructure(path)
        paths.append(str(path))
        signatures.append(signature)
        cn.append(compute_coordination_number(struct, li_cutoff))
        comp_rows.append(composition_vector(signature))
    comp, comp_keys = matrix_from_dicts(comp_rows)
    return {
        'paths': paths,
        'signatures': np.asarray(signatures),
        'coordination': np.asarray(cn, dtype=np.float32),
        'composition': comp,
        'composition_keys': comp_keys,
    }


def export_embeddings(cfg: EvalConfig, device: torch.device) -> np.ndarray:
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
        feat_dim=feature_dim_for_mode(cfg.feat_dim, cfg.feature_mode),
        dim=cfg.dim,
        depth=cfg.depth,
        num_nearest_neighbors=cfg.num_nearest_neighbors,
    )
    model = SolvContrastive(encoder, dim=cfg.dim, proj_dim=cfg.dim)
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
    encoder = SolvEncoder(feature_dim_for_mode(cfg.feat_dim, cfg.feature_mode), cfg.dim, cfg.depth, cfg.num_nearest_neighbors)
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
    if len(np.unique(y)) < 2 or len(y) < 4:
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
    if len(y) < 4 or np.allclose(y, y[0]):
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


def run_evaluation_suite(cfg: EvalConfig, device: torch.device) -> dict[str, Any]:
    set_seed(cfg.seed)
    meta = load_metadata_and_descriptors(cfg.data_dir, cfg.li_cutoff)
    embeddings = export_embeddings(cfg, device)
    signature_embed = classification_probe(embeddings, meta['signatures'], cfg.seed)
    signature_comp = classification_probe(meta['composition'], meta['signatures'], cfg.seed)
    cn_embed = regression_probe(embeddings, meta['coordination'], cfg.seed)
    cn_comp = regression_probe(meta['composition'], meta['coordination'], cfg.seed)
    metrics = {
        **pair_auc(cfg, device),
        'signature_probe_accuracy': signature_embed['accuracy'],
        'signature_majority_accuracy': signature_embed['majority_accuracy'],
        'signature_composition_accuracy': signature_comp['accuracy'],
        'coordination_probe_mae': cn_embed['mae'],
        'coordination_probe_r2': cn_embed['r2'],
        'coordination_dummy_mae': cn_embed['dummy_mae'],
        'coordination_composition_mae': cn_comp['mae'],
        'num_samples': int(len(meta['paths'])),
        'num_signatures': int(len(np.unique(meta['signatures']))),
    }
    return {'metrics': metrics, 'metadata': meta, 'embeddings': embeddings}
