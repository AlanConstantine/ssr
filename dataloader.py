"""
dataloader.py
Constructs contrastive pairs for solvation-structure representation learning.

A positive pair = two different frames/IDs that share the same
“Li_XDMC_YEC_ZEMC” solvent signature.
A negative pair = any two structures with different signatures.

Returned DataLoader yields (anchor, pos/neg, label) where
label = 1 for positive, 0 for negative.
"""

from __future__ import annotations
from dataclasses import dataclass
import re
import random
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict, Sequence
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from augment import AugmentConfig, augment_structure
from physics import PhysicalFeatureConfig, append_shell_features, center_and_crop_on_li
from utils import seed_worker

# ------------------------------------------------------------------ #
# Common electrolyte / battery-interface elements used for one-hot identity.
ELEMENTS = [
    'H', 'Li', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al',
    'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Br', 'I',
]

# Raw values: atomic number, atomic mass, Pauling electronegativity,
# covalent radius, vdW radius, group, period, valence electrons.
ATOM_PROPERTIES = {
    'H': (1, 1.008, 2.20, 0.31, 1.20, 1, 1, 1),
    'Li': (3, 6.94, 0.98, 1.28, 1.82, 1, 2, 1),
    'B': (5, 10.81, 2.04, 0.84, 1.92, 13, 2, 3),
    'C': (6, 12.011, 2.55, 0.76, 1.70, 14, 2, 4),
    'N': (7, 14.007, 3.04, 0.71, 1.55, 15, 2, 5),
    'O': (8, 15.999, 3.44, 0.66, 1.52, 16, 2, 6),
    'F': (9, 18.998, 3.98, 0.57, 1.47, 17, 2, 7),
    'Na': (11, 22.990, 0.93, 1.66, 2.27, 1, 3, 1),
    'Mg': (12, 24.305, 1.31, 1.41, 1.73, 2, 3, 2),
    'Al': (13, 26.982, 1.61, 1.21, 1.84, 13, 3, 3),
    'Si': (14, 28.085, 1.90, 1.11, 2.10, 14, 3, 4),
    'P': (15, 30.974, 2.19, 1.07, 1.80, 15, 3, 5),
    'S': (16, 32.06, 2.58, 1.05, 1.80, 16, 3, 6),
    'Cl': (17, 35.45, 3.16, 1.02, 1.75, 17, 3, 7),
    'K': (19, 39.098, 0.82, 2.03, 2.75, 1, 4, 1),
    'Ca': (20, 40.078, 1.00, 1.76, 2.31, 2, 4, 2),
    'Br': (35, 79.904, 2.96, 1.20, 1.85, 17, 4, 7),
    'I': (53, 126.904, 2.66, 1.39, 1.98, 17, 5, 7),
}
ATOM_PROPERTY_DIM = 8


@dataclass(frozen=True)
class TemporalMetadata:
    trajectory_id: str
    center_id: str
    temporal_id: str
    frame_index: int
    signature: str


def element_one_hot(symbol: str) -> torch.Tensor:
    vec = torch.zeros(len(ELEMENTS))
    try:
        vec[ELEMENTS.index(symbol)] = 1.0
    except ValueError:              # unknown element → all zeros
        pass
    return vec


def atom_property_features(symbol: str) -> torch.Tensor:
    raw = ATOM_PROPERTIES.get(symbol)
    if raw is None:
        return torch.zeros(ATOM_PROPERTY_DIM)
    atomic_number, mass, electronegativity, covalent_radius, vdw_radius, group, period, valence = raw
    return torch.tensor([
        atomic_number / 60.0,
        mass / 130.0,
        electronegativity / 4.0,
        covalent_radius / 2.5,
        vdw_radius / 3.0,
        group / 18.0,
        period / 6.0,
        valence / 8.0,
    ], dtype=torch.float32)


def atom_identity_and_property_features(symbol: str) -> torch.Tensor:
    return torch.cat([element_one_hot(symbol), atom_property_features(symbol)])


# ------------------------------------------------------------------ #
class SolvationStructure:
    """
    Small helper object that holds
      - coordinates  (N, 3)
      - atom symbols (N,)
      - features     (N, F)   default = one-hot element
      - signature    str      e.g.  Li_2DMC_2EC_2EMC
    """
    def __init__(self,
                 xyz_path: Path,
                 feat_fn: Optional[Callable[[str], torch.Tensor]] = None,
                 physical_config: Optional[PhysicalFeatureConfig] = None):
        self.path = xyz_path
        self.physical_config = physical_config or PhysicalFeatureConfig()
        self.coords: torch.Tensor
        self.symbols: List[str]
        self.signature: str
        self.features: torch.Tensor
        default_feat_fn = atom_identity_and_property_features if self.physical_config.mode.startswith('atom_phys') else element_one_hot
        self._load(feat_fn or default_feat_fn)

    # -------------------------------------------------------------- #
    def _load(self, feat_fn: Callable[[str], torch.Tensor]) -> None:
        with open(self.path, 'r') as f:
            lines = [ln.strip() for ln in f.readlines()]

        if len(lines) < 3:
            raise ValueError(f'{self.path} is not a valid xyz file: expected header and atom rows')

        # 2nd line → signature
        self.signature = lines[1].split(':')[-1].split('.')[0].strip()

        # skip first two lines
        coords, symbols, feats = [], [], []
        for line_no, ln in enumerate(lines[2:], start=3):
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 4:
                raise ValueError(f'{self.path}:{line_no} has fewer than 4 columns')
            sym, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
            symbols.append(sym)
            coords.append([x, y, z])
            feats.append(feat_fn(sym))

        if not coords:
            raise ValueError(f'{self.path} contains no atoms')

        self.coords = torch.tensor(coords, dtype=torch.float32)
        self.symbols = symbols
        self.features = torch.stack(feats)

        if self.physical_config.center_on_li or self.physical_config.shell_radius is not None:
            self.coords, self.features, self.symbols = center_and_crop_on_li(
                self.coords,
                self.features,
                self.symbols,
                shell_radius=self.physical_config.shell_radius,
            )

        if self.physical_config.mode in {'element_shell', 'atom_phys_shell'}:
            self.features = append_shell_features(
                self.features,
                self.coords,
                self.symbols,
                li_cutoff=self.physical_config.li_cutoff,
            )
        elif self.physical_config.mode not in {'element', 'atom_phys'}:
            raise ValueError(f'unsupported feature mode: {self.physical_config.mode}')


# ------------------------------------------------------------------ #
class ContrastiveDataset(Dataset):
    """
    Builds positive/negative pairs on-the-fly.
    `__getitem__` returns (anchor, other, label)
    """
    def __init__(self,
                 data_dir: Path,
                 max_neg: Optional[int] = None,
                 feat_fn: Optional[Callable[[str], torch.Tensor]] = None,
                 physical_config: Optional[PhysicalFeatureConfig] = None,
                 pair_list: Optional[Sequence[Tuple[int, int, float]]] = None):
        self.data_dir, self.paths = _discover_xyz_paths(data_dir)
        self.max_neg = max_neg
        self.feat_fn = feat_fn
        self.physical_config = physical_config or PhysicalFeatureConfig()
        self.pair_list = list(pair_list) if pair_list is not None else None

        # bucket by signature
        self.sig2idx: Dict[str, List[int]] = {}
        for idx, p in enumerate(self.paths):
            sig = self._signature_from_path(p)
            self.sig2idx.setdefault(sig, []).append(idx)

        self.all_indices = list(range(len(self.paths)))
        self.signatures = [self._signature_from_path(p) for p in self.paths]

        if self.pair_list is None and len(self.sig2idx) < 2:
            raise ValueError('contrastive training requires at least two different signatures')

    # -------------------------------------------------------------- #
    @staticmethod
    def _signature_from_path(p: Path) -> str:
        # Frame100_Li_2DMC_2EC_2EMC_id1030.xyz  → Li_2DMC_2EC_2EMC
        name = p.stem
        m = re.search(r'Li(?:_\d+[A-Z]+)+', name)
        return m.group(0) if m else name

    # -------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.paths)

    # -------------------------------------------------------------- #
    def __getitem__(self, idx: int) -> Tuple[SolvationStructure,
                                             SolvationStructure,
                                             torch.Tensor]:
        """
        Returns (anchor, other, label)
        """
        if self.pair_list is not None:
            anchor_idx, other_idx, label_value = self.pair_list[idx]
            anchor = SolvationStructure(self.paths[anchor_idx], self.feat_fn, self.physical_config)
            other = SolvationStructure(self.paths[other_idx], self.feat_fn, self.physical_config)
            return anchor, other, torch.tensor(float(label_value))

        anchor_path = self.paths[idx]
        anchor_sig = self.signatures[idx]

        # decide positive or negative
        if random.random() < 0.5:        # positive
            candidates = [i for i in self.sig2idx[anchor_sig] if i != idx]
            if not candidates:           # fallback to self
                other_idx = idx
            else:
                other_idx = random.choice(candidates)
            label = torch.tensor(1.0)
        else:                            # negative
            candidates = [i for i in self.all_indices
                          if self.signatures[i] != anchor_sig]
            if self.max_neg:
                candidates = random.sample(candidates,
                                           min(len(candidates), self.max_neg))
            if not candidates:
                raise ValueError(f'no negative candidates found for signature {anchor_sig}')
            other_idx = random.choice(candidates)
            label = torch.tensor(0.0)

        anchor = SolvationStructure(anchor_path, self.feat_fn, self.physical_config)
        other = SolvationStructure(self.paths[other_idx], self.feat_fn, self.physical_config)

        return anchor, other, label

    def __len__(self) -> int:
        if self.pair_list is not None:
            return len(self.pair_list)
        return len(self.paths)


class SimCLRDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        feat_fn: Optional[Callable[[str], torch.Tensor]] = None,
        augment_config: Optional[AugmentConfig] = None,
        physical_config: Optional[PhysicalFeatureConfig] = None,
    ):
        self.data_dir, self.paths = _discover_xyz_paths(data_dir)
        self.feat_fn = feat_fn
        self.augment_config = augment_config or AugmentConfig()
        self.physical_config = physical_config or PhysicalFeatureConfig()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[SolvationStructure, SolvationStructure]:
        struct = SolvationStructure(self.paths[idx], self.feat_fn, self.physical_config)
        coords_a, feats_a = augment_structure(struct.coords, struct.features, self.augment_config)
        coords_b, feats_b = augment_structure(struct.coords, struct.features, self.augment_config)
        view_a = _view_from_structure(struct, coords_a, feats_a)
        view_b = _view_from_structure(struct, coords_b, feats_b)
        return view_a, view_b


class TemporalDataset(Dataset):
    """
    Builds trajectory-aware pairs.

    Positive pairs are nearby frames from the same trajectory. Negatives are
    preferentially distant frames with the same signature, then frames from
    other trajectories. This avoids using composition as an easy shortcut when
    the data provides same-composition temporal alternatives.
    """
    def __init__(
        self,
        data_dir: Path,
        feat_fn: Optional[Callable[[str], torch.Tensor]] = None,
        physical_config: Optional[PhysicalFeatureConfig] = None,
        positive_window: int = 5,
        min_lag: int = 1,
        negative_min_gap: int = 50,
        same_signature_negatives: bool = True,
        seed: int = 42,
    ):
        if positive_window < min_lag:
            raise ValueError('positive_window must be >= min_lag')
        if min_lag < 1:
            raise ValueError('min_lag must be >= 1 to avoid self-pairs')
        if negative_min_gap <= positive_window:
            raise ValueError('negative_min_gap must be larger than positive_window')

        self.data_dir, self.paths = _discover_xyz_paths(data_dir)
        self.feat_fn = feat_fn
        self.physical_config = physical_config or PhysicalFeatureConfig()
        self.positive_window = positive_window
        self.min_lag = min_lag
        self.negative_min_gap = negative_min_gap
        self.same_signature_negatives = same_signature_negatives
        self.rng = random.Random(seed)

        self.metadata = [parse_temporal_metadata(path) for path in self.paths]
        self.traj2idx: Dict[str, List[int]] = {}
        for idx, meta in enumerate(self.metadata):
            self.traj2idx.setdefault(meta.temporal_id, []).append(idx)
        for indices in self.traj2idx.values():
            indices.sort(key=lambda i: self.metadata[i].frame_index)

        self.positive_candidates: Dict[int, List[int]] = {}
        self.negative_candidates: Dict[int, List[int]] = {}
        for idx, meta in enumerate(self.metadata):
            positives = []
            negatives = []
            for other_idx, other_meta in enumerate(self.metadata):
                if other_idx == idx:
                    continue
                same_traj = other_meta.temporal_id == meta.temporal_id
                frame_gap = abs(other_meta.frame_index - meta.frame_index)
                if same_traj and self.min_lag <= frame_gap <= self.positive_window:
                    positives.append(other_idx)
                elif same_traj and frame_gap >= self.negative_min_gap:
                    negatives.append(other_idx)
                elif not same_traj:
                    negatives.append(other_idx)

            if self.same_signature_negatives:
                same_sig = [i for i in negatives if self.metadata[i].signature == meta.signature]
                if same_sig:
                    negatives = same_sig

            if not positives:
                raise ValueError(
                    f'no temporal positive candidates for {self.paths[idx].name}; '
                    f'check trajectory ids, frame indices, and positive_window'
                )
            if not negatives:
                raise ValueError(
                    f'no temporal negative candidates for {self.paths[idx].name}; '
                    f'increase data diversity or lower negative_min_gap'
                )
            self.positive_candidates[idx] = positives
            self.negative_candidates[idx] = negatives

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[SolvationStructure, SolvationStructure, torch.Tensor]:
        if self.rng.random() < 0.5:
            other_idx = self.rng.choice(self.positive_candidates[idx])
            label = torch.tensor(1.0)
        else:
            other_idx = self.rng.choice(self.negative_candidates[idx])
            label = torch.tensor(0.0)

        anchor = SolvationStructure(self.paths[idx], self.feat_fn, self.physical_config)
        other = SolvationStructure(self.paths[other_idx], self.feat_fn, self.physical_config)
        return anchor, other, label


def _discover_xyz_paths(data_dir: Path) -> tuple[Path, list[Path]]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f'data_dir does not exist: {data_dir}')
    paths = sorted(data_dir.glob('*.xyz'))
    if not paths:
        raise ValueError(f'no .xyz files found in {data_dir}')
    return data_dir, paths


def _read_xyz_comment(path: Path) -> str:
    with open(path, 'r') as f:
        f.readline()
        return f.readline().strip()


def parse_temporal_metadata(path: Path) -> TemporalMetadata:
    comment = _read_xyz_comment(path)
    name = path.stem
    signature = ContrastiveDataset._signature_from_path(path)

    trajectory_id = _find_named_value(comment, ('trajectory', 'traj', 'run', 'sim'))
    if trajectory_id is None:
        trajectory_id = _find_named_value(name, ('trajectory', 'traj', 'run', 'sim'))
    if trajectory_id is None:
        raise ValueError(
            f'could not parse trajectory id from {path.name}; use filename tokens '
            f'like TrajA_Frame100_... or xyz comment fields like "trajectory: TrajA frame: 100"'
        )

    frame_value = _find_named_value(comment, ('frame', 'step', 'timestep'))
    if frame_value is None:
        frame_value = _find_named_value(name, ('frame', 'step', 'timestep'))
    if frame_value is None or not str(frame_value).isdigit():
        raise ValueError(
            f'could not parse frame index from {path.name}; use Frame100, step100, '
            f'or xyz comment fields like "frame: 100"'
        )
    center_id = _find_named_value(comment, ('center_id', 'center', 'li_id', 'id'))
    if center_id is None:
        center_id = _find_named_value(name, ('center_id', 'center', 'li_id', 'id'))
    if center_id is None:
        raise ValueError(
            f'could not parse center Li id from {path.name}; use id1030, center1030, '
            f'or xyz comment fields like "center_id: 1030"'
        )
    trajectory_id = str(trajectory_id)
    center_id = str(center_id)
    return TemporalMetadata(
        trajectory_id=trajectory_id,
        center_id=center_id,
        temporal_id=f'{trajectory_id}:{center_id}',
        frame_index=int(frame_value),
        signature=signature,
    )


def _find_named_value(text: str, names: Sequence[str]) -> Optional[str]:
    for name in names:
        pattern = rf'(?:^|[^A-Za-z0-9]){name}[\s_:=.-]*([A-Za-z0-9]+)'
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _view_from_structure(
    struct: SolvationStructure,
    coords: torch.Tensor,
    feats: torch.Tensor,
) -> SolvationStructure:
    view = object.__new__(SolvationStructure)
    view.path = struct.path
    view.coords = coords
    view.features = feats
    view.signature = struct.signature
    view.symbols = struct.symbols[:coords.size(0)]
    return view


def build_fixed_pair_list(
    data_dir: str,
    max_pairs_per_anchor: int = 2,
    seed: int = 42,
) -> list[tuple[int, int, float]]:
    ds = ContrastiveDataset(Path(data_dir))
    rng = random.Random(seed)
    pairs: list[tuple[int, int, float]] = []
    for idx, sig in enumerate(ds.signatures):
        positives = [i for i in ds.sig2idx[sig] if i != idx]
        negatives = [i for i in ds.all_indices if ds.signatures[i] != sig]
        rng.shuffle(positives)
        rng.shuffle(negatives)
        for other_idx in positives[:max_pairs_per_anchor]:
            pairs.append((idx, other_idx, 1.0))
        for other_idx in negatives[:max_pairs_per_anchor]:
            pairs.append((idx, other_idx, 0.0))
    rng.shuffle(pairs)
    return pairs


# ------------------------------------------------------------------ #
def get_dataloader(data_dir: str,
                   batch_size: int = 32,
                   num_workers: int = 4,
                   max_neg: Optional[int] = None,
                   feat_fn: Optional[Callable[[str], torch.Tensor]] = None,
                   mode: str = 'pair',
                   augment_config: Optional[AugmentConfig] = None,
                   physical_config: Optional[PhysicalFeatureConfig] = None,
                   pair_list: Optional[Sequence[Tuple[int, int, float]]] = None,
                   sampler: Optional[str] = None,
                   rank: Optional[int] = None,
                   world_size: Optional[int] = None,
                   drop_last: bool = False,
                   seed: int = 42,
                   temporal_positive_window: int = 5,
                   temporal_min_lag: int = 1,
                   temporal_negative_min_gap: int = 50,
                   temporal_same_signature_negatives: bool = True,
                   ) -> DataLoader:
    if physical_config is not None and physical_config.mode in {'atom_phys', 'atom_phys_shell'} and feat_fn is None:
        feat_fn = atom_identity_and_property_features

    if mode == 'pair':
        ds = ContrastiveDataset(Path(data_dir), max_neg, feat_fn,
                                physical_config=physical_config,
                                pair_list=pair_list)
        collate_fn = _collate_fn
    elif mode == 'simclr':
        ds = SimCLRDataset(Path(data_dir), feat_fn=feat_fn,
                           augment_config=augment_config,
                           physical_config=physical_config)
        collate_fn = _simclr_collate_fn
    elif mode == 'temporal':
        ds = TemporalDataset(Path(data_dir),
                             feat_fn=feat_fn,
                             physical_config=physical_config,
                             positive_window=temporal_positive_window,
                             min_lag=temporal_min_lag,
                             negative_min_gap=temporal_negative_min_gap,
                             same_signature_negatives=temporal_same_signature_negatives,
                             seed=seed)
        collate_fn = _collate_fn
    else:
        raise ValueError(f'unsupported dataloader mode: {mode}')

    generator = torch.Generator()
    generator.manual_seed(seed)

    if sampler == 'distributed':
        if rank is None or world_size is None:
            raise ValueError('rank and world_size must be provided for distributed sampler')
        data_sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        return DataLoader(ds,
                          batch_size=batch_size,
                          sampler=data_sampler,
                          num_workers=num_workers,
                          collate_fn=collate_fn,
                          pin_memory=torch.cuda.is_available(),
                          drop_last=drop_last,
                          worker_init_fn=seed_worker,
                          generator=generator)

    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers,
                      collate_fn=collate_fn,
                      worker_init_fn=seed_worker,
                      generator=generator)

def get_dataloader_ddp(data_dir: str,
                   batch_size: int = 32,
                   num_workers: int = 4,
                   max_neg: Optional[int] = None,
                   feat_fn: Optional[Callable[[str], torch.Tensor]] = None,
                   rank: Optional[int] = None,            # DDP
                   world_size: Optional[int] = None,     # DDP
                   sampler = None,
                   mode: str = 'pair',
                   physical_config: Optional[PhysicalFeatureConfig] = None,
                   ) -> DataLoader:
    return get_dataloader(data_dir,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          max_neg=max_neg,
                          feat_fn=feat_fn,
                          sampler=sampler,
                          rank=rank,
                          world_size=world_size,
                          drop_last=sampler == 'distributed',
                          mode=mode,
                          physical_config=physical_config)

# ------------------------------------------------------------------ #
def _collate_fn(batch):
    """
    batch = [(anchor, other, label), ...]
    Returns dict of padded tensors.
    """
    anchors, others, labels = zip(*batch)

    def pack(structs):
        coords = [s.coords for s in structs]
        feats = [s.features for s in structs]
        lengths = torch.tensor([len(c) for c in coords])
        coords = torch.nn.utils.rnn.pad_sequence(coords, batch_first=True)
        feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
        mask = (torch.arange(coords.size(1)).unsqueeze(0) <
                lengths.unsqueeze(1))
        return coords, feats, mask

    a_coords, a_feats, a_mask = pack(anchors)
    o_coords, o_feats, o_mask = pack(others)
    # print(o_feats.shape)

    return dict(
        a_coords=a_coords,
        a_feats=a_feats,
        a_mask=a_mask,
        o_coords=o_coords,
        o_feats=o_feats,
        o_mask=o_mask,
        labels=torch.stack(labels)
    )


def _simclr_collate_fn(batch):
    view_a, view_b = zip(*batch)

    def pack(structs):
        coords = [s.coords for s in structs]
        feats = [s.features for s in structs]
        lengths = torch.tensor([len(c) for c in coords])
        coords = torch.nn.utils.rnn.pad_sequence(coords, batch_first=True)
        feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
        mask = (torch.arange(coords.size(1)).unsqueeze(0) <
                lengths.unsqueeze(1))
        return coords, feats, mask

    a_coords, a_feats, a_mask = pack(view_a)
    b_coords, b_feats, b_mask = pack(view_b)
    return dict(
        view1_coords=a_coords,
        view1_feats=a_feats,
        view1_mask=a_mask,
        view2_coords=b_coords,
        view2_feats=b_feats,
        view2_mask=b_mask,
    )


# ------------------------------------------------------------------ #
if __name__ == '__main__':
    # quick sanity check
    dl = get_dataloader('../solvation_structure/solvation_structures', batch_size=2)
    for batch in dl:
        print({k: v.shape for k, v in batch.items() if hasattr(v, 'shape')})
        break
