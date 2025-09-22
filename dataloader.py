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
import os
import re
import random
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from rdkit import Chem
from rdkit.Chem.rdchem import Atom
from tqdm import tqdm

# ------------------------------------------------------------------ #
# Helper: periodic table → one-hot index
ELEMENTS = ['H', 'C', 'N', 'O', 'F', 'Li', 'P', 'S', 'Cl', 'Br']


def element_one_hot(symbol: str) -> torch.Tensor:
    vec = torch.zeros(len(ELEMENTS))
    try:
        vec[ELEMENTS.index(symbol)] = 1.0
    except ValueError:              # unknown element → all zeros
        pass
    return vec


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
                 feat_fn: Optional[Callable[[Atom], torch.Tensor]] = None):
        self.path = xyz_path
        self.coords: torch.Tensor
        self.symbols: List[str]
        self.signature: str
        self.features: torch.Tensor
        self._load(feat_fn or element_one_hot)

    # -------------------------------------------------------------- #
    def _load(self, feat_fn: Callable[[Atom], torch.Tensor]) -> None:
        with open(self.path, 'r') as f:
            lines = [ln.strip() for ln in f.readlines()]

        # 2nd line → signature
        self.signature = lines[1].split(':')[-1].split('.')[0].strip()

        # skip first two lines
        coords, symbols, feats = [], [], []
        for ln in lines[2:]:
            if not ln:
                continue
            parts = ln.split()
            sym, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
            symbols.append(sym)
            coords.append([x, y, z])
            feats.append(feat_fn(Chem.Atom(sym)))

        self.coords = torch.tensor(coords, dtype=torch.float32)
        self.symbols = symbols
        self.features = torch.stack(feats)


# ------------------------------------------------------------------ #
class ContrastiveDataset(Dataset):
    """
    Builds positive/negative pairs on-the-fly.
    `__getitem__` returns (anchor, other, label)
    """
    def __init__(self,
                 data_dir: Path,
                 max_neg: Optional[int] = None,
                 feat_fn: Optional[Callable[[Atom], torch.Tensor]] = None):
        self.data_dir = Path(data_dir)
        self.paths = sorted(self.data_dir.glob('*.xyz'))
        self.max_neg = max_neg
        self.feat_fn = feat_fn

        # bucket by signature
        self.sig2idx: Dict[str, List[int]] = {}
        for idx, p in enumerate(self.paths):
            sig = self._signature_from_path(p)
            self.sig2idx.setdefault(sig, []).append(idx)

        self.all_indices = list(range(len(self.paths)))

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
        anchor_path = self.paths[idx]
        anchor_sig = self._signature_from_path(anchor_path)

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
                          if self._signature_from_path(self.paths[i]) != anchor_sig]
            if self.max_neg:
                candidates = random.sample(candidates,
                                           min(len(candidates), self.max_neg))
            other_idx = random.choice(candidates)
            label = torch.tensor(0.0)

        anchor = SolvationStructure(anchor_path, self.feat_fn)
        other = SolvationStructure(self.paths[other_idx], self.feat_fn)

        return anchor, other, label


# ------------------------------------------------------------------ #
def get_dataloader(data_dir: str,
                   batch_size: int = 32,
                   num_workers: int = 4,
                   max_neg: Optional[int] = None,
                   feat_fn: Optional[Callable[[Atom], torch.Tensor]] = None
                   ) -> DataLoader:
    ds = ContrastiveDataset(Path(data_dir), max_neg, feat_fn)
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers,
                      collate_fn=_collate_fn)

def get_dataloader_ddp(data_dir: str,
                   batch_size: int = 32,
                   num_workers: int = 4,
                   max_neg: Optional[int] = None,
                   feat_fn: Optional[Callable[[Atom], torch.Tensor]] = None,
                   rank: Optional[int] = None,            # DDP
                   world_size: Optional[int] = None,     # DDP
                   sampler = None
                   ) -> DataLoader:
    ds = ContrastiveDataset(Path(data_dir), max_neg, feat_fn)
    if sampler == 'distributed':
        assert rank is not None and world_size is not None, "rank and world_size must be provided for distributed sampler"
        data_sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, sampler=data_sampler, pin_memory=True, drop_last=True)
        dl.sampler = data_sampler  # 方便外部 set_epoch
    else:
        return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers,
                      collate_fn=_collate_fn)

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


# ------------------------------------------------------------------ #
if __name__ == '__main__':
    # quick sanity check
    dl = get_dataloader('../solvation_structure/solvation_structures', batch_size=2)
    for batch in dl:
        print({k: v.shape for k, v in batch.items() if hasattr(v, 'shape')})
        break