from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class PhysicalFeatureConfig:
    mode: str = 'element'
    li_cutoff: float = 2.5
    center_on_li: bool = False
    shell_radius: float | None = None


def nearest_li_distance(coords: torch.Tensor, symbols: Sequence[str]) -> torch.Tensor:
    li_indices = [idx for idx, sym in enumerate(symbols) if sym == 'Li']
    if not li_indices:
        return torch.full((coords.size(0),), float('inf'), dtype=coords.dtype)
    li_coords = coords[torch.tensor(li_indices, dtype=torch.long)]
    dists = torch.cdist(coords, li_coords)
    return dists.min(dim=1).values


def append_shell_features(
    features: torch.Tensor,
    coords: torch.Tensor,
    symbols: Sequence[str],
    li_cutoff: float = 2.5,
) -> torch.Tensor:
    """
    Add xyz-derived physical priors:
      - is Li atom
      - nearest Li distance scaled by cutoff
      - in Li first shell flag
    """
    min_dist = nearest_li_distance(coords, symbols)
    finite_dist = torch.where(torch.isfinite(min_dist), min_dist, torch.zeros_like(min_dist))
    is_li = torch.tensor([sym == 'Li' for sym in symbols], dtype=features.dtype).unsqueeze(-1)
    in_shell = ((min_dist <= li_cutoff) & ~is_li.squeeze(-1).bool()).to(features.dtype).unsqueeze(-1)
    scaled_dist = (finite_dist / max(li_cutoff, 1e-8)).clamp(max=10.0).unsqueeze(-1)
    return torch.cat([features, is_li, scaled_dist, in_shell], dim=-1)


def center_and_crop_on_li(
    coords: torch.Tensor,
    features: torch.Tensor,
    symbols: Sequence[str],
    shell_radius: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    li_indices = [idx for idx, sym in enumerate(symbols) if sym == 'Li']
    if not li_indices:
        return coords, features, list(symbols)

    center = coords[li_indices].mean(dim=0, keepdim=True)
    centered = coords - center
    if shell_radius is None:
        return centered, features, list(symbols)

    keep = centered.norm(dim=-1) <= shell_radius
    if not torch.any(keep):
        keep[li_indices[0]] = True
    cropped_symbols = [sym for sym, keep_flag in zip(symbols, keep.tolist()) if keep_flag]
    return centered[keep], features[keep], cropped_symbols


def feature_dim_for_mode(base_dim: int, mode: str) -> int:
    if mode == 'element':
        return base_dim
    if mode == 'element_shell':
        return base_dim + 3
    raise ValueError(f'unsupported feature mode: {mode}')
