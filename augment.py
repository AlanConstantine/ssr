from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AugmentConfig:
    rotate: bool = True
    translate: bool = True
    translation_std: float = 0.0
    noise_std: float = 0.01
    atom_dropout: float = 0.0
    min_atoms: int = 2


def random_rotation_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    q = torch.randn(4, device=device, dtype=dtype)
    q = q / q.norm().clamp_min(1e-8)
    w, x, y, z = q
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
    ])


def augment_structure(
    coords: torch.Tensor,
    feats: torch.Tensor,
    cfg: Optional[AugmentConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = cfg or AugmentConfig()
    out_coords = coords.clone()
    out_feats = feats.clone()

    if cfg.rotate:
        out_coords = out_coords @ random_rotation_matrix(out_coords.device, out_coords.dtype).T

    if cfg.noise_std > 0:
        out_coords = out_coords + torch.randn_like(out_coords) * cfg.noise_std

    if cfg.translate:
        if cfg.translation_std > 0:
            shift = torch.randn(1, 3, device=out_coords.device, dtype=out_coords.dtype) * cfg.translation_std
        else:
            shift = torch.randn(1, 3, device=out_coords.device, dtype=out_coords.dtype)
        out_coords = out_coords + shift

    if cfg.atom_dropout > 0 and out_coords.size(0) > cfg.min_atoms:
        keep = torch.rand(out_coords.size(0), device=out_coords.device) >= cfg.atom_dropout
        if keep.sum().item() < cfg.min_atoms:
            perm = torch.randperm(out_coords.size(0), device=out_coords.device)
            keep = torch.zeros_like(keep)
            keep[perm[:cfg.min_atoms]] = True
        out_coords = out_coords[keep]
        out_feats = out_feats[keep]

    return out_coords, out_feats
