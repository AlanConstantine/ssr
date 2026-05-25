from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from augment import AugmentConfig, augment_structure
from dataloader import build_fixed_pair_list, get_dataloader
from model import SolvContrastive, SolvEncoder, simclr_nt_xent
from physics import PhysicalFeatureConfig, append_shell_features, feature_dim_for_mode


def write_xyz(path: Path, signature: str) -> None:
    path.write_text(
        "\n".join([
            "3",
            f"signature: {signature}",
            "Li 0.0 0.0 0.0",
            "O 1.0 0.0 0.0",
            "C 0.0 1.0 0.0",
        ])
    )


def test_pair_and_simclr_dataloaders(tmp_path):
    write_xyz(tmp_path / "Frame0_Li_1DMC_id0.xyz", "Li_1DMC")
    write_xyz(tmp_path / "Frame1_Li_1DMC_id1.xyz", "Li_1DMC")
    write_xyz(tmp_path / "Frame2_Li_1EC_id2.xyz", "Li_1EC")

    pair_list = build_fixed_pair_list(str(tmp_path), max_pairs_per_anchor=1, seed=7)
    pair_dl = get_dataloader(str(tmp_path), batch_size=2, mode="pair", pair_list=pair_list)
    pair_batch = next(iter(pair_dl))
    assert pair_batch["a_feats"].shape[-1] == 10
    assert "labels" in pair_batch

    simclr_dl = get_dataloader(str(tmp_path), batch_size=2, mode="simclr")
    simclr_batch = next(iter(simclr_dl))
    assert simclr_batch["view1_feats"].shape[-1] == 10
    assert simclr_batch["view2_coords"].shape[-1] == 3

    shell_dl = get_dataloader(
        str(tmp_path),
        batch_size=2,
        mode="simclr",
        physical_config=PhysicalFeatureConfig(mode="element_shell", center_on_li=True),
    )
    shell_batch = next(iter(shell_dl))
    assert shell_batch["view1_feats"].shape[-1] == feature_dim_for_mode(10, "element_shell")


def test_model_and_simclr_loss():
    encoder = SolvEncoder(feat_dim=10, dim=16, depth=1, num_nearest_neighbors=2)
    model = SolvContrastive(encoder, dim=16, proj_dim=16)
    feats = torch.randn(3, 4, 10)
    coords = torch.randn(3, 4, 3)
    mask = torch.ones(3, 4).bool()
    z1 = model(feats, coords, mask)
    z2 = model(feats, coords, mask)
    assert z1.shape == (3, 16)
    assert torch.isfinite(simclr_nt_xent(z1, z2))


def test_augment_keeps_feature_alignment():
    coords = torch.randn(5, 3)
    feats = torch.randn(5, 10)
    out_coords, out_feats = augment_structure(
        coords,
        feats,
        AugmentConfig(atom_dropout=0.4, noise_std=0.0, translate=False),
    )
    assert out_coords.size(0) == out_feats.size(0)
    assert out_coords.size(1) == 3
    assert out_feats.size(1) == 10


def test_shell_feature_builder():
    coords = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    feats = torch.eye(10)[:3]
    out = append_shell_features(feats, coords, ["Li", "O", "C"], li_cutoff=2.5)
    assert out.shape == (3, 13)
    assert out[1, -1].item() == 1.0
    assert out[2, -1].item() == 0.0
