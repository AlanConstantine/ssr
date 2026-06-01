from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from augment import AugmentConfig, augment_structure
from dataloader import ATOM_PROPERTY_DIM, ELEMENTS, ContrastiveDataset, SolvationStructure, atom_identity_and_property_features, build_fixed_pair_list, get_dataloader, parse_temporal_metadata
from evaluation import acsf_like_descriptor, rdf_descriptor, shell_composition
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


def write_temporal_xyz(path: Path, signature: str, trajectory: str, frame: int) -> None:
    path.write_text(
        "\n".join([
            "3",
            f"signature: {signature} trajectory: {trajectory} frame: {frame}",
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
    pair_dl = get_dataloader(str(tmp_path), batch_size=2, mode="pair", pair_list=pair_list, num_workers=0)
    pair_batch = next(iter(pair_dl))
    assert pair_batch["a_feats"].shape[-1] == len(ELEMENTS)
    assert "labels" in pair_batch

    simclr_dl = get_dataloader(str(tmp_path), batch_size=2, mode="simclr", num_workers=0)
    simclr_batch = next(iter(simclr_dl))
    assert simclr_batch["view1_feats"].shape[-1] == len(ELEMENTS)
    assert simclr_batch["view2_coords"].shape[-1] == 3

    shell_dl = get_dataloader(
        str(tmp_path),
        batch_size=2,
        mode="simclr",
        num_workers=0,
        physical_config=PhysicalFeatureConfig(mode="element_shell", center_on_li=True),
    )
    shell_batch = next(iter(shell_dl))
    assert shell_batch["view1_feats"].shape[-1] == feature_dim_for_mode(len(ELEMENTS), "element_shell")

    atom_phys_dl = get_dataloader(
        str(tmp_path),
        batch_size=2,
        mode="simclr",
        num_workers=0,
        physical_config=PhysicalFeatureConfig(mode="atom_phys_shell", center_on_li=True),
    )
    atom_phys_batch = next(iter(atom_phys_dl))
    base_dim = len(ELEMENTS) + ATOM_PROPERTY_DIM
    assert atom_phys_batch["view1_feats"].shape[-1] == feature_dim_for_mode(base_dim, "atom_phys_shell")


def test_recursive_formula_directories_allow_duplicate_filenames(tmp_path):
    for formula_id in ["F001", "F002"]:
        formula_dir = tmp_path / formula_id
        formula_dir.mkdir()
        write_xyz(formula_dir / "Frame0_Li_1DMC_id1030.xyz", "Li_1DMC")
        write_xyz(formula_dir / "Frame1_Li_1EC_id1030.xyz", "Li_1EC")

    ds = ContrastiveDataset(tmp_path)
    assert len(ds.paths) == 4
    assert len({str(path) for path in ds.paths}) == 4
    struct = SolvationStructure(tmp_path / "F001" / "Frame0_Li_1DMC_id1030.xyz")
    assert struct.formulation_id == "F001"
    assert struct.signature == "Li_1DMC"


def test_temporal_metadata_and_dataloader(tmp_path):
    write_temporal_xyz(tmp_path / "TrajA_Frame0_Li_1DMC_id1030.xyz", "Li_1DMC", "A", 0)
    write_temporal_xyz(tmp_path / "TrajA_Frame2_Li_1DMC_id1030.xyz", "Li_1DMC", "A", 2)
    write_temporal_xyz(tmp_path / "TrajA_Frame18_Li_1DMC_id1030.xyz", "Li_1DMC", "A", 18)
    write_temporal_xyz(tmp_path / "TrajA_Frame20_Li_1DMC_id1030.xyz", "Li_1DMC", "A", 20)
    write_temporal_xyz(tmp_path / "TrajA_Frame0_Li_1DMC_id2040.xyz", "Li_1DMC", "A", 0)
    write_temporal_xyz(tmp_path / "TrajA_Frame2_Li_1DMC_id2040.xyz", "Li_1DMC", "A", 2)

    meta = parse_temporal_metadata(tmp_path / "TrajA_Frame20_Li_1DMC_id1030.xyz")
    assert meta.trajectory_id == "A"
    assert meta.center_id == "1030"
    assert meta.temporal_id == f"{tmp_path.name}:A:1030"
    assert meta.frame_index == 20
    assert meta.signature == "Li_1DMC"

    temporal_dl = get_dataloader(
        str(tmp_path),
        batch_size=2,
        mode="temporal",
        num_workers=0,
        temporal_positive_window=3,
        temporal_negative_min_gap=10,
    )
    batch = next(iter(temporal_dl))
    assert batch["a_feats"].shape[-1] == len(ELEMENTS)
    assert "labels" in batch
    assert set(batch["labels"].tolist()).issubset({0.0, 1.0})


def test_temporal_identity_includes_formula_directory(tmp_path):
    for formula_id in ["F001", "F002"]:
        formula_dir = tmp_path / formula_id
        formula_dir.mkdir()
        write_temporal_xyz(formula_dir / "TrajA_Frame0_Li_1DMC_id1030.xyz", "Li_1DMC", "A", 0)
        write_temporal_xyz(formula_dir / "TrajA_Frame2_Li_1DMC_id1030.xyz", "Li_1DMC", "A", 2)
        write_temporal_xyz(formula_dir / "TrajA_Frame18_Li_1DMC_id1030.xyz", "Li_1DMC", "A", 18)
        write_temporal_xyz(formula_dir / "TrajA_Frame20_Li_1DMC_id1030.xyz", "Li_1DMC", "A", 20)

    meta = parse_temporal_metadata(tmp_path / "F001" / "TrajA_Frame0_Li_1DMC_id1030.xyz")
    assert meta.formulation_id == "F001"
    assert meta.temporal_id == "F001:A:1030"

    temporal_dl = get_dataloader(
        str(tmp_path),
        batch_size=2,
        mode="temporal",
        num_workers=0,
        temporal_positive_window=3,
        temporal_negative_min_gap=10,
    )
    batch = next(iter(temporal_dl))
    assert "labels" in batch


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
    feats = torch.eye(len(ELEMENTS))[:3]
    out = append_shell_features(feats, coords, ["Li", "O", "C"], li_cutoff=2.5)
    assert out.shape == (3, len(ELEMENTS) + 3)
    assert out[1, -1].item() == 1.0
    assert out[2, -1].item() == 0.0


def test_stage4_descriptors(tmp_path):
    path = tmp_path / "Frame0_Li_1EC_id0.xyz"
    write_xyz(path, "Li_1EC")
    from dataloader import SolvationStructure

    struct = SolvationStructure(path)
    assert shell_composition(struct, li_cutoff=2.5)["O"] == 1.0
    assert rdf_descriptor(struct, bins=4, max_distance=4.0).shape == (4,)
    assert acsf_like_descriptor(struct, bins=4, max_distance=4.0).shape[0] == (len(ELEMENTS) - 1) * 4


def test_electrolyte_atom_features_include_boron_and_silicon():
    b = atom_identity_and_property_features("B")
    si = atom_identity_and_property_features("Si")
    unknown = atom_identity_and_property_features("Xe")
    assert b.shape[0] == len(ELEMENTS) + ATOM_PROPERTY_DIM
    assert si.shape[0] == len(ELEMENTS) + ATOM_PROPERTY_DIM
    assert b.sum().item() > 0
    assert si.sum().item() > 0
    assert unknown.sum().item() == 0
