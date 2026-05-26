from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataloader import ATOM_PROPERTY_DIM, ELEMENTS, ContrastiveDataset, _collate_fn
from model import SolvContrastive, SolvEncoder
from physics import PhysicalFeatureConfig, feature_dim_for_mode
from utils import config_hash, git_commit_hash, load_model_state, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--feat_dim', type=int, default=len(ELEMENTS))
    parser.add_argument('--feature_mode', default='atom_phys_shell',
                        choices=['element', 'element_shell', 'atom_phys', 'atom_phys_shell'])
    parser.add_argument('--li_cutoff', type=float, default=2.5)
    parser.add_argument('--center_on_li', action='store_true')
    parser.add_argument('--shell_radius', type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != 'auto'
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    physical_config = PhysicalFeatureConfig(
        mode=args.feature_mode,
        li_cutoff=args.li_cutoff,
        center_on_li=args.center_on_li,
        shell_radius=args.shell_radius,
    )
    if args.feature_mode.startswith('atom_phys') and args.feat_dim == len(ELEMENTS):
        args.feat_dim = len(ELEMENTS) + ATOM_PROPERTY_DIM
    dataset = ContrastiveDataset(Path(args.data_dir), physical_config=physical_config, pair_list=[])
    loader = torch.utils.data.DataLoader(
        list(range(len(dataset.paths))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    encoder = SolvEncoder(feat_dim=feature_dim_for_mode(args.feat_dim, args.feature_mode),
                          dim=128, depth=4, num_nearest_neighbors=12)
    model = SolvContrastive(encoder, dim=128, proj_dim=128)
    model.load_state_dict(load_model_state(args.ckpt, map_location='cpu'))
    model.to(device)
    model.eval()

    embeddings = []
    metadata = []
    with torch.no_grad():
        for indices in loader:
            batch_structs = []
            paths = [dataset.paths[int(idx)] for idx in indices]
            signatures = [dataset.signatures[int(idx)] for idx in indices]
            for path in paths:
                from dataloader import SolvationStructure
                batch_structs.append(SolvationStructure(Path(path), physical_config=physical_config))
            batch = _collate_fn([(s, s, torch.tensor(1.0)) for s in batch_structs])
            feats = batch['a_feats'].to(device).float()
            coords = batch['a_coords'].to(device).float()
            mask = batch['a_mask'].to(device)
            z = model(feats, coords, mask)
            embeddings.append(z.cpu().numpy())
            metadata.extend(zip([str(p) for p in paths], list(signatures)))

    arr = np.concatenate(embeddings, axis=0)
    np.save(out_dir / 'embeddings.npy', arr)
    with open(out_dir / 'metadata.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'signature'])
        writer.writerows(metadata)
    with open(out_dir / 'embeddings.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'signature'] + [f'z{i}' for i in range(arr.shape[1])])
        for (path, signature), row in zip(metadata, arr):
            writer.writerow([path, signature] + [float(value) for value in row])
    export_config = vars(args).copy()
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump({
            'git_commit': git_commit_hash(),
            'config_hash': config_hash(export_config),
            'config': export_config,
            'num_samples': int(arr.shape[0]),
            'embedding_dim': int(arr.shape[1]),
        }, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
