from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataloader import ContrastiveDataset, _collate_fn
from model import SolvContrastive, SolvEncoder
from utils import load_model_state, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != 'auto'
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    dataset = ContrastiveDataset(Path(args.data_dir), pair_list=[])
    loader = torch.utils.data.DataLoader(
        [(dataset.paths[i], dataset.signatures[i]) for i in range(len(dataset.paths))],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    encoder = SolvEncoder(feat_dim=10, dim=128, depth=4, num_nearest_neighbors=12)
    model = SolvContrastive(encoder, dim=128, proj_dim=128)
    model.load_state_dict(load_model_state(args.ckpt, map_location='cpu'))
    model.to(device)
    model.eval()

    embeddings = []
    metadata = []
    with torch.no_grad():
        for paths, signatures in loader:
            batch_structs = []
            for path in paths:
                from dataloader import SolvationStructure
                batch_structs.append(SolvationStructure(Path(path)))
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


if __name__ == '__main__':
    main()
