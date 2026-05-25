from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataloader import ContrastiveDataset, SolvationStructure
from physics import nearest_li_distance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out_csv', required=True)
    parser.add_argument('--li_cutoff', type=float, default=2.5)
    return parser.parse_args()


def main():
    args = parse_args()
    ds = ContrastiveDataset(Path(args.data_dir), pair_list=[])
    rows = []
    for path, signature in zip(ds.paths, ds.signatures):
        struct = SolvationStructure(path)
        min_dist = nearest_li_distance(struct.coords, struct.symbols)
        non_li = torch.tensor([sym != 'Li' for sym in struct.symbols], dtype=torch.bool)
        finite_non_li = min_dist[non_li & torch.isfinite(min_dist)]
        coordination = int((finite_non_li <= args.li_cutoff).sum().item())
        rows.append({
            'path': str(path),
            'signature': signature,
            'num_atoms': len(struct.symbols),
            'num_li': sum(sym == 'Li' for sym in struct.symbols),
            'li_cutoff': args.li_cutoff,
            'coordination_number': coordination,
            'nearest_li_dist_mean': float(finite_non_li.mean().item()) if finite_non_li.numel() else '',
            'nearest_li_dist_min': float(finite_non_li.min().item()) if finite_non_li.numel() else '',
        })

    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
