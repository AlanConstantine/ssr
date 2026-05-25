from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation import EvalConfig, run_evaluation_suite


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_pairs_per_anchor', type=int, default=2)
    parser.add_argument('--feature_mode', default='element_shell',
                        choices=['element', 'element_shell'])
    parser.add_argument('--li_cutoff', type=float, default=2.5)
    parser.add_argument('--center_on_li', action='store_true')
    parser.add_argument('--shell_radius', type=float, default=None)
    parser.add_argument('--device', default='auto')
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != 'auto'
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    cfg = EvalConfig(
        data_dir=args.data_dir,
        ckpt=args.ckpt,
        batch_size=args.batch_size,
        seed=args.seed,
        max_pairs_per_anchor=args.max_pairs_per_anchor,
        feature_mode=args.feature_mode,
        li_cutoff=args.li_cutoff,
        center_on_li=args.center_on_li,
        shell_radius=args.shell_radius,
    )
    result = run_evaluation_suite(cfg, device)
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(result['metrics'], f, indent=2, sort_keys=True)
    with open(out_dir / 'metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerows(sorted(result['metrics'].items()))
    np.save(out_dir / 'embeddings.npy', result['embeddings'])
    meta = result['metadata']
    with open(out_dir / 'metadata.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'signature', 'coordination_number'])
        writer.writerows(zip(meta['paths'], meta['signatures'], meta['coordination']))


if __name__ == '__main__':
    main()
