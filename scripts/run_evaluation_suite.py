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
from utils import config_hash, env_info, git_commit_hash


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_pairs_per_anchor', type=int, default=2)
    parser.add_argument('--feature_mode', default='atom_phys_shell',
                        choices=['element', 'element_shell', 'atom_phys', 'atom_phys_shell'])
    parser.add_argument('--li_cutoff', type=float, default=2.5)
    parser.add_argument('--center_on_li', action='store_true')
    parser.add_argument('--shell_radius', type=float, default=None)
    parser.add_argument('--rdf_bins', type=int, default=12)
    parser.add_argument('--rdf_max_distance', type=float, default=6.0)
    parser.add_argument('--downstream_csv', default=None)
    parser.add_argument('--downstream_target', default=None)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--num_nearest_neighbors', type=int, default=12)
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
        dim=args.dim,
        depth=args.depth,
        num_nearest_neighbors=args.num_nearest_neighbors,
        rdf_bins=args.rdf_bins,
        rdf_max_distance=args.rdf_max_distance,
        downstream_csv=args.downstream_csv,
        downstream_target=args.downstream_target,
    )
    result = run_evaluation_suite(cfg, device)
    config_payload = vars(args).copy()
    metadata_payload = {
        'git_commit': git_commit_hash(),
        'config_hash': config_hash(config_payload),
        'config': config_payload,
        'environment': env_info(),
        'downstream': result.get('downstream', {}),
    }
    with open(out_dir / 'run_metadata.json', 'w') as f:
        json.dump(metadata_payload, f, indent=2, sort_keys=True)
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
        writer.writerow(['path', 'formulation_id', 'signature', 'coordination_number', 'shell_state'])
        writer.writerows(zip(meta['paths'], meta['formulation_ids'], meta['signatures'], meta['coordination'], meta['shell_state']))
    with open(out_dir / 'embeddings.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        dim = result['embeddings'].shape[1]
        writer.writerow(['path', 'formulation_id', 'signature'] + [f'z{i}' for i in range(dim)])
        for path, formulation_id, signature, row in zip(meta['paths'], meta['formulation_ids'], meta['signatures'], result['embeddings']):
            writer.writerow([path, formulation_id, signature] + [float(value) for value in row])
    np.save(out_dir / 'rdf_descriptors.npy', meta['rdf'])
    np.save(out_dir / 'acsf_like_descriptors.npy', meta['acsf_like'])
    with open(out_dir / 'descriptor_keys.json', 'w') as f:
        json.dump({
            'composition_keys': meta['composition_keys'],
            'shell_composition_keys': meta['shell_composition_keys'],
            'rdf_bins': args.rdf_bins,
            'rdf_max_distance': args.rdf_max_distance,
        }, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
