from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--downstream_csv', default=None)
    parser.add_argument('--downstream_target', default=None)
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print('+ ' + shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = log_dir / 'checkpoints'
    metrics_dir = log_dir / 'metrics'
    embeddings_dir = log_dir / 'embeddings'
    plots_dir = log_dir / 'plots'
    for path in (checkpoint_dir, metrics_dir, embeddings_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)
    commands = []

    train_cmd = [
        sys.executable, 'train.py',
        '--config', args.config,
        '--data_dir', args.data_dir,
        '--log_dir', str(checkpoint_dir),
        '--seed', str(args.seed),
        '--device', args.device,
        '--num_workers', str(args.num_workers),
    ]
    if args.epochs is not None:
        train_cmd += ['--epochs', str(args.epochs)]
    if args.batch_size is not None:
        train_cmd += ['--batch_size', str(args.batch_size)]
    commands.append(train_cmd)

    eval_cmd = [
        sys.executable, 'scripts/run_evaluation_suite.py',
        '--data_dir', args.data_dir,
        '--ckpt', str(checkpoint_dir / 'best.pt'),
        '--out_dir', str(metrics_dir),
        '--seed', str(args.seed),
        '--device', args.device,
        '--feature_mode', 'element_shell',
        '--center_on_li',
    ]
    if args.downstream_csv and args.downstream_target:
        eval_cmd += ['--downstream_csv', args.downstream_csv, '--downstream_target', args.downstream_target]
    commands.append(eval_cmd)

    desc_cmd = [
        sys.executable, 'scripts/compute_physical_descriptors.py',
        '--data_dir', args.data_dir,
        '--out_csv', str(metrics_dir / 'physical_descriptors.csv'),
    ]
    commands.append(desc_cmd)

    export_cmd = [
        sys.executable, 'scripts/export_embeddings.py',
        '--data_dir', args.data_dir,
        '--ckpt', str(checkpoint_dir / 'best.pt'),
        '--out_dir', str(embeddings_dir),
        '--seed', str(args.seed),
        '--device', args.device,
        '--feature_mode', 'element_shell',
        '--center_on_li',
    ]
    commands.append(export_cmd)

    plot_cmd = [
        sys.executable, 'scripts/visualize_embeddings.py',
        '--embeddings', str(embeddings_dir / 'embeddings.npy'),
        '--metadata_csv', str(embeddings_dir / 'metadata.csv'),
        '--out_dir', str(plots_dir),
    ]
    commands.append(plot_cmd)

    with open(log_dir / 'commands.json', 'w') as f:
        json.dump(commands, f, indent=2)

    if not args.skip_train:
        run(train_cmd)
    run(eval_cmd)
    run(desc_cmd)
    run(export_cmd)
    run(plot_cmd)


if __name__ == '__main__':
    main()
