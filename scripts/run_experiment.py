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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--skip_train', action='store_true')
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print('+ ' + shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    commands = []

    train_cmd = [
        sys.executable, 'train.py',
        '--config', args.config,
        '--data_dir', args.data_dir,
        '--log_dir', str(log_dir),
        '--seed', str(args.seed),
        '--device', args.device,
    ]
    if args.epochs is not None:
        train_cmd += ['--epochs', str(args.epochs)]
    if args.batch_size is not None:
        train_cmd += ['--batch_size', str(args.batch_size)]
    commands.append(train_cmd)

    eval_dir = log_dir / 'evaluation'
    eval_cmd = [
        sys.executable, 'scripts/run_evaluation_suite.py',
        '--data_dir', args.data_dir,
        '--ckpt', str(log_dir / 'best.pt'),
        '--out_dir', str(eval_dir),
        '--seed', str(args.seed),
        '--device', args.device,
        '--feature_mode', 'element_shell',
        '--center_on_li',
    ]
    commands.append(eval_cmd)

    desc_cmd = [
        sys.executable, 'scripts/compute_physical_descriptors.py',
        '--data_dir', args.data_dir,
        '--out_csv', str(log_dir / 'physical_descriptors.csv'),
    ]
    commands.append(desc_cmd)

    with open(log_dir / 'commands.json', 'w') as f:
        json.dump(commands, f, indent=2)

    if not args.skip_train:
        run(train_cmd)
    run(eval_cmd)
    run(desc_cmd)


if __name__ == '__main__':
    main()
