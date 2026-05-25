from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in minimal environments
    yaml = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    if yaml is None:
        raise ImportError('PyYAML is required to load config files. Install requirements.txt first.')
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def merge_config(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    values = vars(args).copy()
    for key, value in config.items():
        if key not in values or values[key] == get_parser_default(args, key):
            values[key] = value
    return argparse.Namespace(**values)


def get_parser_default(args: argparse.Namespace, key: str) -> Any:
    defaults = getattr(args, '_defaults', {})
    return defaults.get(key)


def attach_defaults(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    args._defaults = {
        action.dest: action.default
        for action in parser._actions
        if action.dest != 'help'
    }
    return args


def save_config(args: argparse.Namespace, log_dir: Path) -> None:
    serializable = {
        key: value
        for key, value in vars(args).items()
        if not key.startswith('_')
    }
    with open(log_dir / 'config.json', 'w') as f:
        json.dump(serializable, f, indent=2, sort_keys=True)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    args: argparse.Namespace,
) -> None:
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save({
        'model_state': state_dict,
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'best_metric': best_metric,
        'args': {k: v for k, v in vars(args).items() if not k.startswith('_')},
    }, path)


def load_model_state(path: str, map_location: str | torch.device = 'cpu') -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        return checkpoint['model_state']
    return checkpoint


def env_info() -> dict[str, Any]:
    return {
        'python': os.sys.version,
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def save_env_info(log_dir: Path) -> None:
    with open(log_dir / 'env.json', 'w') as f:
        json.dump(env_info(), f, indent=2, sort_keys=True)
