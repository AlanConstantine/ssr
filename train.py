"""
train.py
Single-GPU or multi-GPU training with
- progress bar
- TensorBoard logging
- Early stopping
"""

from __future__ import annotations
import os
import argparse
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from augment import AugmentConfig
from dataloader import get_dataloader
from model import SolvEncoder, SolvContrastive, compute_contrastive_loss
from physics import PhysicalFeatureConfig, feature_dim_for_mode
from utils import attach_defaults, load_config, merge_config, save_checkpoint, save_config, save_env_info, set_seed
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

# ------------------------------------------------------------------ #
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--log_dir', default='./runs')
    parser.add_argument('--mode', default='simclr',
                        choices=['pair', 'simclr', 'temporal', 'physical'])
    parser.add_argument('--loss', default='simclr',
                        choices=['nt_xent', 'simclr', 'bce_similarity', 'triplet_margin'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10,
                        help='early-stopping patience')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--feat_dim', type=int, default=10)
    parser.add_argument('--feature_mode', default='element',
                        choices=['element', 'element_shell'])
    parser.add_argument('--li_cutoff', type=float, default=2.5)
    parser.add_argument('--center_on_li', action='store_true')
    parser.add_argument('--shell_radius', type=float, default=None)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--num_nearest_neighbors', type=int, default=12)
    parser.add_argument('--augment_noise_std', type=float, default=0.01)
    parser.add_argument('--augment_translation_std', type=float, default=0.0)
    parser.add_argument('--augment_atom_dropout', type=float, default=0.0)
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1'])
    parser.add_argument('--local_rank', type=int, default=-1)   # DDP
    parser.add_argument('--debug_grad', action='store_true',
                        help='print parameters whose gradients are all zero')
    args = attach_defaults(parser, parser.parse_args())
    return merge_config(args, load_config(args.config))


# ------------------------------------------------------------------ #
def main():
    args = parse_args()
    if args.mode in {'temporal', 'physical'}:
        raise NotImplementedError(f'{args.mode} mode is planned but not implemented yet')
    if args.mode == 'simclr' and args.loss not in {'simclr', 'nt_xent'}:
        raise ValueError('simclr mode supports simclr/nt_xent loss')

    set_seed(args.seed)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    save_config(args, log_dir)
    save_env_info(log_dir)
    # ---------------------------------------------------------------- #
    # Device & DDP
    distributed = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device('cuda', local_rank)
    elif args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # ---------------------------------------------------------------- #
    # Data
    augment_config = AugmentConfig(
        noise_std=args.augment_noise_std,
        translation_std=args.augment_translation_std,
        atom_dropout=args.augment_atom_dropout,
    )
    physical_config = PhysicalFeatureConfig(
        mode=args.feature_mode,
        li_cutoff=args.li_cutoff,
        center_on_li=args.center_on_li,
        shell_radius=args.shell_radius,
    )
    dl = get_dataloader(args.data_dir,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        mode=args.mode,
                        augment_config=augment_config,
                        physical_config=physical_config,
                        sampler='distributed' if distributed else None,
                        rank=dist.get_rank() if distributed else None,
                        world_size=dist.get_world_size() if distributed else None,
                        drop_last=distributed or args.mode == 'simclr',
                        seed=args.seed)
    # ---------------------------------------------------------------- #
    # Model
    model_feat_dim = feature_dim_for_mode(args.feat_dim, args.feature_mode)
    encoder = SolvEncoder(feat_dim=model_feat_dim,
                          dim=args.dim,
                          depth=args.depth,
                          num_nearest_neighbors=args.num_nearest_neighbors)
    model = SolvContrastive(encoder, dim=args.dim, proj_dim=args.dim).to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    is_main = not distributed or dist.get_rank() == 0
    writer = SummaryWriter(args.log_dir) if is_main else None

    # ---------------------------------------------------------------- #
    # Training loop
    best_loss = 1e9
    patience_counter = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        if distributed:
            dl.sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        pbar = tqdm(dl, desc=f'Epoch {epoch}', disable=not is_main)
        for batch in pbar:
            if args.mode == 'pair':
                feats_a = batch['a_feats'].to(device).float()
                coords_a = batch['a_coords'].to(device).float()
                mask_a = batch['a_mask'].to(device)

                feats_o = batch['o_feats'].to(device).float()
                coords_o = batch['o_coords'].to(device).float()
                mask_o = batch['o_mask'].to(device)

                labels = batch['labels'].to(device)
                z_a = model(feats_a, coords_a, mask_a)
                z_o = model(feats_o, coords_o, mask_o)
                loss = compute_contrastive_loss(args.loss, z_a, z_o, labels, args.temperature)
            elif args.mode == 'simclr':
                feats_a = batch['view1_feats'].to(device).float()
                coords_a = batch['view1_coords'].to(device).float()
                mask_a = batch['view1_mask'].to(device)

                feats_o = batch['view2_feats'].to(device).float()
                coords_o = batch['view2_coords'].to(device).float()
                mask_o = batch['view2_mask'].to(device)
                z_a = model(feats_a, coords_a, mask_a)
                z_o = model(feats_o, coords_o, mask_o)
                loss = compute_contrastive_loss('simclr', z_a, z_o, temperature=args.temperature)
            else:
                raise ValueError(f'unsupported mode: {args.mode}')

            opt.zero_grad()
            loss.backward()
            if args.debug_grad and is_main:
                for name, param in model.named_parameters():
                    if param.grad is not None and (param.grad == 0).all():
                        print(f"Warning: {name} grad is all zero!")
            opt.step()

            running += loss.item()
            global_step += 1
            pbar.set_postfix(loss=loss.item())
            if writer is not None:
                writer.add_scalar('train/loss_step', loss.item(), global_step)

        avg_loss = running / len(dl)
        if distributed:
            avg_loss_tensor = torch.tensor([avg_loss], device=device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss_tensor.item() / dist.get_world_size()

        if writer is not None:
            writer.add_scalar('train/loss_epoch', avg_loss, epoch)
            print(f'Epoch {epoch}: avg loss {avg_loss:.4f}')

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            if is_main:
                save_checkpoint(log_dir / 'best.pt', model, opt, epoch, best_loss, args)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                if is_main:
                    print('Early stopping triggered.')
                break

    if writer is not None:
        writer.close()
    if distributed:
        dist.destroy_process_group()


# ------------------------------------------------------------------ #
if __name__ == '__main__':
    main()
