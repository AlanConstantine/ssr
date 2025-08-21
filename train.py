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
import time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from dataloader import get_dataloader
from model import SolvEncoder, SolvContrastive, nt_xent
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

# ------------------------------------------------------------------ #
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--log_dir', default='./runs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10,
                        help='early-stopping patience')
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1'])
    parser.add_argument('--local_rank', type=int, default=-1)   # DDP
    return parser.parse_args()


# ------------------------------------------------------------------ #
def main():
    args = parse_args()
    # ---------------------------------------------------------------- #
    # Device & DDP
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    if args.local_rank != -1:                # DDP
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device('cuda', args.local_rank)

    # ---------------------------------------------------------------- #
    # Data
    dl = get_dataloader(args.data_dir,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers)
    # ---------------------------------------------------------------- #
    # Model
    encoder = SolvEncoder(num_tokens=64, dim=128, depth=4, num_nearest_neighbors=12)
    model = SolvContrastive(encoder, dim=128, proj_dim=128).to(device)

    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    writer = SummaryWriter(args.log_dir)

    # ---------------------------------------------------------------- #
    # Training loop
    best_loss = 1e9
    patience_counter = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(dl, desc=f'Epoch {epoch}')
        for batch in pbar:
            feats_a = batch['a_feats'].to(device)
            coords_a = batch['a_coords'].to(device)
            mask_a = batch['a_mask'].to(device)

            feats_o = batch['o_feats'].to(device)
            coords_o = batch['o_coords'].to(device)
            mask_o = batch['o_mask'].to(device)

            labels = batch['labels'].to(device)

            z_a = model(feats_a, coords_a, mask_a)
            z_o = model(feats_o, coords_o, mask_o)

            loss = nt_xent(z_a, z_o, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()
            global_step += 1
            pbar.set_postfix(loss=loss.item())
            if args.local_rank in (-1, 0):
                writer.add_scalar('train/loss_step', loss.item(), global_step)

        avg_loss = running / len(dl)
        if args.local_rank in (-1, 0):
            writer.add_scalar('train/loss_epoch', avg_loss, epoch)
            print(f'Epoch {epoch}: avg loss {avg_loss:.4f}')

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), Path(args.log_dir) / 'best.pt')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print('Early stopping triggered.')
                break

    writer.close()


# ------------------------------------------------------------------ #
if __name__ == '__main__':
    main()