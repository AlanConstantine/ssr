from __future__ import annotations
import os
import argparse
import time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler
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

def main():
    args = parse_args()

    # ---------------------------------------------------------------- #
    # DDP setup
    distributed = False
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        distributed = True
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device('cuda', args.local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')

    # ---------------------------------------------------------------- #
    # Data
    if distributed:
        # Use DistributedSampler for multi-GPU
        dl = get_dataloader(args.data_dir,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            sampler='distributed',
                            rank=args.local_rank,
                            world_size=dist.get_world_size())
    else:
        dl = get_dataloader(args.data_dir,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)

    # ---------------------------------------------------------------- #
    # Model
    encoder = SolvEncoder(num_tokens=64, dim=128, depth=4, num_nearest_neighbors=12)
    model = SolvContrastive(encoder, dim=128, proj_dim=128).to(device)

    if distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if not distributed or args.local_rank == 0:
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

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
        pbar = tqdm(dl, desc=f'Epoch {epoch}', disable=distributed and args.local_rank != 0)
        for batch in pbar:
            feats_a = batch['a_feats'].to(device).float()
            coords_a = batch['a_coords'].to(device).float()
            mask_a = batch['a_mask'].to(device)

            feats_o = batch['o_feats'].to(device).float()
            coords_o = batch['o_coords'].to(device).float()
            mask_o = batch['o_mask'].to(device)

            labels = batch['labels'].to(device)

            z_a = model(feats_a, coords_a, mask_a)
            z_o = model(feats_o, coords_o, mask_o)

            loss = nt_xent(z_a, z_o, labels)

            opt.zero_grad()
            loss.backward()
            # 梯度检查：输出所有梯度全为0的参数名
            for name, param in model.named_parameters():
                if param.grad is not None and (param.grad == 0).all():
                    print(f"Warning: {name} grad is all zero!")
            opt.step()

            running += loss.item()
            global_step += 1
            if writer is not None:
                writer.add_scalar('train/loss_step', loss.item(), global_step)
            pbar.set_postfix(loss=loss.item())

        # 计算所有进程的平均loss
        avg_loss = running / len(dl)
        if distributed:
            avg_loss_tensor = torch.tensor([avg_loss], device=device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss_tensor.item() / dist.get_world_size()

        if writer is not None:
            writer.add_scalar('train/loss_epoch', avg_loss, epoch)
            print(f'Epoch {epoch}: avg loss {avg_loss:.4f}')

        # Early stopping（仅主进程保存模型）
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            if writer is not None:
                torch.save(model.module.state_dict() if distributed else model.state_dict(),
                           Path(args.log_dir) / 'best.pt')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                if writer is not None:
                    print('Early stopping triggered.')
                break

    if writer is not None:
        writer.close()
    if distributed:
        dist.destroy_process_group()


# ------------------------------------------------------------------ #
if __name__ == '__main__':
    main()

"""
torchrun --nproc_per_node=4 train.py --data_dir ./data --log_dir ./runs

CUDA_VISIBLE_DEVICES=1,3 torchrun --nproc_per_node=2 ddp_train.py --data_dir ./data --log_dir ./runs
"""