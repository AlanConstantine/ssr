"""
eval.py
Evaluate the trained encoder on a held-out set.
Computes
  - average cosine similarity for positive vs negative pairs
  - ROC-AUC
"""

from __future__ import annotations
import argparse
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from dataloader import build_fixed_pair_list, get_dataloader
from model import SolvEncoder, SolvContrastive
from utils import load_model_state

# ------------------------------------------------------------------ #
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_pairs_per_anchor', type=int, default=2)
    return parser.parse_args()


# ------------------------------------------------------------------ #
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device != 'auto'
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))

    pair_list = build_fixed_pair_list(args.data_dir,
                                      max_pairs_per_anchor=args.max_pairs_per_anchor,
                                      seed=args.seed)
    dl = get_dataloader(args.data_dir,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        pair_list=pair_list,
                        mode='pair',
                        seed=args.seed)

    encoder = SolvEncoder(feat_dim=10, dim=128, depth=4, num_nearest_neighbors=12)
    model = SolvContrastive(encoder, dim=128, proj_dim=128)
    model.load_state_dict(load_model_state(args.ckpt, map_location='cpu'))
    model.to(device)
    model.eval()

    sims, labels = [], []
    with torch.no_grad():
        for batch in dl:
            feats_a = batch['a_feats'].to(device)
            coords_a = batch['a_coords'].to(device)
            mask_a = batch['a_mask'].to(device)

            feats_o = batch['o_feats'].to(device)
            coords_o = batch['o_coords'].to(device)
            mask_o = batch['o_mask'].to(device)

            z_a = model(feats_a, coords_a, mask_a)
            z_o = model(feats_o, coords_o, mask_o)

            sim = torch.nn.functional.cosine_similarity(z_a, z_o, dim=-1)
            sims.append(sim.cpu().numpy())
            labels.append(batch['labels'].numpy())

    sims = np.concatenate(sims)
    labels = np.concatenate(labels)

    if not np.any(labels == 1) or not np.any(labels == 0):
        raise ValueError('evaluation requires both positive and negative pairs')

    pos_sim = sims[labels == 1].mean()
    neg_sim = sims[labels == 0].mean()
    auc = roc_auc_score(labels, sims)

    print(f'Positive cosine similarity: {pos_sim:.4f}')
    print(f'Negative cosine similarity: {neg_sim:.4f}')
    print(f'ROC-AUC: {auc:.4f}')


# ------------------------------------------------------------------ #
if __name__ == '__main__':
    main()
