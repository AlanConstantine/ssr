"""
model.py
Encoder = EGNN_Network from egnn-pytorch
Contrastive projection head + NT-Xent loss
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
# from egnn_pytorch import EGNN_Network
from egnn_torch_c import EGNN_NetworkC


class SolvEncoder(nn.Module):
    """
    Wrap EGNN_Network and produce a single vector representation
    that is invariant to rotation/translation.
    """
    def __init__(
        self,
        num_tokens: int = 64,               # size of atom feature vocab
        feat_dim: int = 10,
        dim: int = 128,
        depth: int = 4,
        num_nearest_neighbors: int = 12,
    ):
        super().__init__()
        self.egnn = EGNN_NetworkC(
            num_tokens=num_tokens,
            feat_dim=feat_dim,
            dim=dim,
            depth=depth,
            num_nearest_neighbors=num_nearest_neighbors,
            norm_coors=True,
            coor_weights_clamp_value=2.0,
            update_coors=True,
            update_feats=True
        )

    def forward(self,
                feats: torch.Tensor,
                coords: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        feats:  (B, N, F)
        coords: (B, N, 3)
        mask:   (B, N)
        Returns: (B, dim)
        """
        out_feats, _ = self.egnn(feats, coords, mask=mask)
        # global mean pooling over valid nodes
        out = (out_feats * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return out


class ContrastiveHead(nn.Module):
    """
    Small MLP that maps encoder output to the final embedding space
    for NT-Xent.
    """
    def __init__(self, dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class SolvContrastive(nn.Module):
    """
    End-to-end model.
    """
    def __init__(self, encoder: SolvEncoder, dim, proj_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.head = ContrastiveHead(dim, proj_dim)

    def forward(self,
                feats: torch.Tensor,
                coords: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        z = self.encoder(feats, coords, mask)
        return self.head(z)


# ------------------------------------------------------------------ #
def nt_xent(z1: torch.Tensor,
            z2: torch.Tensor,
            labels: torch.Tensor,
            temperature: float = 0.1) -> torch.Tensor:
    """
    NT-Xent for pairs (anchor, other) with given labels.
    Only positive pairs (label==1) contribute to the numerator.
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)               # 2B x d
    sim = torch.mm(z, z.T) / temperature         # 2B x 2B

    # mask out diagonal
    logits_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim = sim - logits_max.detach()

    mask = torch.eye(2 * batch_size,
                     device=sim.device).bool()
    sim.masked_fill_(mask, float('-inf'))

    # labels = 1 for positive, 0 for negative
    # positive mask: [B, 2B]  only (i, i+B) or (i+B, i)
    pos_mask = torch.zeros_like(sim)
    idx = torch.arange(batch_size, device=sim.device)
    pos_mask[idx, idx + batch_size] = labels      # anchor vs other
    pos_mask[idx + batch_size, idx] = labels
    pos_mask = pos_mask.bool()

    exp = torch.exp(sim)
    num = (exp * pos_mask).sum(1)
    den = exp.sum(1)
    loss = -torch.log(num / den + 1e-8)
    # only keep positive pairs
    keep = torch.cat([labels, labels]).bool()
    return loss[keep].mean()


# ------------------------------------------------------------------ #
if __name__ == '__main__':
    # quick shape test
    B, N, Ft = 3, 20, 10
    encoder = SolvEncoder(num_tokens=Ft, dim=64)
    model = SolvContrastive(encoder, dim=64, proj_dim=32)
    feats = torch.randint(0, Ft, (B, N))
    coords = torch.randn(B, N, 3)
    mask = torch.ones(B, N).bool()
    out = model(feats, coords, mask)
    print('out shape:', out.shape)   # (B, 32)