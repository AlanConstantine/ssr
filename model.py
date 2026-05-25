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
        feat_dim: int = 10,
        dim: int = 128,
        depth: int = 4,
        num_nearest_neighbors: int = 12,
    ):
        super().__init__()
        self.egnn = EGNN_NetworkC(
            num_tokens=feat_dim,
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
    def __init__(self, encoder: SolvEncoder, dim: int = 128, proj_dim: int = 128):
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
    if not torch.any(labels > 0):
        return (z1.sum() + z2.sum()) * 0.0

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


def simclr_nt_xent(z1: torch.Tensor,
                   z2: torch.Tensor,
                   temperature: float = 0.1) -> torch.Tensor:
    """
    Standard SimCLR loss for two augmented views of the same batch.
    Each sample has exactly one positive: i <-> i + batch_size.
    """
    if z1.size(0) != z2.size(0):
        raise ValueError('z1 and z2 must have the same batch size')

    batch_size = z1.size(0)
    if batch_size < 2:
        raise ValueError('SimCLR loss requires batch_size >= 2')

    z = torch.cat([F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)], dim=0)
    logits = torch.mm(z, z.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    logits = logits.masked_fill(self_mask, float('-inf'))

    targets = torch.arange(2 * batch_size, device=z.device)
    targets = (targets + batch_size) % (2 * batch_size)
    return F.cross_entropy(logits, targets)


def bce_similarity_loss(z1: torch.Tensor,
                        z2: torch.Tensor,
                        labels: torch.Tensor,
                        scale: float = 10.0) -> torch.Tensor:
    logits = F.cosine_similarity(z1, z2, dim=-1) * scale
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def triplet_margin_from_pairs(z1: torch.Tensor,
                              z2: torch.Tensor,
                              labels: torch.Tensor,
                              margin: float = 0.2) -> torch.Tensor:
    positives = z2[labels > 0]
    anchors = z1[labels > 0]
    negatives = z2[labels <= 0]
    if anchors.numel() == 0 or negatives.numel() == 0:
        return (z1.sum() + z2.sum()) * 0.0
    repeat = min(anchors.size(0), negatives.size(0))
    return F.triplet_margin_loss(
        anchors[:repeat],
        positives[:repeat],
        negatives[:repeat],
        margin=margin,
    )


def compute_contrastive_loss(
    loss_name: str,
    z1: torch.Tensor,
    z2: torch.Tensor,
    labels: torch.Tensor | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    if loss_name == 'nt_xent':
        if labels is None:
            return simclr_nt_xent(z1, z2, temperature=temperature)
        return nt_xent(z1, z2, labels, temperature=temperature)
    if loss_name == 'simclr':
        return simclr_nt_xent(z1, z2, temperature=temperature)
    if loss_name == 'bce_similarity':
        if labels is None:
            raise ValueError('bce_similarity requires labels')
        return bce_similarity_loss(z1, z2, labels)
    if loss_name == 'triplet_margin':
        if labels is None:
            raise ValueError('triplet_margin requires labels')
        return triplet_margin_from_pairs(z1, z2, labels)
    raise ValueError(f'unsupported loss: {loss_name}')


# ------------------------------------------------------------------ #
if __name__ == '__main__':
    # quick shape test
    B, N, Ft = 3, 20, 10
    encoder = SolvEncoder(feat_dim=Ft, dim=64)
    model = SolvContrastive(encoder, dim=64, proj_dim=32)
    feats = torch.randn(B, N, Ft)
    coords = torch.randn(B, N, 3)
    mask = torch.ones(B, N).bool()
    out = model(feats, coords, mask)
    print('out shape:', out.shape)   # (B, 32)
