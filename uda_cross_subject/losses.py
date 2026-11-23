"""Losses used by the paper: CORAL, MMD and gradient reversal for DANN.

Paper equations (sections 4.5 and 4.6)::

    L_CORAL = 1/(4 d^2) * ||C_s - C_t||_F^2
    L_MMD   = || mean(phi(z_s)) - mean(phi(z_t)) ||_H^2
    L_DANN  = L_cls - lambda * L_domain   (implemented with gradient reversal)
"""

import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    """Identity in forward, sign-flipped gradient in backward."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


def coral_loss(z_source: torch.Tensor, z_target: torch.Tensor,
               eps: float = 1e-4) -> torch.Tensor:
    """Squared Frobenius distance between source/target covariances."""
    zs = z_source - z_source.mean(dim=0, keepdim=True)
    zt = z_target - z_target.mean(dim=0, keepdim=True)

    ns = max(zs.shape[0] - 1, 1)
    nt = max(zt.shape[0] - 1, 1)
    d = zs.shape[1]

    cs = (zs.T @ zs) / ns + eps * torch.eye(
        d, device=zs.device, dtype=zs.dtype
    )
    ct = (zt.T @ zt) / nt + eps * torch.eye(
        d, device=zt.device, dtype=zt.dtype
    )
    return ((cs - ct) ** 2).sum() / (4.0 * d * d)


def _pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True)
    dist = x_norm + y_norm.T - 2.0 * x @ y.T
    return torch.clamp(dist, min=0.0)


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-_pairwise_sq_dist(x, y) / (2.0 * sigma ** 2))


def mmd_loss(z_source: torch.Tensor, z_target: torch.Tensor,
             sigmas=(0.1, 0.5, 1.0, 2.0), kernel: str = "rbf") -> torch.Tensor:
    """Multi-kernel squared MMD between source and target representations."""
    if kernel.lower() != "rbf":
        raise ValueError(f"Unsupported MMD kernel: {kernel}")

    sigmas = tuple(float(s) for s in sigmas)
    if not sigmas:
        # Median heuristic fallback.
        dist = _pairwise_sq_dist(z_source, z_source)
        median = torch.median(dist[dist > 0]).sqrt().clamp(min=1e-3)
        sigmas = (float(median.item()),)

    losses = []
    for sigma in sigmas:
        kss = rbf_kernel(z_source, z_source, sigma)
        ktt = rbf_kernel(z_target, z_target, sigma)
        kst = rbf_kernel(z_source, z_target, sigma)
        losses.append(kss.mean() + ktt.mean() - 2.0 * kst.mean())
    return torch.stack(losses).mean()
