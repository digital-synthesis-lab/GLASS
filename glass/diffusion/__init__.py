"""Variance-exploding SDE diffuser used by the score-model training and sampling.

Ported from graphite (LLNL) — ``graphite/src/graphite/diffusion/general.py`` —
so glass does not depend on the external graphite package.
"""

from torch import Tensor

# Re-export for backwards compatibility
from .sampling import denoise_by_sde

__all__ = ["VarianceExplodingDiffuser", "denoise_by_sde"]


class VarianceExplodingDiffuser:
    """Variance-exploding SDE: sigma(t) = k*t."""

    def __init__(self, k: float = 1.0, t_min: float = 1e-3, t_max: float = 0.999) -> None:
        self.t_min = t_min
        self.t_max = t_max

        self.alpha = lambda t: 1
        self.sigma = lambda t: k * t
        self.f = lambda t: 0
        self.g2 = lambda t: 2 * (k**2) * t
        self.g = lambda t: self.g2(t) ** 0.5

    def forward_noise(self, x: Tensor, t: Tensor):
        import torch

        alpha = self.alpha(t)
        sigma = self.sigma(t)
        eps = torch.randn_like(x)
        return alpha * x + sigma * eps, eps
