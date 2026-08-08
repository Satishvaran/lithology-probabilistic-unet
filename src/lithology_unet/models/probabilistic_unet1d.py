from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .unet1d import UNet1D


class GaussianEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, base: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(base, base * 2, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(base * 2, base * 4, 5, stride=2, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.mu = nn.Linear(base * 4, latent_dim)
        self.logvar = nn.Linear(base * 4, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h).clamp(-10, 10)


class ProbabilisticUNet1D(nn.Module):
    """Conditional-VAE U-Net that samples coherent sequence hypotheses."""

    def __init__(self, in_channels: int, num_classes: int, base: int = 32,
                 latent_dim: int = 8):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.unet = UNet1D(in_channels, num_classes, base, return_features=True)
        self.prior = GaussianEncoder(in_channels, latent_dim, base)
        self.posterior = GaussianEncoder(in_channels + num_classes, latent_dim, base)
        self.fuse = nn.Sequential(
            nn.Conv1d(base + latent_dim, base, 1), nn.GELU(),
            nn.Conv1d(base, num_classes, 1),
        )

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def _decode(self, features, z):
        zmap = z.unsqueeze(-1).expand(-1, -1, features.shape[-1])
        return self.fuse(torch.cat([features, zmap], dim=1))

    def forward(self, x, y=None):
        _, features = self.unet(x)
        prior_mu, prior_logvar = self.prior(x)
        if y is None:
            z = self.reparameterize(prior_mu, prior_logvar)
            return self._decode(features, z)
        safe_y = y.clamp_min(0)
        one_hot = F.one_hot(safe_y, self.num_classes).permute(0, 2, 1).float()
        one_hot = one_hot * (y.unsqueeze(1) >= 0)
        post_mu, post_logvar = self.posterior(torch.cat([x, one_hot], dim=1))
        z = self.reparameterize(post_mu, post_logvar)
        return self._decode(features, z), (prior_mu, prior_logvar), (post_mu, post_logvar)

    @torch.no_grad()
    def sample(self, x, n: int = 8):
        _, features = self.unet(x)
        mu, logvar = self.prior(x)
        return torch.stack([self._decode(features, self.reparameterize(mu, logvar)) for _ in range(n)])


def kl_normal(q_mu, q_logvar, p_mu, p_logvar):
    value = p_logvar - q_logvar + (q_logvar.exp() + (q_mu - p_mu).pow(2)) / p_logvar.exp() - 1
    return 0.5 * value.sum(dim=1).mean()
