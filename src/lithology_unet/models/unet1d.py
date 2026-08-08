from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels), nn.GELU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet1D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base: int = 32,
                 return_features: bool = False):
        super().__init__()
        self.return_features = return_features
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.bridge = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool1d(2)
        self.up3 = nn.ConvTranspose1d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose1d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose1d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.head = nn.Conv1d(base, num_classes, 1)

    @staticmethod
    def _match(x, ref):
        return F.interpolate(x, size=ref.shape[-1], mode="linear", align_corners=False)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bridge(self.pool(e3))
        d3 = self.dec3(torch.cat([self._match(self.up3(b), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._match(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._match(self.up1(d2), e1), e1], dim=1))
        logits = self.head(d1)
        return (logits, d1) if self.return_features else logits
