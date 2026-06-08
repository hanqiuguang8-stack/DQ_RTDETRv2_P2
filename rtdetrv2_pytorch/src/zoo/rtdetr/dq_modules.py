"""Lightweight DQ-DETR style modules for RT-DETRv2-P2."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext


class CategoricalCountingModule(nn.Module):
    def __init__(
        self,
        in_channels=256,
        hidden_channels=256,
        num_count_classes=4,
        use_dilated=True,
    ):
        super().__init__()
        dilation = 2 if use_dilated else 1
        padding = dilation

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels, hidden_channels, kernel_size=3,
                padding=padding, dilation=dilation, bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels, hidden_channels, kernel_size=3,
                padding=padding, dilation=dilation, bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )

        self.density_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels // 4, 1, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(hidden_channels, num_count_classes)

    def forward(self, p2_feat):
        autocast = (
            torch.cuda.amp.autocast(enabled=False)
            if p2_feat.is_cuda else nullcontext()
        )
        with autocast:
            density_feat = self.stem(p2_feat.float())
            density_feat = torch.nan_to_num(
                density_feat, nan=0.0, posinf=20.0, neginf=-20.0,
            ).clamp(min=-20.0, max=20.0)
            density_map = self.density_head(density_feat)
            density_map = torch.nan_to_num(
                density_map, nan=0.0, posinf=20.0, neginf=-20.0,
            ).clamp(min=-20.0, max=20.0)
            pooled = self.pool(density_feat).flatten(1)
            count_logits = self.cls_head(pooled)
            count_logits = torch.nan_to_num(
                count_logits, nan=0.0, posinf=30.0, neginf=-30.0,
            ).clamp(min=-30.0, max=30.0)
        return count_logits, density_map, density_feat


class DensityGuidedCGFE(nn.Module):
    def __init__(self, num_levels=4, init_alpha=0.5):
        super().__init__()
        self.num_levels = num_levels
        self.alpha = nn.Parameter(torch.ones(num_levels) * init_alpha)

    def forward(self, feats, density_map):
        if density_map is None:
            return feats

        enhanced = []
        for i, feat in enumerate(feats):
            safe_density = torch.nan_to_num(
                density_map, nan=0.0, posinf=20.0, neginf=-20.0,
            ).clamp(min=-20.0, max=20.0)
            attn = F.interpolate(
                safe_density, size=feat.shape[-2:], mode="bilinear", align_corners=False,
            )
            attn = torch.sigmoid(attn).to(dtype=feat.dtype)
            scale = torch.sigmoid(self.alpha[i])
            enhanced.append(feat + scale * attn * feat)
        return enhanced


def build_count_targets(targets, count_bins, device):
    labels = []
    for target in targets:
        count = len(target["labels"])
        level = 0
        for boundary in count_bins:
            if count >= boundary:
                level += 1
            else:
                break
        labels.append(level)
    return torch.tensor(labels, dtype=torch.long, device=device)
