"""
EEG-specific data augmentations for improving cross-subject generalization.

Augmentations operate on raw EEG tensors of shape (channels, timepoints).
Used during training only (not validation/test).
"""

import random

import numpy as np
import torch


class GaussianNoise:
    """Add Gaussian noise with configurable standard deviation."""

    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, x):
        noise = torch.randn_like(x) * self.std
        return x + noise


class TimeShift:
    """Random circular shift along time axis."""

    def __init__(self, max_shift_ratio=0.1):
        self.max_shift_ratio = max_shift_ratio

    def __call__(self, x):
        # x shape: (channels, timepoints) or dict of tensors
        if isinstance(x, dict):
            # Apply to all time-domain tensors
            result = {}
            shift = None
            for k, v in x.items():
                if v.dim() >= 2:
                    if shift is None:
                        max_shift = int(v.shape[-1] * self.max_shift_ratio)
                        if max_shift > 0:
                            shift = random.randint(-max_shift, max_shift)
                        else:
                            shift = 0
                    result[k] = torch.roll(v, shifts=shift, dims=-1)
                else:
                    result[k] = v
            return result
        else:
            max_shift = int(x.shape[-1] * self.max_shift_ratio)
            if max_shift > 0:
                shift = random.randint(-max_shift, max_shift)
                return torch.roll(x, shifts=shift, dims=-1)
            return x


class ChannelDropout:
    """Zero out random channels (electrodes)."""

    def __init__(self, max_channels=3):
        self.max_channels = max_channels

    def __call__(self, x):
        if isinstance(x, dict):
            # Determine channels to drop once, apply to all
            num_drop = random.randint(1, self.max_channels)
            result = {}
            drop_indices = None
            for k, v in x.items():
                if v.dim() >= 2:
                    num_channels = v.shape[-2] if v.dim() == 2 else v.shape[-2]
                    if drop_indices is None:
                        drop_indices = random.sample(
                            range(min(num_channels, 19)), min(num_drop, num_channels)
                        )
                    v_out = v.clone()
                    for idx in drop_indices:
                        if idx < v_out.shape[-2]:
                            v_out[..., idx, :] = 0
                    result[k] = v_out
                else:
                    result[k] = v
            return result
        else:
            num_channels = x.shape[0]
            num_drop = random.randint(1, min(self.max_channels, num_channels))
            drop_indices = random.sample(range(num_channels), num_drop)
            x_out = x.clone()
            x_out[drop_indices] = 0
            return x_out


class AmplitudeScale:
    """Random amplitude scaling per channel."""

    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_min = scale_range[0]
        self.scale_max = scale_range[1]

    def __call__(self, x):
        if isinstance(x, dict):
            result = {}
            scales = None
            for k, v in x.items():
                if v.dim() >= 2:
                    num_channels = v.shape[-2] if v.dim() >= 2 else 1
                    if scales is None:
                        scales = torch.empty(num_channels).uniform_(
                            self.scale_min, self.scale_max
                        )
                    # Broadcast scale across time dimension
                    if v.dim() == 2 and v.shape[0] == num_channels:
                        result[k] = v * scales.unsqueeze(1).to(v.device)
                    else:
                        result[k] = v
                else:
                    result[k] = v
            return result
        else:
            num_channels = x.shape[0]
            scales = torch.empty(num_channels).uniform_(
                self.scale_min, self.scale_max
            ).to(x.device)
            return x * scales.unsqueeze(1)


class EEGAugmentor:
    """Compose augmentations with per-transform probability.

    Args:
        augment_prob: Global probability of applying any augmentation to a sample.
        noise_std: Standard deviation for Gaussian noise.
    """

    def __init__(self, augment_prob=0.3, noise_std=0.05):
        self.augment_prob = augment_prob
        self.transforms = [
            GaussianNoise(std=noise_std),
            TimeShift(max_shift_ratio=0.1),
            ChannelDropout(max_channels=3),
            AmplitudeScale(scale_range=(0.8, 1.2)),
        ]

    def __call__(self, x):
        """Apply augmentations with probability augment_prob each."""
        if self.augment_prob <= 0:
            return x
        for transform in self.transforms:
            if random.random() < self.augment_prob:
                x = transform(x)
        return x
