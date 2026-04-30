"""
Advanced Augmentation Pipeline for Bird Call Detection
Implements: SpecAugment, Mixup, CutMix, noise overlay, pitch shift, time stretch
All spectrogram augmentations operate on torch.Tensor [C, H, W].
"""

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import random
from typing import Dict, Tuple


def _rand(lo: float, hi: float) -> float:
    return random.uniform(lo, hi)


def _coin(p: float) -> bool:
    return random.random() < p


# ---------------------------------------------------------------------------
# Spectrogram-level augmentations
# ---------------------------------------------------------------------------

class SpecAugment:
    """Time and frequency masking (Park et al. 2019)."""

    def __init__(self, cfg: Dict):
        self.tm_cfg = cfg.get("time_masking", {})
        self.fm_cfg = cfg.get("frequency_masking", {})

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        _, n_mels, n_frames = spec.shape

        if self.tm_cfg.get("enabled", False) and _coin(self.tm_cfg.get("probability", 0.5)):
            max_t = max(1, int(n_frames * self.tm_cfg.get("max_mask_pct", 0.15)))
            for _ in range(self.tm_cfg.get("num_masks", 2)):
                t = random.randint(1, max_t)
                t0 = random.randint(0, max(0, n_frames - t))
                spec[:, :, t0: t0 + t] = 0.0

        if self.fm_cfg.get("enabled", False) and _coin(self.fm_cfg.get("probability", 0.5)):
            max_f = max(1, int(n_mels * self.fm_cfg.get("max_mask_pct", 0.15)))
            for _ in range(self.fm_cfg.get("num_masks", 2)):
                f = random.randint(1, max_f)
                f0 = random.randint(0, max(0, n_mels - f))
                spec[:, f0: f0 + f, :] = 0.0

        return spec


class NoiseOverlay:
    """Add Gaussian noise to simulate field recording conditions."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.cfg.get("enabled", False) or not _coin(self.cfg.get("probability", 0.4)):
            return spec
        lo, hi = self.cfg.get("noise_factor_range", [0.002, 0.015])
        return (spec + torch.randn_like(spec) * _rand(lo, hi)).clamp(0.0, 1.0)


class GainAugment:
    """Random amplitude scaling."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.cfg.get("enabled", False) or not _coin(self.cfg.get("probability", 0.4)):
            return spec
        lo, hi = self.cfg.get("gain_range", [0.7, 1.3])
        return (spec * _rand(lo, hi)).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Batch-level Mixup and CutMix
# ---------------------------------------------------------------------------

def mixup_batch(specs: torch.Tensor, labels: torch.Tensor,
                num_classes: int, alpha: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mixup on a batch. Returns mixed specs and soft one-hot labels."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(specs.size(0), device=specs.device)
    mixed = lam * specs + (1.0 - lam) * specs[idx]
    one_hot = F.one_hot(labels, num_classes).float()
    mixed_labels = lam * one_hot + (1.0 - lam) * one_hot[idx]
    return mixed, mixed_labels


def cutmix_batch(specs: torch.Tensor, labels: torch.Tensor,
                 num_classes: int, alpha: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """CutMix on a batch. Pastes a rectangular region from another sample."""
    lam = float(np.random.beta(alpha, alpha))
    B, C, H, W = specs.shape
    idx = torch.randperm(B, device=specs.device)
    cut_h = int(H * np.sqrt(1.0 - lam))
    cut_w = int(W * np.sqrt(1.0 - lam))
    cy, cx = random.randint(0, H), random.randint(0, W)
    y1, y2 = max(0, cy - cut_h // 2), min(H, cy + cut_h // 2)
    x1, x2 = max(0, cx - cut_w // 2), min(W, cx + cut_w // 2)
    mixed = specs.clone()
    mixed[:, :, y1:y2, x1:x2] = specs[idx, :, y1:y2, x1:x2]
    actual_lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    one_hot = F.one_hot(labels, num_classes).float()
    mixed_labels = actual_lam * one_hot + (1.0 - actual_lam) * one_hot[idx]
    return mixed, mixed_labels


# ---------------------------------------------------------------------------
# Audio-level augmentations (before spectrogram)
# ---------------------------------------------------------------------------

class AudioAugment:
    """Pitch shift and time stretch on raw waveforms (numpy)."""

    def __init__(self, cfg: Dict, sample_rate: int = 32000):
        self.ps_cfg = cfg.get("pitch_shift", {})
        self.ts_cfg = cfg.get("time_stretch", {})
        self.sr = sample_rate

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if self.ps_cfg.get("enabled", False) and _coin(self.ps_cfg.get("probability", 0.3)):
            lo, hi = self.ps_cfg.get("steps_range", [-1.5, 1.5])
            try:
                audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=_rand(lo, hi))
            except Exception:
                pass

        if self.ts_cfg.get("enabled", False) and _coin(self.ts_cfg.get("probability", 0.3)):
            lo, hi = self.ts_cfg.get("rate_range", [0.85, 1.15])
            try:
                audio = librosa.effects.time_stretch(audio, rate=_rand(lo, hi))
            except Exception:
                pass
        return audio


# ---------------------------------------------------------------------------
# Composed pipeline
# ---------------------------------------------------------------------------

class SpectrogramAugmentPipeline:
    """Chains all spectrogram-level augmentations. Call on each tensor in Dataset."""

    def __init__(self, cfg: Dict):
        aug_cfg = cfg.get("augmentation", {})
        self.spec_aug = SpecAugment(aug_cfg)
        self.noise = NoiseOverlay(aug_cfg.get("noise_overlay", {}))
        self.gain = GainAugment(aug_cfg.get("gain", {}))
        self.enabled = aug_cfg.get("enabled", True)

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return spec
        spec = self.spec_aug(spec)
        spec = self.noise(spec)
        spec = self.gain(spec)
        return spec
