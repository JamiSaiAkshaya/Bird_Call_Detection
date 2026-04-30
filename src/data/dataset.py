"""
Dataset classes for Bird Call Detection — ML Extension v2
Multi-class labels, audio-level augmentation, stratified K-fold support.
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import logging
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class BirdCallDataset(Dataset):
    """
    PyTorch Dataset for multi-class bird call classification.
    Supports audio-level augmentation (pitch shift, time stretch) and
    spectrogram-level augmentation (SpecAugment, noise, gain).
    """

    def __init__(self,
                 dataframe: pd.DataFrame,
                 audio_processor,
                 spec_transform: Optional[Callable] = None,
                 audio_augment=None,
                 cache_spectrograms: bool = False,
                 validate_files: bool = True):
        """
        Args:
            dataframe:          Must have columns 'local_path' and 'label' (int).
            audio_processor:    AudioProcessor instance.
            spec_transform:     Callable applied to spectrogram tensor.
            audio_augment:      AudioAugment instance (pitch shift / time stretch).
            cache_spectrograms: Cache processed tensors in RAM (use only if RAM allows).
            validate_files:     Drop rows where audio file is missing.
        """
        self.audio_processor = audio_processor
        self.spec_transform = spec_transform
        self.audio_augment = audio_augment
        self.cache = {} if cache_spectrograms else None

        if validate_files:
            self.df = self._validate(dataframe)
        else:
            self.df = dataframe.copy()
        self.df = self.df.reset_index(drop=True)

        self.classes = sorted(self.df["label"].unique().tolist())
        self.num_classes = len(self.classes)
        self.class_weights = self._class_weights()
        logger.info(f"Dataset: {len(self.df)} samples, {self.num_classes} classes")

    # ------------------------------------------------------------------
    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        valid = []
        for idx, row in df.iterrows():
            p = Path(str(row.get("local_path", "")))
            if p.exists() and p.stat().st_size > 0:
                valid.append(idx)
        dropped = len(df) - len(valid)
        if dropped:
            logger.warning(f"Dropped {dropped} rows with missing audio files")
        return df.loc[valid].copy()

    def _class_weights(self) -> torch.FloatTensor:
        counts = self.df["label"].value_counts()
        total = len(self.df)
        weights = []
        for c in self.classes:
            w = total / (self.num_classes * counts.get(c, 1))
            weights.append(w)
        return torch.FloatTensor(weights)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        label = int(row["label"])

        if self.cache is not None and idx in self.cache:
            spec = self.cache[idx].clone()
        else:
            spec = self._load(row["local_path"])
            if self.cache is not None:
                self.cache[idx] = spec.clone()

        if self.spec_transform is not None:
            try:
                spec = self.spec_transform(spec)
            except Exception as e:
                logger.debug(f"spec_transform failed idx={idx}: {e}")

        return spec, label

    def _load(self, path: str) -> torch.Tensor:
        try:
            if self.audio_augment is not None:
                # Load raw audio, augment, then compute spectrogram
                audio = self.audio_processor.load_audio(path)
                audio = self.audio_augment(audio)
                audio = self.audio_processor.normalize_audio(audio)
                audio = self.audio_processor.apply_bandpass_filter(audio)
                audio = self.audio_processor.pad_or_truncate(audio)
                mel = self.audio_processor.compute_mel_spectrogram(audio)
                tensor = torch.FloatTensor(mel).unsqueeze(0)
            else:
                tensor = self.audio_processor.preprocess_for_cnn(path)

            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                tensor = torch.zeros_like(tensor)
            return tensor
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            n_mels = self.audio_processor.n_mels
            frames = int(self.audio_processor.duration *
                         self.audio_processor.sample_rate //
                         self.audio_processor.hop_length) + 1
            return torch.zeros(1, n_mels, frames)

    # ------------------------------------------------------------------
    def sample_weights(self) -> torch.FloatTensor:
        return torch.FloatTensor([
            self.class_weights[self.classes.index(int(r["label"]))]
            for _, r in self.df.iterrows()
        ])

    def weighted_sampler(self) -> WeightedRandomSampler:
        sw = self.sample_weights()
        return WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

    def class_distribution(self) -> Dict[int, int]:
        return self.df["label"].value_counts().to_dict()


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_loaders(train_df: pd.DataFrame, val_df: pd.DataFrame,
                 audio_processor, config: Dict,
                 spec_transform=None, audio_augment=None) -> Tuple[DataLoader, DataLoader]:
    """Create train and val DataLoaders."""
    nw = min(config["hardware"].get("num_workers", 2), 4)
    pin = config["hardware"].get("pin_memory", True) and torch.cuda.is_available()
    bs = config["training"]["batch_size"]

    train_ds = BirdCallDataset(train_df, audio_processor,
                               spec_transform=spec_transform,
                               audio_augment=audio_augment,
                               cache_spectrograms=False,
                               validate_files=True)

    val_ds = BirdCallDataset(val_df, audio_processor,
                             spec_transform=None,   # no aug on val
                             audio_augment=None,
                             cache_spectrograms=False,
                             validate_files=True)

    train_loader = DataLoader(
        train_ds, batch_size=bs,
        sampler=train_ds.weighted_sampler(),
        num_workers=nw, pin_memory=pin,
        persistent_workers=(nw > 0), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=pin,
        persistent_workers=(nw > 0),
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Multi-class label builder (called in the notebook / data pipeline)
# ---------------------------------------------------------------------------

def build_multiclass_dataframe(recordings_df: pd.DataFrame,
                               species_config: List[Dict],
                               background_config: List[Dict]) -> pd.DataFrame:
    """
    Assign integer class labels from species configs.
    Works on the DataFrame returned by XenoCantoAPI.
    """
    # Build a lookup: scientific_name -> class_id
    label_map = {}
    for s in species_config:
        label_map[s["scientific_name"]] = s["class_id"]
    for s in background_config:
        label_map[s["scientific_name"]] = s["class_id"]

    df = recordings_df.copy()
    df["label"] = df["target_species"].map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df
