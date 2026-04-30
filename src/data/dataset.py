"""
Professional Dataset Classes for Bird Call Detection
PyTorch datasets with comprehensive data handling and augmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from tqdm.auto import tqdm
import warnings

class BirdCallDataset(Dataset):
    """
    Professional PyTorch Dataset for bird call classification
    Features robust error handling, data validation, and performance optimization
    """
    
    def __init__(self, 
                 dataframe: pd.DataFrame,
                 audio_processor,
                 transform: Optional[Callable] = None,
                 cache_spectrograms: bool = True,
                 max_retries: int = 3,
                 validate_files: bool = True):
        """
        Initialize the bird call dataset
        
        Args:
            dataframe: DataFrame with audio file paths and labels
            audio_processor: AudioProcessor instance for preprocessing
            transform: Optional transforms to apply to audio tensors
            cache_spectrograms: Whether to cache processed spectrograms
            max_retries: Maximum retries for failed audio loading
            validate_files: Whether to validate file existence during init
        """
        self.audio_processor = audio_processor
        self.transform = transform
        self.cache_spectrograms = cache_spectrograms
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
        
        # Cache for processed spectrograms
        self._spectrogram_cache = {} if cache_spectrograms else None
        
        # Validate and filter dataframe
        if validate_files:
            self.df = self._validate_files(dataframe)
        else:
            self.df = dataframe.copy()
            
        self.df = self.df.reset_index(drop=True)
        
        # Extract unique classes and create mapping
        self.classes = sorted(self.df['label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Calculate class weights for balanced sampling
        self.class_weights = self._calculate_class_weights()
        
        self.logger.info(f"Dataset initialized with {len(self.df)} valid samples across {len(self.classes)} classes")
        
    def _validate_files(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Validate that audio files exist and are accessible"""
        valid_indices = []
        invalid_count = 0
        
        for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Validating files"):
            file_path = Path(row['local_path'])
            
            # Check file existence and basic validity
            if (file_path.exists() and 
                file_path.is_file() and 
                file_path.stat().st_size > 0 and
                file_path.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']):
                valid_indices.append(idx)
            else:
                invalid_count += 1
                
        if invalid_count > 0:
            self.logger.warning(f"Found {invalid_count} invalid files, keeping {len(valid_indices)} valid files")
            
        return dataframe.loc[valid_indices].copy()
    
    def _calculate_class_weights(self) -> torch.FloatTensor:
        """Calculate class weights for balanced sampling"""
        class_counts = self.df['label'].value_counts().sort_index()
        total_samples = len(self.df)
        
        weights = []
        for class_label in self.classes:
            if class_label in class_counts:
                weight = total_samples / (len(self.classes) * class_counts[class_label])
                weights.append(weight)
            else:
                weights.append(1.0)
                
        return torch.FloatTensor(weights)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset with comprehensive error handling
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (audio_tensor, label)
        """
        if idx >= len(self.df):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.df)}")
            
        row = self.df.iloc[idx]
        audio_path = row['local_path']
        label = row['label']
        
        # Try to get from cache first
        if self.cache_spectrograms and idx in self._spectrogram_cache:
            audio_tensor = self._spectrogram_cache[idx]
        else:
            # Process audio with retries
            audio_tensor = self._load_audio_with_retries(audio_path, idx)
            
            # Cache if enabled
            if self.cache_spectrograms:
                self._spectrogram_cache[idx] = audio_tensor.clone()
        
        # Apply transforms if provided
        if self.transform:
            try:
                audio_tensor = self.transform(audio_tensor)
            except Exception as e:
                self.logger.warning(f"Transform failed for {audio_path}: {e}")
        
        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return audio_tensor, label_tensor
    
    def _load_audio_with_retries(self, audio_path: str, idx: int) -> torch.Tensor:
        """Load and preprocess audio with multiple retry attempts"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Load and preprocess audio
                audio_tensor = self.audio_processor.preprocess_for_cnn(audio_path)
                
                # Validate tensor
                if torch.isnan(audio_tensor).any() or torch.isinf(audio_tensor).any():
                    raise ValueError(f"Invalid tensor values (NaN/Inf) in {audio_path}")
                
                if audio_tensor.numel() == 0:
                    raise ValueError(f"Empty tensor from {audio_path}")
                    
                return audio_tensor
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    self.logger.warning(f"Attempt {attempt + 1} failed for {audio_path}: {e}")
                    continue
                else:
                    self.logger.error(f"All attempts failed for {audio_path}: {e}")
        
        # Return zero tensor as fallback
        self.logger.error(f"Returning zero tensor for index {idx} due to persistent failures")
        return torch.zeros(1, 
                          self.audio_processor.n_mels, 
                          int(self.audio_processor.duration * self.audio_processor.sample_rate // self.audio_processor.hop_length) + 1)
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset"""
        return self.df['label'].value_counts().to_dict()
    
    def get_sample_weights(self) -> torch.FloatTensor:
        """Get sample weights for balanced sampling"""
        weights = []
        for _, row in self.df.iterrows():
            class_weight = self.class_weights[row['label']]
            weights.append(class_weight)
        return torch.FloatTensor(weights)
    
    def create_weighted_sampler(self) -> WeightedRandomSampler:
        """Create a weighted random sampler for balanced training"""
        sample_weights = self.get_sample_weights()
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def get_metadata(self, idx: int) -> Dict:
        """Get metadata for a specific sample"""
        if idx >= len(self.df):
            raise IndexError(f"Index {idx} out of range")
            
        row = self.df.iloc[idx]
        return {
            'file_path': row['local_path'],
            'label': row['label'],
            'species': row.get('common_name', 'Unknown'),
            'scientific_name': row.get('target_species', 'Unknown'),
            'priority': row.get('priority', 'Unknown'),
            'recording_id': row.get('id', 'Unknown'),
            'quality': row.get('q', 'Unknown'),
            'country': row.get('cnt', 'Unknown'),
            'length': row.get('length', 'Unknown')
        }
    
    def clear_cache(self):
        """Clear the spectrogram cache to free memory"""
        if self._spectrogram_cache:
            self._spectrogram_cache.clear()
            self.logger.info("Spectrogram cache cleared")


class DataAugmentation:
    """
    Professional data augmentation for audio spectrograms
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.augmentation_config = config.get('augmentation', {})
        self.enabled = self.augmentation_config.get('enabled', True)
        
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to spectrogram"""
        if not self.enabled:
            return spectrogram
            
        # Time masking
        if (self.augmentation_config.get('time_masking', {}).get('enabled', False) and
            np.random.random() < self.augmentation_config['time_masking'].get('probability', 0.3)):
            spectrogram = self._time_mask(spectrogram)
        
        # Frequency masking
        if (self.augmentation_config.get('frequency_masking', {}).get('enabled', False) and
            np.random.random() < self.augmentation_config['frequency_masking'].get('probability', 0.3)):
            spectrogram = self._frequency_mask(spectrogram)
        
        # Add gaussian noise
        if (self.augmentation_config.get('noise_injection', {}).get('enabled', False) and
            np.random.random() < self.augmentation_config['noise_injection'].get('probability', 0.3)):
            spectrogram = self._add_noise(spectrogram)
            
        return spectrogram
    
    def _time_mask(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply time masking to spectrogram"""
        config = self.augmentation_config['time_masking']
        max_mask_size = config.get('max_mask_size', 10)
        num_masks = config.get('num_masks', 2)
        
        _, _, time_steps = spectrogram.shape
        masked_spec = spectrogram.clone()
        
        for _ in range(num_masks):
            mask_size = np.random.randint(1, min(max_mask_size, time_steps // 4))
            mask_start = np.random.randint(0, max(1, time_steps - mask_size))
            masked_spec[:, :, mask_start:mask_start + mask_size] = 0
            
        return masked_spec
    
    def _frequency_mask(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply frequency masking to spectrogram"""
        config = self.augmentation_config['frequency_masking']
        max_mask_size = config.get('max_mask_size', 8)
        num_masks = config.get('num_masks', 2)
        
        _, freq_bins, _ = spectrogram.shape
        masked_spec = spectrogram.clone()
        
        for _ in range(num_masks):
            mask_size = np.random.randint(1, min(max_mask_size, freq_bins // 4))
            mask_start = np.random.randint(0, max(1, freq_bins - mask_size))
            masked_spec[:, mask_start:mask_start + mask_size, :] = 0
            
        return masked_spec
    
    def _add_noise(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to spectrogram"""
        noise_factor = self.augmentation_config['noise_injection'].get('noise_factor', 0.005)
        noise = torch.randn_like(spectrogram) * noise_factor
        return spectrogram + noise


def create_data_loaders(train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       audio_processor,
                       config: Dict,
                       use_weighted_sampling: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create optimized data loaders for training and validation
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe  
        audio_processor: AudioProcessor instance
        config: Configuration dictionary
        use_weighted_sampling: Whether to use weighted sampling for training
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    # Create augmentation
    augmentation = DataAugmentation(config) if config.get('augmentation', {}).get('enabled', False) else None
    
    # Create datasets
    train_dataset = BirdCallDataset(
        dataframe=train_df,
        audio_processor=audio_processor,
        transform=augmentation,
        cache_spectrograms=True,
        validate_files=True
    )
    
    val_dataset = BirdCallDataset(
        dataframe=val_df,
        audio_processor=audio_processor,
        transform=None,  # No augmentation for validation
        cache_spectrograms=True,
        validate_files=True
    )
    
    # Create samplers
    train_sampler = train_dataset.create_weighted_sampler() if use_weighted_sampling else None
    
    # Data loader settings
    num_workers = min(config['hardware']['num_workers'], 4)  # Limit for stability
    pin_memory = config['hardware']['pin_memory'] and torch.cuda.is_available()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
        collate_fn=None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
        collate_fn=None
    )
    
    return train_loader, val_loader


def get_dataset_statistics(dataset: BirdCallDataset) -> Dict:
    """Get comprehensive statistics about the dataset"""
    
    stats = {
        'total_samples': len(dataset),
        'num_classes': len(dataset.classes),
        'classes': dataset.classes,
        'class_distribution': dataset.get_class_distribution(),
        'class_weights': dataset.class_weights.tolist()
    }
    
    # Sample some tensors to get shape info
    if len(dataset) > 0:
        sample_tensor, sample_label = dataset[0]
        stats['tensor_shape'] = list(sample_tensor.shape)
        stats['tensor_dtype'] = str(sample_tensor.dtype)
    
    return stats