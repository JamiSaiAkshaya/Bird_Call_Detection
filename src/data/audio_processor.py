"""
Professional Audio Preprocessing Pipeline for Bird Call Detection
Optimized for real-time processing and CNN input preparation
"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Dict, Optional, Union, List
import torch
import logging
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class AudioProcessor:
    """
    High-performance audio preprocessing pipeline for bird call classification
    """
    
    def __init__(self, config: Dict):
        """Initialize audio processor with configuration"""
        self.sample_rate = config['audio']['sample_rate']
        self.duration = config['audio']['duration']
        self.hop_length = config['audio']['hop_length']
        self.n_fft = config['audio']['n_fft']
        self.n_mels = config['audio']['n_mels']
        self.fmax = config['audio']['fmax']
        self.fmin = config['audio']['fmin']
        self.window = config['audio']['window']
        self.center = config['audio']['center']
        self.norm = config['audio']['norm']
        
        # Preprocessing settings
        self.normalization = config['preprocessing']['normalization']
        self.remove_silence = config['preprocessing']['remove_silence']
        self.silence_threshold = config['preprocessing']['silence_threshold']
        self.bandpass_filter = config['preprocessing']['bandpass_filter']
        self.low_cutoff = config['preprocessing']['low_cutoff']
        self.high_cutoff = config['preprocessing']['high_cutoff']
        self.apply_trim = config['preprocessing']['apply_trim']
        self.pad_mode = config['preprocessing']['pad_mode']
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized AudioProcessor - SR: {self.sample_rate}, "
                        f"Duration: {self.duration}s, N_mels: {self.n_mels}")
    
    def load_audio(self, file_path: Union[str, Path], 
                   target_sr: Optional[int] = None,
                   offset: float = 0.0,
                   duration: Optional[float] = None) -> np.ndarray:
        """
        Load and resample audio file with robust error handling
        
        Args:
            file_path: Path to audio file
            target_sr: Target sample rate (uses config if None)
            offset: Start time in seconds
            duration: Duration in seconds (uses config if None)
            
        Returns:
            Audio array
        """
        target_sr = target_sr or self.sample_rate
        duration = duration or self.duration
        
        try:
            # Try with librosa first (handles more formats)
            audio, sr = librosa.load(
                file_path,
                sr=target_sr,
                offset=offset,
                duration=duration,
                mono=True,
                res_type='kaiser_fast'
            )
            
            if len(audio) == 0:
                raise ValueError(f"Empty audio file: {file_path}")
                
            return audio
            
        except Exception as e:
            self.logger.warning(f"Librosa failed for {file_path}, trying soundfile: {e}")
            
            try:
                # Fallback to soundfile
                audio, sr = sf.read(file_path, start=int(offset*target_sr), 
                                  stop=int((offset+duration)*target_sr) if duration else None)
                
                # Convert to mono if needed
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                
                # Resample if needed
                if sr != target_sr:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                    
                return audio.astype(np.float32)
                
            except Exception as e2:
                self.logger.error(f"Failed to load audio {file_path}: {e2}")
                # Return silence as fallback
                return np.zeros(int(duration * target_sr), dtype=np.float32)
    
    def normalize_audio(self, audio: np.ndarray, 
                       method: str = None) -> np.ndarray:
        """
        Normalize audio amplitude
        
        Args:
            audio: Input audio array
            method: Normalization method ('peak', 'rms', 'lufs')
            
        Returns:
            Normalized audio array
        """
        method = method or self.normalization
        
        if len(audio) == 0:
            return audio
            
        try:
            if method == 'peak':
                # Peak normalization
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio / peak
                    
            elif method == 'rms':
                # RMS normalization
                rms = np.sqrt(np.mean(audio**2))
                if rms > 0:
                    target_rms = 0.1  # Target RMS level
                    audio = audio * (target_rms / rms)
                    
            elif method == 'lufs':
                # Simplified LUFS-style normalization
                audio = librosa.util.normalize(audio)
                
            # Clip to prevent overflow
            audio = np.clip(audio, -1.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Normalization failed: {e}")
            
        return audio
    
    def remove_silence_segments(self, audio: np.ndarray, 
                               threshold_db: int = None) -> np.ndarray:
        """
        Remove silent segments from audio
        
        Args:
            audio: Input audio array
            threshold_db: Silence threshold in dB below peak
            
        Returns:
            Audio with silence removed
        """
        if not self.remove_silence:
            return audio
            
        threshold_db = threshold_db or self.silence_threshold
        
        try:
            # Trim silence from beginning and end
            audio_trimmed, _ = librosa.effects.trim(
                audio,
                top_db=threshold_db,
                frame_length=2048,
                hop_length=512
            )
            
            # If trimming results in very short audio, return original
            if len(audio_trimmed) < 0.1 * self.sample_rate:  # Less than 0.1 seconds
                self.logger.warning("Trimming resulted in very short audio, using original")
                return audio
                
            return audio_trimmed
            
        except Exception as e:
            self.logger.warning(f"Failed to trim silence: {e}")
            return audio
    
    def apply_bandpass_filter(self, audio: np.ndarray, 
                            low_freq: float = None, 
                            high_freq: float = None) -> np.ndarray:
        """
        Apply bandpass filter to focus on bird vocalization frequencies
        
        Args:
            audio: Input audio array
            low_freq: Low cutoff frequency (Hz)
            high_freq: High cutoff frequency (Hz)
            
        Returns:
            Filtered audio array
        """
        if not self.bandpass_filter:
            return audio
            
        low_freq = low_freq or self.low_cutoff
        high_freq = high_freq or self.high_cutoff
        
        try:
            # Design Butterworth bandpass filter
            nyquist = self.sample_rate / 2
            low = max(low_freq / nyquist, 0.01)  # Avoid zero frequency
            high = min(high_freq / nyquist, 0.99)  # Avoid nyquist frequency
            
            if low >= high:
                self.logger.warning(f"Invalid filter frequencies: {low_freq}-{high_freq} Hz")
                return audio
            
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_audio = signal.filtfilt(b, a, audio)
            
            return filtered_audio.astype(np.float32)
            
        except Exception as e:
            self.logger.warning(f"Bandpass filter failed: {e}")
            return audio
    
    def compute_mel_spectrogram(self, audio: np.ndarray, 
                               power: float = 2.0,
                               apply_log: bool = True) -> np.ndarray:
        """
        Compute mel-scale spectrogram optimized for CNN input
        
        Args:
            audio: Input audio array
            power: Exponent for magnitude spectrogram
            apply_log: Whether to apply log scaling
            
        Returns:
            Mel-spectrogram array [n_mels, time_frames]
        """
        try:
            # Compute mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
                power=power,
                window=self.window,
                center=self.center,
                norm=self.norm
            )
            
            if apply_log:
                # Convert to log scale (dB)
                mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Normalize to [0, 1] range for CNN stability
                mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
            
            return mel_spec.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Failed to compute mel-spectrogram: {e}")
            # Return zero spectrogram as fallback
            time_frames = 1 + int(len(audio) // self.hop_length)
            return np.zeros((self.n_mels, time_frames), dtype=np.float32)
    
    def compute_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """
        Compute MFCC features
        
        Args:
            audio: Input audio array
            n_mfcc: Number of MFCC coefficients
            
        Returns:
            MFCC feature array [n_features, time_frames]
        """
        try:
            # Compute MFCCs
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                fmin=self.fmin,
                fmax=self.fmax
            )
            
            # Include delta and delta-delta features
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # Combine features
            features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
            
            return features.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Failed to compute MFCC: {e}")
            time_frames = 1 + int(len(audio) // self.hop_length)
            return np.zeros((n_mfcc * 3, time_frames), dtype=np.float32)
    
    def pad_or_truncate(self, audio: np.ndarray, 
                       target_length: Optional[int] = None) -> np.ndarray:
        """
        Pad or truncate audio to fixed length
        
        Args:
            audio: Input audio array
            target_length: Target length in samples
            
        Returns:
            Audio array with fixed length
        """
        if target_length is None:
            target_length = int(self.duration * self.sample_rate)
        
        if len(audio) > target_length:
            # Truncate from center to preserve important parts
            start = (len(audio) - target_length) // 2
            audio = audio[start:start + target_length]
            
        elif len(audio) < target_length:
            # Pad with zeros or repeat audio
            pad_length = target_length - len(audio)
            
            if self.pad_mode == 'constant':
                audio = np.pad(audio, (0, pad_length), mode='constant')
            elif self.pad_mode == 'reflect':
                audio = np.pad(audio, (0, pad_length), mode='reflect')
            elif self.pad_mode == 'repeat' and len(audio) > 0:
                # Repeat audio to fill length
                repeats = (pad_length // len(audio)) + 1
                audio_repeated = np.tile(audio, repeats)
                audio = audio_repeated[:target_length]
            else:
                audio = np.pad(audio, (0, pad_length), mode='constant')
                
        return audio
    
    def preprocess_for_cnn(self, audio_path: Union[str, Path],
                          return_raw: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, np.ndarray]]:
        """
        Complete preprocessing pipeline for CNN input
        
        Args:
            audio_path: Path to audio file
            return_raw: Whether to also return raw audio
            
        Returns:
            Preprocessed tensor ready for CNN [1, H, W] or tuple with raw audio
        """
        try:
            # Load and preprocess audio
            audio = self.load_audio(audio_path)
            
            if len(audio) == 0:
                raise ValueError(f"Empty audio loaded from {audio_path}")
            
            # Store original for return if needed
            raw_audio = audio.copy() if return_raw else None
            
            # Apply preprocessing steps
            audio = self.normalize_audio(audio)
            audio = self.remove_silence_segments(audio)
            audio = self.apply_bandpass_filter(audio)
            audio = self.pad_or_truncate(audio)
            
            # Compute mel-spectrogram
            mel_spec = self.compute_mel_spectrogram(audio)
            
            # Convert to tensor and add channel dimension [1, H, W]
            mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)
            
            if return_raw:
                return mel_tensor, raw_audio
            return mel_tensor
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed for {audio_path}: {e}")
            # Return zero tensor as fallback
            zero_tensor = torch.zeros(1, self.n_mels, 
                                    int(self.duration * self.sample_rate // self.hop_length) + 1)
            if return_raw:
                return zero_tensor, np.zeros(int(self.duration * self.sample_rate))
            return zero_tensor
    
    def batch_preprocess(self, audio_paths: List[Union[str, Path]], 
                        batch_size: int = 32) -> torch.Tensor:
        """
        Batch preprocessing for efficiency
        
        Args:
            audio_paths: List of audio file paths
            batch_size: Processing batch size
            
        Returns:
            Batched tensor [B, 1, H, W]
        """
        processed_tensors = []
        
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]
            batch_tensors = []
            
            for path in batch_paths:
                try:
                    tensor = self.preprocess_for_cnn(path)
                    batch_tensors.append(tensor)
                except Exception as e:
                    self.logger.warning(f"Skipping {path}: {e}")
                    # Add zero tensor as placeholder
                    zero_tensor = torch.zeros(1, self.n_mels, 
                                            int(self.duration * self.sample_rate // self.hop_length) + 1)
                    batch_tensors.append(zero_tensor)
            
            if batch_tensors:
                batch_tensor = torch.stack(batch_tensors, dim=0)
                processed_tensors.append(batch_tensor)
        
        if processed_tensors:
            return torch.cat(processed_tensors, dim=0)
        else:
            self.logger.error("No valid audio files processed")
            raise ValueError("No valid audio files in batch")
    
    def extract_features(self, audio_path: Union[str, Path], 
                        feature_types: List[str] = ['mel']) -> Dict[str, np.ndarray]:
        """
        Extract multiple feature types for analysis
        
        Args:
            audio_path: Path to audio file
            feature_types: List of feature types to extract
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        try:
            # Load and preprocess audio
            audio = self.load_audio(audio_path)
            audio = self.normalize_audio(audio)
            audio = self.apply_bandpass_filter(audio)
            audio = self.pad_or_truncate(audio)
            
            # Extract requested features
            if 'mel' in feature_types:
                features['mel_spectrogram'] = self.compute_mel_spectrogram(audio)
                
            if 'mfcc' in feature_types:
                features['mfcc'] = self.compute_mfcc(audio)
                
            if 'spectral_centroid' in feature_types:
                features['spectral_centroid'] = librosa.feature.spectral_centroid(
                    y=audio, sr=self.sample_rate, hop_length=self.hop_length)[0]
                
            if 'zero_crossing_rate' in feature_types:
                features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(
                    audio, hop_length=self.hop_length)[0]
                
            if 'chroma' in feature_types:
                features['chroma'] = librosa.feature.chroma_stft(
                    y=audio, sr=self.sample_rate, hop_length=self.hop_length)
                
            if 'spectral_rolloff' in feature_types:
                features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
                    y=audio, sr=self.sample_rate, hop_length=self.hop_length)[0]
                
            if 'tonnetz' in feature_types:
                features['tonnetz'] = librosa.feature.tonnetz(
                    y=audio, sr=self.sample_rate)
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed for {audio_path}: {e}")
        
        return features
    
    def get_audio_stats(self, audio: np.ndarray) -> Dict:
        """Get statistical information about audio"""
        stats = {
            'duration': len(audio) / self.sample_rate,
            'sample_rate': self.sample_rate,
            'channels': 1,  # Always mono
            'amplitude_max': float(np.max(np.abs(audio))),
            'amplitude_mean': float(np.mean(np.abs(audio))),
            'amplitude_std': float(np.std(audio)),
            'rms': float(np.sqrt(np.mean(audio**2))),
            'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(audio)[0])),
        }
        
        return stats