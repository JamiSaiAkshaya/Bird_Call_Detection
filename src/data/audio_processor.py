"""
Audio Preprocessing Pipeline — ML Extension v2
Clean, self-contained, compatible with AudioAugment in the training loop.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, Optional, Union, List
import torch
import logging
from scipy import signal
import warnings; warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Handles: load → normalise → silence-trim → bandpass → pad/truncate → mel-spectrogram.
    All public methods return numpy arrays except preprocess_for_cnn which returns a tensor.
    """

    def __init__(self, config: Dict):
        a = config['audio']
        self.sample_rate  = a['sample_rate']
        self.duration     = a['duration']
        self.hop_length   = a['hop_length']
        self.n_fft        = a['n_fft']
        self.n_mels       = a['n_mels']
        self.fmax         = a['fmax']
        self.fmin         = a.get('fmin', 150)
        self.window       = a.get('window', 'hann')
        self.center       = a.get('center', True)
        self.norm         = a.get('norm', 'slaney')

        p = config['preprocessing']
        self.norm_method      = p.get('normalization', 'peak')
        self.do_trim          = p.get('remove_silence', True)
        self.trim_db          = p.get('silence_threshold', 25)
        self.do_bandpass      = p.get('bandpass_filter', True)
        self.low_cut          = p.get('low_cutoff', 150)
        self.high_cut         = p.get('high_cutoff', 15000)
        self.pad_mode         = p.get('pad_mode', 'constant')

        self._target_len = int(self.duration * self.sample_rate)
        logger.info(f'AudioProcessor — sr={self.sample_rate} dur={self.duration}s '
                    f'n_mels={self.n_mels}')

    # ------------------------------------------------------------------
    def load_audio(self, path: Union[str, Path],
                   offset: float = 0.0,
                   duration: Optional[float] = None) -> np.ndarray:
        dur = duration or self.duration
        try:
            audio, _ = librosa.load(str(path), sr=self.sample_rate,
                                    offset=offset, duration=dur,
                                    mono=True, res_type='kaiser_fast')
            return audio.astype(np.float32)
        except Exception as e1:
            logger.debug(f'librosa failed ({e1}), trying soundfile')
            try:
                raw, sr = sf.read(str(path), always_2d=False)
                if raw.ndim > 1:
                    raw = raw.mean(axis=1)
                if sr != self.sample_rate:
                    raw = librosa.resample(raw, orig_sr=sr, target_sr=self.sample_rate)
                return raw.astype(np.float32)
            except Exception as e2:
                logger.warning(f'Audio load failed {path}: {e2}')
                return np.zeros(self._target_len, dtype=np.float32)

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return audio
        if self.norm_method == 'peak':
            peak = np.max(np.abs(audio))
            if peak > 1e-8:
                audio = audio / peak
        elif self.norm_method == 'rms':
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 1e-8:
                audio = audio * (0.1 / rms)
        return np.clip(audio, -1.0, 1.0)

    def remove_silence_segments(self, audio: np.ndarray) -> np.ndarray:
        if not self.do_trim or len(audio) == 0:
            return audio
        try:
            trimmed, _ = librosa.effects.trim(audio, top_db=self.trim_db)
            return trimmed if len(trimmed) > 0.1 * self.sample_rate else audio
        except Exception:
            return audio

    def apply_bandpass_filter(self, audio: np.ndarray) -> np.ndarray:
        if not self.do_bandpass or len(audio) == 0:
            return audio
        try:
            nyq = self.sample_rate / 2.0
            lo  = max(self.low_cut / nyq, 0.01)
            hi  = min(self.high_cut / nyq, 0.99)
            if lo >= hi:
                return audio
            b, a = signal.butter(4, [lo, hi], btype='band')
            return signal.filtfilt(b, a, audio).astype(np.float32)
        except Exception:
            return audio

    def pad_or_truncate(self, audio: np.ndarray,
                        target: Optional[int] = None) -> np.ndarray:
        tgt = target or self._target_len
        if len(audio) > tgt:
            start = (len(audio) - tgt) // 2
            return audio[start: start + tgt]
        if len(audio) < tgt:
            pad = tgt - len(audio)
            return np.pad(audio, (0, pad), mode=self.pad_mode)
        return audio

    def compute_mel_spectrogram(self, audio: np.ndarray,
                                apply_log: bool = True) -> np.ndarray:
        try:
            mel = librosa.feature.melspectrogram(
                y=audio, sr=self.sample_rate,
                n_fft=self.n_fft, hop_length=self.hop_length,
                n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax,
                window=self.window, center=self.center)
            if apply_log:
                mel = librosa.power_to_db(mel, ref=np.max)
                mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
            return mel.astype(np.float32)
        except Exception as e:
            logger.warning(f'Mel-spectrogram failed: {e}')
            frames = 1 + int(self._target_len // self.hop_length)
            return np.zeros((self.n_mels, frames), dtype=np.float32)

    # ------------------------------------------------------------------
    def preprocess_for_cnn(self, path: Union[str, Path]) -> torch.Tensor:
        """Full pipeline → tensor [1, n_mels, frames]."""
        audio = self.load_audio(path)
        audio = self.normalize_audio(audio)
        audio = self.remove_silence_segments(audio)
        audio = self.apply_bandpass_filter(audio)
        audio = self.pad_or_truncate(audio)
        mel   = self.compute_mel_spectrogram(audio)
        return torch.FloatTensor(mel).unsqueeze(0)

    def get_audio_stats(self, audio: np.ndarray) -> Dict:
        return {
            'duration'       : round(len(audio) / self.sample_rate, 3),
            'sample_rate'    : self.sample_rate,
            'amplitude_max'  : float(np.max(np.abs(audio))),
            'rms'            : float(np.sqrt(np.mean(audio ** 2))),
            'zero_cross_rate': float(np.mean(librosa.feature.zero_crossing_rate(audio))),
        }
