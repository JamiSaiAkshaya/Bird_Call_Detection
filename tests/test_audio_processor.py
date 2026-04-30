"""Tests for AudioProcessor."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pytest
import yaml

@pytest.fixture
def config():
    with open(Path(__file__).parent.parent / 'config' / 'config.yaml') as f:
        return yaml.safe_load(f)

@pytest.fixture
def proc(config):
    from src.data.audio_processor import AudioProcessor
    return AudioProcessor(config)

def test_normalize_peak(proc):
    audio = np.array([0.5, -1.0, 0.3], dtype=np.float32)
    out = proc.normalize_audio(audio)
    assert np.isclose(np.max(np.abs(out)), 1.0)

def test_pad(proc):
    short = np.zeros(100, dtype=np.float32)
    out = proc.pad_or_truncate(short)
    assert len(out) == proc._target_len

def test_truncate(proc):
    long = np.zeros(proc._target_len * 3, dtype=np.float32)
    out = proc.pad_or_truncate(long)
    assert len(out) == proc._target_len

def test_mel_shape(proc):
    audio = np.random.randn(proc._target_len).astype(np.float32)
    mel = proc.compute_mel_spectrogram(audio)
    assert mel.shape[0] == proc.n_mels
    assert mel.min() >= 0.0 and mel.max() <= 1.0 + 1e-5

def test_preprocess_returns_tensor(proc, tmp_path):
    import soundfile as sf
    audio = np.random.randn(proc._target_len).astype(np.float32)
    p = tmp_path / 'test.wav'
    sf.write(str(p), audio, proc.sample_rate)
    t = proc.preprocess_for_cnn(str(p))
    assert isinstance(t, torch.Tensor)
    assert t.shape[0] == 1
    assert t.shape[1] == proc.n_mels
