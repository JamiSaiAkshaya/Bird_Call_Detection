"""Tests for BirdCallClassifier."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
import yaml

@pytest.fixture
def config():
    with open(Path(__file__).parent.parent / 'config' / 'config.yaml') as f:
        return yaml.safe_load(f)

@pytest.fixture
def model(config):
    from src.models.efficientnet_classifier import create_model
    return create_model(config)

def test_forward_shape(model, config):
    nc = config['model']['num_classes']
    x = torch.zeros(2, 1, 128, 313)
    out = model(x)
    assert out.shape == (2, nc)

def test_freeze_unfreeze(model):
    model.freeze_backbone()
    frozen = sum(p.requires_grad for p in model.backbone.parameters())
    assert frozen == 0
    model.unfreeze_last_n_blocks(2)
    unfrozen = sum(p.requires_grad for p in model.backbone.parameters())
    assert unfrozen > 0

def test_predict_proba_sums_to_one(model):
    x = torch.zeros(3, 1, 128, 313)
    probs = model.predict_proba(x)
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-5)
