"""Tests for augmentation pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pytest
import yaml

@pytest.fixture
def config():
    with open(Path(__file__).parent.parent / 'config' / 'config.yaml') as f:
        return yaml.safe_load(f)

def test_spec_augment_shape(config):
    from src.training.augmentation import SpecAugment
    aug = SpecAugment(config['augmentation'])
    t = torch.ones(1, 128, 313)
    out = aug(t)
    assert out.shape == t.shape

def test_mixup_shape(config):
    from src.training.augmentation import mixup_batch
    specs  = torch.rand(4, 1, 128, 313)
    labels = torch.tensor([0, 1, 2, 3])
    ms, ml = mixup_batch(specs, labels, num_classes=9, alpha=0.3)
    assert ms.shape == specs.shape
    assert ml.shape == (4, 9)
    assert torch.allclose(ml.sum(dim=1), torch.ones(4), atol=1e-5)

def test_cutmix_shape(config):
    from src.training.augmentation import cutmix_batch
    specs  = torch.rand(4, 1, 128, 313)
    labels = torch.tensor([0, 1, 2, 3])
    ms, ml = cutmix_batch(specs, labels, num_classes=9, alpha=0.5)
    assert ms.shape == specs.shape
    assert ml.shape == (4, 9)

def test_focal_loss_forward(config):
    from src.training.losses import FocalLoss
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.randn(8, 9)
    labels = torch.randint(0, 9, (8,))
    loss = loss_fn(logits, labels)
    assert loss.item() > 0

def test_focal_loss_soft_labels(config):
    from src.training.losses import FocalLoss
    import torch.nn.functional as F
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.randn(4, 9)
    soft   = F.one_hot(torch.tensor([0,1,2,3]), 9).float()
    loss = loss_fn(logits, soft)
    assert loss.item() > 0
