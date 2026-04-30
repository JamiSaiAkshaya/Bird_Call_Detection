"""
Two-Phase Training Pipeline for Bird Call Detection — ML Extension v2

Phase 1: Backbone frozen, train head only (fast convergence, no catastrophic forgetting)
Phase 2: Unfreeze last N blocks, fine-tune end-to-end with low LR

Also includes:
  - Linear warm-up → cosine annealing LR schedule
  - Mixup / CutMix batch augmentation
  - Stratified K-Fold cross-validation
  - Class-weighted loss (Focal or CE)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import logging
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm

from src.training.augmentation import mixup_batch, cutmix_batch
from src.training.losses import build_loss
from src.data.dataset import make_loaders

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LR schedule helpers
# ---------------------------------------------------------------------------

def _warmup_cosine_lr(optimizer: optim.Optimizer,
                      step: int, warmup_steps: int, total_steps: int,
                      base_lr: float, min_lr: float):
    """Linear warm-up then cosine decay — called every step."""
    if step < warmup_steps:
        lr = base_lr * (step + 1) / max(warmup_steps, 1)
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ---------------------------------------------------------------------------
# Core trainer
# ---------------------------------------------------------------------------

class BirdDetectionTrainer:
    """Two-phase trainer for BirdCallClassifier."""

    def __init__(self, model: nn.Module, config: Dict,
                 audio_processor, device: Optional[str] = None):
        self.config = config
        self.audio_processor = audio_processor
        tc = config["training"]

        if device is None:
            hw = config["hardware"]["device"]
            if hw == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(hw)
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.ckpt_dir = Path(config["paths"]["models_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.use_amp = (config["hardware"].get("mixed_precision", True)
                        and torch.cuda.is_available())
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # augmentation flags
        self.mixup_cfg = config.get("augmentation", {}).get("mixup", {})
        self.cutmix_cfg = config.get("augmentation", {}).get("cutmix", {})
        self.num_classes = config["model"]["num_classes"]

        logger.info(f"Trainer | device={self.device} | AMP={self.use_amp}")

    # ------------------------------------------------------------------
    # Phase execution
    # ------------------------------------------------------------------

    def _run_phase(self, train_loader, val_loader,
                   num_epochs: int, base_lr: float,
                   phase: int, class_weights=None) -> Dict:
        """Run one training phase; return best metrics."""
        tc = self.config["training"]
        min_lr = tc.get("min_lr", 1e-7)
        warmup_epochs = tc.get("warmup_epochs", 5) if phase == 1 else 2
        patience = tc.get("early_stopping_patience", 12)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=base_lr,
            weight_decay=tc.get("weight_decay", 1e-4),
        )

        total_steps = num_epochs * len(train_loader)
        warmup_steps = warmup_epochs * len(train_loader)

        cw = class_weights.to(self.device) if class_weights is not None else None
        criterion = build_loss(self.config, cw)

        best_f1, best_state, patience_counter = 0.0, None, 0
        history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
        global_step = 0

        for epoch in range(1, num_epochs + 1):
            # --- train ---
            self.model.train()
            run_loss, n = 0.0, 0

            for specs, labels in tqdm(train_loader,
                                      desc=f"[P{phase}] Epoch {epoch}/{num_epochs}",
                                      leave=False):
                specs, labels = specs.to(self.device), labels.to(self.device)

                # batch augmentation
                soft_labels = None
                r = np.random.random()
                if self.mixup_cfg.get("enabled") and r < self.mixup_cfg.get("probability", 0.4):
                    specs, soft_labels = mixup_batch(
                        specs, labels, self.num_classes, self.mixup_cfg.get("alpha", 0.3))
                elif self.cutmix_cfg.get("enabled") and r < self.cutmix_cfg.get("probability", 0.3):
                    specs, soft_labels = cutmix_batch(
                        specs, labels, self.num_classes, self.cutmix_cfg.get("alpha", 0.5))

                optimizer.zero_grad()
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        out = self.model(specs)
                        loss = criterion(out, soft_labels if soft_labels is not None else labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    out = self.model(specs)
                    loss = criterion(out, soft_labels if soft_labels is not None else labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                _warmup_cosine_lr(optimizer, global_step, warmup_steps,
                                  total_steps, base_lr, min_lr)
                global_step += 1
                run_loss += loss.item() * specs.size(0)
                n += specs.size(0)

            train_loss = run_loss / max(n, 1)

            # --- validate ---
            val_loss, val_acc, val_f1 = self._validate(val_loader, criterion)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [P{phase}] Ep {epoch:3d} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | "
                  f"val_acc={val_acc:.3f} | "
                  f"val_f1={val_f1:.3f} | "
                  f"lr={lr_now:.2e}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # restore best weights
        if best_state:
            self.model.load_state_dict(best_state)

        return {"best_f1": best_f1, "history": history}

    # ------------------------------------------------------------------
    def _validate(self, loader, criterion) -> Tuple[float, float, float]:
        self.model.eval()
        run_loss, preds_all, labels_all = 0.0, [], []
        with torch.no_grad():
            for specs, labels in loader:
                specs, labels = specs.to(self.device), labels.to(self.device)
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        out = self.model(specs)
                else:
                    out = self.model(specs)
                loss = criterion(out, labels)
                run_loss += loss.item() * specs.size(0)
                preds_all.extend(out.argmax(1).cpu().numpy())
                labels_all.extend(labels.cpu().numpy())

        n = max(len(labels_all), 1)
        val_loss = run_loss / n
        val_acc = accuracy_score(labels_all, preds_all)
        val_f1 = f1_score(labels_all, preds_all, average="weighted", zero_division=0)
        return val_loss, val_acc, val_f1

    # ------------------------------------------------------------------
    # Two-phase training
    # ------------------------------------------------------------------

    def train_two_phase(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                        class_weights=None, fold: int = 0) -> Dict:
        """
        Phase 1: head only (backbone frozen).
        Phase 2: last N backbone blocks unfrozen, low LR.
        """
        tc = self.config["training"]
        spec_aug, audio_aug = self._build_augmenters()

        train_loader, val_loader = make_loaders(
            train_df, val_df, self.audio_processor, self.config,
            spec_transform=spec_aug, audio_augment=audio_aug,
        )

        # Phase 1
        print(f"\n{'='*60}")
        print(f"PHASE 1 — Training head only (fold={fold})")
        print(f"{'='*60}")
        self.model.freeze_backbone()
        p1_result = self._run_phase(
            train_loader, val_loader,
            num_epochs=tc.get("phase1_epochs", 15),
            base_lr=tc.get("phase1_lr", 1e-3),
            phase=1,
            class_weights=class_weights,
        )

        # Phase 2
        print(f"\n{'='*60}")
        print(f"PHASE 2 — Fine-tuning last {tc.get('phase2_unfreeze_blocks', 3)} blocks")
        print(f"{'='*60}")
        self.model.unfreeze_last_n_blocks(tc.get("phase2_unfreeze_blocks", 3))
        p2_result = self._run_phase(
            train_loader, val_loader,
            num_epochs=tc.get("phase2_epochs", 45),
            base_lr=tc.get("phase2_lr", 1e-4),
            phase=2,
            class_weights=class_weights,
        )

        # save checkpoint
        ckpt = self.ckpt_dir / f"best_model_fold{fold}.pth"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "phase1_best_f1": p1_result["best_f1"],
            "phase2_best_f1": p2_result["best_f1"],
        }, ckpt)
        logger.info(f"Checkpoint saved: {ckpt}")

        return {
            "fold": fold,
            "phase1": p1_result,
            "phase2": p2_result,
            "best_val_f1": p2_result["best_f1"],
        }

    # ------------------------------------------------------------------
    # K-Fold cross-validation
    # ------------------------------------------------------------------

    def kfold_train(self, full_df: pd.DataFrame,
                    class_weights=None) -> List[Dict]:
        """
        Stratified K-Fold cross-validation.
        Returns per-fold results; saves best overall model.
        """
        tc = self.config["training"]
        n_splits = tc.get("kfold_splits", 5)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=tc.get("random_seed", 42))

        fold_results = []
        best_f1, best_state = 0.0, None

        for fold, (train_idx, val_idx) in enumerate(
                skf.split(full_df, full_df["label"])):
            print(f"\n{'#'*60}")
            print(f"  K-Fold: fold {fold + 1} / {n_splits}")
            print(f"{'#'*60}")

            # reset model weights for each fold
            self._reset_model()

            train_df = full_df.iloc[train_idx].copy()
            val_df = full_df.iloc[val_idx].copy()

            result = self.train_two_phase(
                train_df, val_df, class_weights=class_weights, fold=fold + 1)
            fold_results.append(result)

            if result["best_val_f1"] > best_f1:
                best_f1 = result["best_val_f1"]
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}

        # save overall best
        if best_state:
            self.model.load_state_dict(best_state)
            ckpt = self.ckpt_dir / "best_model_overall.pth"
            torch.save({"model_state_dict": best_state,
                        "best_f1": best_f1,
                        "config": self.config}, ckpt)
            print(f"\nBest overall model saved (F1={best_f1:.4f}): {ckpt}")

        # summary
        f1s = [r["best_val_f1"] for r in fold_results]
        print(f"\nK-Fold Summary: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        return fold_results

    # ------------------------------------------------------------------
    def _reset_model(self):
        """Re-initialise model weights for each fold."""
        from src.models.efficientnet_classifier import create_model
        fresh = create_model(self.config)
        self.model.load_state_dict(fresh.state_dict())
        self.model.to(self.device)

    def _build_augmenters(self):
        """Build spec and audio augmenters from config."""
        from src.training.augmentation import SpectrogramAugmentPipeline, AudioAugment
        aug_cfg = self.config.get("augmentation", {})
        spec_aug = SpectrogramAugmentPipeline(self.config) if aug_cfg.get("enabled") else None
        audio_aug = AudioAugment(aug_cfg, self.audio_processor.sample_rate) \
            if aug_cfg.get("enabled") else None
        return spec_aug, audio_aug
