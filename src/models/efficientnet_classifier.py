"""
EfficientNet-B1 Bird Call Classifier — ML Extension v2
Key additions:
  - Attention pooling head (replaces global avg pool)
  - Progressive backbone unfreezing helpers
  - Grad-CAM support
  - ONNX export helper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attention Pooling
# ---------------------------------------------------------------------------

class AttentionPool2d(nn.Module):
    """
    Soft-attention pooling over spatial feature maps.
    Learns which regions of the spectrogram are most discriminative.
    Replaces global average pooling.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        attn = self.attention(x)           # [B, 1, H, W]
        attn = attn.view(attn.size(0), -1) # [B, H*W]
        attn = F.softmax(attn, dim=-1)
        attn = attn.view(attn.size(0), 1, x.size(2), x.size(3))
        pooled = (x * attn).sum(dim=(2, 3))  # [B, C]
        return pooled


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class BirdCallClassifier(nn.Module):
    """
    EfficientNet-B1 backbone + attention-pooling head for bird species classification.
    """

    def __init__(self, num_classes: int = 9, pretrained: bool = True,
                 dropout_rate: float = 0.4, attention_pooling: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.attention_pooling = attention_pooling

        # --- backbone ---
        self.backbone, self.feature_dim = self._build_backbone(pretrained)

        # --- pooling ---
        if attention_pooling:
            self.pool = AttentionPool2d(self.feature_dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)

        # --- classification head ---
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes),
        )

        self._init_head()
        logger.info(f"BirdCallClassifier: {num_classes} classes, "
                    f"attn_pool={attention_pooling}, feat_dim={self.feature_dim}")

    # ------------------------------------------------------------------
    def _build_backbone(self, pretrained: bool) -> Tuple[nn.Module, int]:
        """Try timm first, fall back to torchvision."""
        if TIMM_AVAILABLE:
            model = timm.create_model(
                "efficientnet_b1", pretrained=pretrained, num_classes=0, global_pool=""
            )
            feature_dim = model.num_features
            return model, feature_dim

        if TORCHVISION_AVAILABLE:
            weights = EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
            model = efficientnet_b1(weights=weights)
            feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
            model.avgpool = nn.Identity()
            return model, feature_dim

        raise ImportError("Install timm or torchvision: pip install timm")

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Progressive unfreezing helpers
    # ------------------------------------------------------------------

    def freeze_backbone(self):
        """Freeze all backbone parameters (Phase 1)."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        logger.info("Backbone frozen — training head only")

    def unfreeze_last_n_blocks(self, n: int = 3):
        """
        Unfreeze the last n blocks of EfficientNet backbone (Phase 2).
        Works with both timm and torchvision layouts.
        """
        # collect named children of the backbone
        children = list(self.backbone.named_children())
        # unfreeze the last n
        for name, module in children[-n:]:
            for p in module.parameters():
                p.requires_grad = True
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        logger.info(f"Unfroze last {n} backbone blocks — {trainable:,} backbone params trainable")

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        logger.info("Full backbone unfrozen")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [B,1,H,W] mel-spectrogram to [B,3,H,W] for pretrained backbone."""
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.size(2) < 224 or x.size(3) < 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor,
                return_features: bool = False) -> torch.Tensor:
        x = self._to_rgb(x)
        features = self.backbone(x)  # [B, C, H, W] (timm with global_pool="")

        # torchvision returns [B, C, 1, 1] after its avgpool=Identity
        if features.dim() == 2:
            # timm sometimes returns [B, C] — add spatial dims
            features = features.unsqueeze(-1).unsqueeze(-1)

        # pool
        if self.attention_pooling:
            pooled = self.pool(features)       # [B, C]
        else:
            pooled = self.pool(features).flatten(1)  # [B, C]

        logits = self.head(pooled)

        if return_features:
            return logits, pooled
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=1)

    # ------------------------------------------------------------------
    # Grad-CAM
    # ------------------------------------------------------------------

    def get_gradcam(self, x: torch.Tensor,
                    target_class: Optional[int] = None) -> torch.Tensor:
        """
        Compute Grad-CAM heatmap for input spectrogram batch.
        Returns heatmap tensor [B, H, W] normalised to [0,1].
        """
        self.eval()
        x = self._to_rgb(x)

        feat_maps = []
        grads = []

        # hook on the last backbone block
        children = list(self.backbone.children())
        target_layer = children[-1]

        def fwd_hook(m, inp, out):
            feat_maps.append(out)

        def bwd_hook(m, gin, gout):
            grads.append(gout[0])

        fh = target_layer.register_forward_hook(fwd_hook)
        bh = target_layer.register_full_backward_hook(bwd_hook)

        try:
            out = self.forward(x)
            if target_class is None:
                target_class = out.argmax(dim=1)

            self.zero_grad()
            score = out[:, target_class].sum() if isinstance(target_class, int) \
                else out.gather(1, target_class.view(-1, 1)).sum()
            score.backward()

            fmaps = feat_maps[0]   # [B, C, H, W]
            gs = grads[0]          # [B, C, H, W]
            weights = gs.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
            cam = F.relu((weights * fmaps).sum(dim=1))   # [B, H, W]

            # normalise per sample
            B = cam.size(0)
            cam_flat = cam.view(B, -1)
            cam_min = cam_flat.min(dim=1)[0].view(B, 1, 1)
            cam_max = cam_flat.max(dim=1)[0].view(B, 1, 1)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        finally:
            fh.remove()
            bh.remove()

        return cam.detach()

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def export_onnx(self, save_path: str, input_shape: Tuple = (1, 1, 128, 313)):
        """Export model to ONNX format."""
        self.eval()
        dummy = torch.zeros(*input_shape)
        torch.onnx.export(
            self,
            dummy,
            save_path,
            export_params=True,
            opset_version=17,
            input_names=["spectrogram"],
            output_names=["logits"],
            dynamic_axes={"spectrogram": {0: "batch"}, "logits": {0: "batch"}},
        )
        logger.info(f"ONNX model saved to {save_path}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_model(config: Dict) -> BirdCallClassifier:
    mc = config["model"]
    return BirdCallClassifier(
        num_classes=mc["num_classes"],
        pretrained=mc.get("pretrained", True),
        dropout_rate=mc.get("dropout_rate", 0.4),
        attention_pooling=mc.get("attention_pooling", True),
    )


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
