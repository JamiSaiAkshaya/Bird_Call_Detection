"""
Real-time Bird Call Detector — ML Extension v2
Supports PyTorch and ONNX (INT8) backends.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import time

logger = logging.getLogger(__name__)


class RealtimeDetector:
    """
    Inference wrapper that supports:
      - PyTorch model (.pth checkpoint)
      - ONNX / ONNX-INT8 model (.onnx)
    """

    def __init__(self, config: Dict, audio_processor,
                 model_path: Optional[str] = None,
                 use_onnx: bool = False):
        self.config = config
        self.audio_processor = audio_processor
        self.use_onnx = use_onnx
        self.threshold = config["inference"]["confidence_threshold"]
        self.smoothing = config["inference"].get("smoothing_window", 5)

        # build class names list
        self.class_names = [
            s["common_name"]
            for s in config["species"]["target_species"]
        ] + [
            s["common_name"]
            for s in config["species"].get("background_species", [])
        ]

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        if model_path is None:
            model_path = str(
                Path(config["paths"]["models_dir"]) / "best_model_overall.pth")

        if use_onnx:
            self._load_onnx(model_path)
        else:
            self._load_pytorch(model_path)

        self._history: List[np.ndarray] = []

    # ------------------------------------------------------------------
    def _load_pytorch(self, path: str):
        from src.models.efficientnet_classifier import create_model
        self.model = create_model(self.config)
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        logger.info(f"PyTorch model loaded from {path}")

    def _load_onnx(self, path: str):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(
            path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.model = None
        logger.info(f"ONNX model loaded from {path}")

    # ------------------------------------------------------------------
    def predict_file(self, audio_path: str) -> Dict:
        """Predict species from an audio file."""
        t0 = time.perf_counter()
        spec = self.audio_processor.preprocess_for_cnn(audio_path)  # [1, H, W]
        spec = spec.unsqueeze(0)  # [1, 1, H, W]

        probs = self._infer(spec)
        latency = (time.perf_counter() - t0) * 1000

        smoothed = self._smooth(probs)
        class_idx = int(np.argmax(smoothed))
        confidence = float(smoothed[class_idx])
        species = self.class_names[class_idx] if class_idx < len(self.class_names) \
            else f"class_{class_idx}"

        alert = confidence >= self.threshold
        priority = self._alert_level(confidence)

        return {
            "species": species,
            "class_id": class_idx,
            "confidence": confidence,
            "alert": alert,
            "priority": priority,
            "latency_ms": round(latency, 2),
            "all_probabilities": {
                self.class_names[i] if i < len(self.class_names) else f"class_{i}": float(p)
                for i, p in enumerate(smoothed)
            },
        }

    def _infer(self, spec: torch.Tensor) -> np.ndarray:
        if self.use_onnx:
            out = self.sess.run(None, {"spectrogram": spec.numpy()})[0]
            probs = torch.softmax(torch.from_numpy(out), dim=1).numpy()[0]
        else:
            with torch.no_grad():
                out = self.model(spec.to(self.device))
            probs = F.softmax(out, dim=1).cpu().numpy()[0]
        return probs

    def _smooth(self, probs: np.ndarray) -> np.ndarray:
        self._history.append(probs)
        if len(self._history) > self.smoothing:
            self._history.pop(0)
        return np.mean(self._history, axis=0)

    def _alert_level(self, conf: float) -> str:
        thresholds = self.config["inference"]["alert_thresholds"]
        if conf >= thresholds["critical"]:
            return "CRITICAL"
        if conf >= thresholds["high"]:
            return "HIGH"
        if conf >= thresholds["medium"]:
            return "MEDIUM"
        return "LOW"

    def reset_smoothing(self):
        self._history.clear()
