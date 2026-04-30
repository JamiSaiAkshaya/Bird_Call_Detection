"""
Comprehensive Evaluation Suite — ML Extension v2
Includes: accuracy, per-class F1, ROC/AUC, calibration ECE, Grad-CAM,
          confusion matrix, and ONNX INT8 quantization + latency test.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc,
    classification_report,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """
    Runs full evaluation on a test DataLoader and produces all required
    ML-project metrics and visualisations.
    """

    def __init__(self, model, config: Dict, class_names: List[str],
                 device: Optional[torch.device] = None):
        self.model = model
        self.config = config
        self.class_names = class_names
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.fig_dir = Path(config["paths"].get("figures_dir", "./experiments/figures"))
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def evaluate(self, test_loader) -> Dict:
        """Full evaluation pass. Returns metrics dict."""
        self.model.eval()
        all_labels, all_preds, all_probs = [], [], []

        with torch.no_grad():
            for specs, labels in test_loader:
                specs = specs.to(self.device)
                out = self.model(specs)
                probs = F.softmax(out, dim=1).cpu().numpy()
                preds = out.argmax(1).cpu().numpy()
                all_probs.append(probs)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.numpy().tolist())

        all_probs = np.concatenate(all_probs, axis=0)
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        metrics = self._compute_metrics(y_true, y_pred, all_probs)
        self._save_confusion_matrix(y_true, y_pred)
        self._save_roc_curves(y_true, all_probs)
        self._save_calibration(y_true, all_probs)
        self._print_report(y_true, y_pred)

        return metrics

    # ------------------------------------------------------------------
    def _compute_metrics(self, y_true, y_pred, probs) -> Dict:
        nc = len(self.class_names)
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_per_class": {},
        }
        f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
        for i, name in enumerate(self.class_names):
            metrics["f1_per_class"][name] = float(f1_per[i]) if i < len(f1_per) else 0.0

        # ROC-AUC (one-vs-rest)
        try:
            from sklearn.metrics import roc_auc_score
            if nc == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
            else:
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(y_true, probs, multi_class="ovr", average="weighted"))
        except Exception as e:
            logger.warning(f"ROC-AUC skipped: {e}")

        # Calibration ECE
        metrics["ece"] = float(self._ece(y_true, probs))

        logger.info(f"Evaluation complete — acc={metrics['accuracy']:.4f} "
                    f"f1={metrics['f1_weighted']:.4f}")
        return metrics

    @staticmethod
    def _ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error."""
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        correct = (predictions == y_true).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() == 0:
                continue
            acc = correct[mask].mean()
            conf = confidences[mask].mean()
            ece += (mask.sum() / len(y_true)) * abs(acc - conf)
        return ece

    # ------------------------------------------------------------------
    def _save_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
        fig, ax = plt.subplots(figsize=(max(6, len(self.class_names)),
                                        max(5, len(self.class_names) - 1)))
        sns.heatmap(cm_norm, annot=True, fmt=".2f",
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (normalised)")
        plt.tight_layout()
        fig.savefig(self.fig_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)
        logger.info("Confusion matrix saved")

    def _save_roc_curves(self, y_true, probs):
        nc = len(self.class_names)
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, name in enumerate(self.class_names):
            binary = (y_true == i).astype(int)
            if binary.sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(binary, probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})", linewidth=1.5)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves (one-vs-rest)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(self.fig_dir / "roc_curves.png", dpi=150)
        plt.close(fig)
        logger.info("ROC curves saved")

    def _save_calibration(self, y_true, probs):
        """Reliability diagram."""
        confidences = probs.max(axis=1)
        correct = (probs.argmax(axis=1) == y_true).astype(float)
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_acc, bin_conf, bin_count = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() == 0:
                continue
            bin_acc.append(correct[mask].mean())
            bin_conf.append(confidences[mask].mean())
            bin_count.append(mask.sum())

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=0.8)
        ax.bar(bin_conf, bin_acc, width=0.08, alpha=0.6, label="Model")
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Fraction correct")
        ax.set_title(f"Calibration curve  (ECE={self._ece(y_true, probs):.4f})")
        ax.legend()
        plt.tight_layout()
        fig.savefig(self.fig_dir / "calibration.png", dpi=150)
        plt.close(fig)
        logger.info("Calibration curve saved")

    def _print_report(self, y_true, y_pred):
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(y_true, y_pred,
                                    target_names=self.class_names,
                                    zero_division=0))

    # ------------------------------------------------------------------
    # Grad-CAM visualisation
    # ------------------------------------------------------------------

    def visualise_gradcam(self, sample_loader, n_samples: int = 4):
        """Save Grad-CAM overlays for n_samples from the loader."""
        self.model.eval()
        specs_list, labels_list = [], []
        for specs, labels in sample_loader:
            specs_list.append(specs)
            labels_list.append(labels)
            if sum(s.size(0) for s in specs_list) >= n_samples:
                break
        specs = torch.cat(specs_list)[:n_samples]
        labels = torch.cat(labels_list)[:n_samples]

        try:
            cams = self.model.get_gradcam(specs.to(self.device))
        except Exception as e:
            logger.warning(f"Grad-CAM failed: {e}")
            return

        fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 6))
        for i in range(n_samples):
            spec_img = specs[i, 0].numpy()
            cam_img = F.interpolate(
                cams[i].unsqueeze(0).unsqueeze(0).cpu(),
                size=spec_img.shape, mode="bilinear", align_corners=False
            )[0, 0].numpy()
            true_cls = self.class_names[labels[i].item()] \
                if labels[i].item() < len(self.class_names) else str(labels[i].item())
            axes[0, i].imshow(spec_img, origin="lower", aspect="auto", cmap="magma")
            axes[0, i].set_title(f"True: {true_cls}", fontsize=8)
            axes[0, i].axis("off")
            axes[1, i].imshow(spec_img, origin="lower", aspect="auto", cmap="magma")
            axes[1, i].imshow(cam_img, origin="lower", aspect="auto",
                              cmap="hot", alpha=0.5)
            axes[1, i].set_title("Grad-CAM", fontsize=8)
            axes[1, i].axis("off")
        plt.suptitle("Grad-CAM: attended frequency-time regions")
        plt.tight_layout()
        fig.savefig(self.fig_dir / "gradcam.png", dpi=150)
        plt.close(fig)
        logger.info("Grad-CAM saved")


# ---------------------------------------------------------------------------
# ONNX export + INT8 quantization
# ---------------------------------------------------------------------------

def export_and_quantize(model: torch.nn.Module, config: Dict,
                        sample_input: Optional[torch.Tensor] = None) -> Dict:
    """
    1. Export to ONNX.
    2. Apply INT8 dynamic quantization via onnxruntime.
    3. Benchmark latency.
    Returns dict with paths and timing.
    """
    import onnx
    import onnxruntime as ort

    models_dir = Path(config["paths"]["models_dir"])
    onnx_path = str(models_dir / "bird_detector.onnx")
    quant_path = str(models_dir / "bird_detector_int8.onnx")

    model.eval()
    if sample_input is None:
        sample_input = torch.zeros(1, 1, 128, 313)

    # export
    model.export_onnx(onnx_path, input_shape=tuple(sample_input.shape))
    onnx.checker.check_model(onnx_path)
    logger.info("ONNX model validated")

    # INT8 quantization
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QInt8)
        logger.info(f"INT8 quantized model saved: {quant_path}")
    except Exception as e:
        logger.warning(f"INT8 quantization failed: {e}")
        quant_path = None

    # latency benchmark (float32)
    sess = ort.InferenceSession(onnx_path,
                                providers=["CPUExecutionProvider"])
    inp = sample_input.numpy()
    # warm-up
    for _ in range(5):
        sess.run(None, {"spectrogram": inp})
    # measure
    t0 = time.perf_counter()
    N = 50
    for _ in range(N):
        sess.run(None, {"spectrogram": inp})
    latency_ms = (time.perf_counter() - t0) / N * 1000

    result = {
        "onnx_path": onnx_path,
        "quant_path": quant_path,
        "latency_ms_fp32": round(latency_ms, 2),
    }

    if quant_path:
        sess_q = ort.InferenceSession(quant_path,
                                      providers=["CPUExecutionProvider"])
        for _ in range(5):
            sess_q.run(None, {"spectrogram": inp})
        t0 = time.perf_counter()
        for _ in range(N):
            sess_q.run(None, {"spectrogram": inp})
        result["latency_ms_int8"] = round((time.perf_counter() - t0) / N * 1000, 2)

    print(f"\nONNX FP32 latency: {result['latency_ms_fp32']:.2f} ms/sample")
    if "latency_ms_int8" in result:
        print(f"ONNX INT8 latency: {result['latency_ms_int8']:.2f} ms/sample")

    return result
