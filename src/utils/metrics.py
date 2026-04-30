"""
Comprehensive metrics and evaluation utilities for Bird Call Detection
Professional implementation with conservation-specific metrics
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, precision_recall_curve, roc_curve
)
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging


class MetricsCalculator:
    """
    Comprehensive metrics calculator for bird classification tasks
    """
    
    def __init__(self, class_names: List[str], conservation_weights: Optional[Dict[str, float]] = None):
        """
        Initialize metrics calculator
        
        Args:
            class_names: List of class names
            conservation_weights: Weights for conservation importance (higher = more important)
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.conservation_weights = conservation_weights or {}
        self.logger = logging.getLogger(__name__)
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate basic classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            y_prob: Prediction probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i] if i < len(precision_per_class) else 0.0
            metrics[f'recall_{class_name}'] = recall_per_class[i] if i < len(recall_per_class) else 0.0
            metrics[f'f1_{class_name}'] = f1_per_class[i] if i < len(f1_per_class) else 0.0
        
        # AUC metrics (if probabilities available)
        if y_prob is not None:
            try:
                # Multi-class AUC
                if self.num_classes > 2:
                    metrics['auc_macro'] = roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr')
                    metrics['auc_weighted'] = roc_auc_score(y_true, y_prob, average='weighted', multi_class='ovr')
                else:
                    # Binary AUC
                    metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
                
                # Average Precision
                metrics['avg_precision_macro'] = average_precision_score(
                    self._to_one_hot(y_true), y_prob, average='macro'
                )
            except Exception as e:
                self.logger.warning(f"Failed to calculate AUC metrics: {e}")
        
        return metrics
    
    def calculate_conservation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate conservation-specific metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            
        Returns:
            Dictionary of conservation metrics
        """
        metrics = {}
        
        # False positive and negative costs for conservation
        false_positive_cost = 1.0  # Cost of false alarm
        false_negative_cost = 10.0  # Cost of missing endangered species
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate conservation-weighted costs
        total_fp_cost = 0
        total_fn_cost = 0
        
        for i, class_name in enumerate(self.class_names):
            if i < cm.shape[0]:
                # False positives for this class
                fp = cm[:, i].sum() - cm[i, i]
                # False negatives for this class  
                fn = cm[i, :].sum() - cm[i, i]
                
                # Weight by conservation importance
                conservation_weight = self.conservation_weights.get(class_name, 1.0)
                
                total_fp_cost += fp * false_positive_cost * conservation_weight
                total_fn_cost += fn * false_negative_cost * conservation_weight
                
                # Per-class conservation metrics
                metrics[f'false_positives_{class_name}'] = fp
                metrics[f'false_negatives_{class_name}'] = fn
                metrics[f'conservation_cost_{class_name}'] = (fp * false_positive_cost + fn * false_negative_cost) * conservation_weight
        
        metrics['total_false_positive_cost'] = total_fp_cost
        metrics['total_false_negative_cost'] = total_fn_cost
        metrics['total_conservation_cost'] = total_fp_cost + total_fn_cost
        
        # Conservation effectiveness (lower is better)
        total_samples = len(y_true)
        metrics['conservation_effectiveness'] = (total_fp_cost + total_fn_cost) / total_samples if total_samples > 0 else 0
        
        # Endangered species detection rate
        endangered_classes = [i for i, name in enumerate(self.class_names) 
                            if self.conservation_weights.get(name, 0) > 1.0]
        
        if endangered_classes:
            endangered_mask = np.isin(y_true, endangered_classes)
            if endangered_mask.sum() > 0:
                endangered_accuracy = accuracy_score(y_true[endangered_mask], y_pred[endangered_mask])
                metrics['endangered_species_accuracy'] = endangered_accuracy
                
                # Sensitivity for endangered species (critical for conservation)
                endangered_recall = recall_score(y_true, y_pred, labels=endangered_classes, average='macro', zero_division=0)
                metrics['endangered_species_recall'] = endangered_recall
        
        return metrics
    
    def calculate_confidence_metrics(self, y_true: np.ndarray, y_prob: np.ndarray,
                                   confidence_thresholds: List[float] = None) -> Dict[str, float]:
        """
        Calculate confidence-based metrics for reliable deployment
        
        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            confidence_thresholds: List of confidence thresholds to evaluate
            
        Returns:
            Dictionary of confidence metrics
        """
        if confidence_thresholds is None:
            confidence_thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
        
        metrics = {}
        
        # Get predicted labels and confidence scores
        y_pred = np.argmax(y_prob, axis=1)
        confidence_scores = np.max(y_prob, axis=1)
        
        for threshold in confidence_thresholds:
            # Filter predictions by confidence threshold
            confident_mask = confidence_scores >= threshold
            
            if confident_mask.sum() > 0:
                # Accuracy on confident predictions
                confident_accuracy = accuracy_score(
                    y_true[confident_mask], 
                    y_pred[confident_mask]
                )
                metrics[f'accuracy_at_confidence_{threshold}'] = confident_accuracy
                
                # Coverage (percentage of samples with confident predictions)
                coverage = confident_mask.mean()
                metrics[f'coverage_at_confidence_{threshold}'] = coverage
                
                # Quality metric (accuracy weighted by coverage)
                metrics[f'quality_at_confidence_{threshold}'] = confident_accuracy * coverage
            else:
                metrics[f'accuracy_at_confidence_{threshold}'] = 0.0
                metrics[f'coverage_at_confidence_{threshold}'] = 0.0
                metrics[f'quality_at_confidence_{threshold}'] = 0.0
        
        # Calibration metrics
        try:
            from sklearn.calibration import calibration_curve
            
            # Calculate calibration for each class
            calibration_errors = []
            for class_idx in range(self.num_classes):
                binary_true = (y_true == class_idx).astype(int)
                class_prob = y_prob[:, class_idx]
                
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    binary_true, class_prob, n_bins=10
                )
                
                # Expected Calibration Error (ECE)
                calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                calibration_errors.append(calibration_error)
            
            metrics['expected_calibration_error'] = np.mean(calibration_errors)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate calibration metrics: {e}")
        
        return metrics
    
    def generate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                                normalize: str = 'true', save_path: Optional[Path] = None) -> np.ndarray:
        """
        Generate and optionally save confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalization type ('true', 'pred', 'all', None)
            save_path: Path to save the plot
            
        Returns:
            Confusion matrix array
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            if normalize == 'true':
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            elif normalize == 'pred':
                cm = cm.astype('float') / cm.sum(axis=0)
            elif normalize == 'all':
                cm = cm.astype('float') / cm.sum()
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cmap='Blues', cbar=True)
        
        plt.title(f'Confusion Matrix ({normalize} normalized)' if normalize else 'Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrix saved to {save_path}")
        
        return cm
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    save_path: Optional[Path] = None) -> str:
        """
        Generate detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the report
            
        Returns:
            Classification report string
        """
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            digits=3,
            zero_division=0
        )
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Classification report saved to {save_path}")
        
        return report
    
    def plot_precision_recall_curves(self, y_true: np.ndarray, y_prob: np.ndarray,
                                    save_path: Optional[Path] = None):
        """
        Plot precision-recall curves for all classes
        
        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        y_true_onehot = self._to_one_hot(y_true)
        
        for i, class_name in enumerate(self.class_names):
            precision, recall, _ = precision_recall_curve(
                y_true_onehot[:, i], y_prob[:, i]
            )
            
            ap_score = average_precision_score(y_true_onehot[:, i], y_prob[:, i])
            
            plt.plot(recall, precision, label=f'{class_name} (AP={ap_score:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"PR curves saved to {save_path}")
        
        plt.tight_layout()
    
    def plot_roc_curves(self, y_true: np.ndarray, y_prob: np.ndarray,
                       save_path: Optional[Path] = None):
        """
        Plot ROC curves for all classes
        
        Args:
            y_true: True labels
            y_prob: Prediction probabilities  
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        y_true_onehot = self._to_one_hot(y_true)
        
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
            auc_score = roc_auc_score(y_true_onehot[:, i], y_prob[:, i])
            
            plt.plot(fpr, tpr, label=f'{class_name} (AUC={auc_score:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"ROC curves saved to {save_path}")
        
        plt.tight_layout()
    
    def _to_one_hot(self, y: np.ndarray) -> np.ndarray:
        """Convert labels to one-hot encoding"""
        one_hot = np.zeros((len(y), self.num_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                            y_prob: Optional[np.ndarray] = None,
                            save_plots: bool = False,
                            output_dir: Optional[Path] = None) -> Dict[str, float]:
        """
        Calculate all available metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            save_plots: Whether to save visualization plots
            output_dir: Directory to save plots and reports
            
        Returns:
            Dictionary containing all metrics
        """
        all_metrics = {}
        
        # Basic metrics
        all_metrics.update(self.calculate_basic_metrics(y_true, y_pred, y_prob))
        
        # Conservation metrics
        all_metrics.update(self.calculate_conservation_metrics(y_true, y_pred, y_prob))
        
        # Confidence metrics (if probabilities available)
        if y_prob is not None:
            all_metrics.update(self.calculate_confidence_metrics(y_true, y_prob))
        
        # Generate plots and reports
        if save_plots and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Confusion matrix
            self.generate_confusion_matrix(
                y_true, y_pred, normalize='true',
                save_path=output_dir / 'confusion_matrix.png'
            )
            
            # Classification report
            self.generate_classification_report(
                y_true, y_pred,
                save_path=output_dir / 'classification_report.txt'
            )
            
            # Curves (if probabilities available)
            if y_prob is not None:
                self.plot_precision_recall_curves(
                    y_true, y_prob,
                    save_path=output_dir / 'precision_recall_curves.png'
                )
                
                self.plot_roc_curves(
                    y_true, y_prob,
                    save_path=output_dir / 'roc_curves.png'
                )
        
        self.logger.info(f"Calculated {len(all_metrics)} metrics")
        return all_metrics


def calculate_metrics(y_true: Union[np.ndarray, torch.Tensor], 
                     y_pred: Union[np.ndarray, torch.Tensor],
                     y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
                     class_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Convenience function to calculate metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        class_names: List of class names
        
    Returns:
        Dictionary of metrics
    """
    # Convert tensors to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    # Default class names
    if class_names is None:
        num_classes = len(np.unique(y_true))
        class_names = [f'Class_{i}' for i in range(num_classes)]
    
    calculator = MetricsCalculator(class_names)
    return calculator.calculate_basic_metrics(y_true, y_pred, y_prob)