"""
Professional Training Pipeline for Bird Call Detection
Comprehensive trainer with monitoring, checkpointing, and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time
import logging
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

class BirdDetectionTrainer:
    """
    Professional trainer for bird call detection models
    Features comprehensive monitoring, checkpointing, and evaluation
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 config: Dict,
                 device: str = 'auto',
                 checkpoint_dir: Optional[Path] = None):
        """
        Initialize the trainer
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Device to use ('auto', 'cpu', 'cuda')
            checkpoint_dir: Directory to save checkpoints
        """
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup directories
        self.checkpoint_dir = checkpoint_dir or Path(config['paths']['checkpoints_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        self.training_config = config['training']
        self.hardware_config = config['hardware']
        
        # Setup components
        self._setup_loss_function()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_mixed_precision()
        
        # Monitoring
        self.train_history = {'loss': [], 'accuracy': []}
        self.val_history = {'loss': [], 'accuracy': [], 'f1_score': [], 'precision': [], 'recall': []}
        self.best_metric = 0.0
        self.patience_counter = 0
        self.current_epoch = 0
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Trainer initialized with device: {self.device}")
        
    def _setup_loss_function(self):
        """Setup loss function with class weighting if enabled"""
        if self.training_config.get('use_class_weights', False):
            # Calculate class weights from training data
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=self.training_config.get('label_smoothing', 0.0)
            )
            self.logger.info("Using CrossEntropyLoss with label smoothing")
        else:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=self.training_config.get('label_smoothing', 0.0)
            )
            
    def _setup_optimizer(self):
        """Setup optimizer based on configuration"""
        optimizer_name = self.training_config.get('optimizer', 'adamw').lower()
        lr = self.training_config['learning_rate']
        weight_decay = self.training_config['weight_decay']
        
        if optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
        self.logger.info(f"Using {optimizer_name.upper()} optimizer with LR: {lr}")
        
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_name = self.training_config.get('scheduler', 'reduce_lr_on_plateau')
        
        if scheduler_name == 'reduce_lr_on_plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max' if 'accuracy' in self.training_config.get('monitor_metric', 'val_accuracy') else 'min',
                patience=self.training_config.get('reduce_lr_patience', 5),
                factor=self.training_config.get('reduce_lr_factor', 0.5),
                min_lr=self.training_config.get('min_lr', 1e-7),
                verbose=True
            )
        elif scheduler_name == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config['epochs'],
                eta_min=self.training_config.get('min_lr', 1e-7)
            )
        elif scheduler_name == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=self.training_config['epochs'] // 3,
                gamma=0.1
            )
        else:
            self.scheduler = None
            
        self.logger.info(f"Using {scheduler_name} scheduler")
        
    def _setup_mixed_precision(self):
        """Setup mixed precision training if enabled and supported"""
        self.use_amp = (self.hardware_config.get('mixed_precision', False) and 
                       torch.cuda.is_available())
        
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info("Using Automatic Mixed Precision (AMP)")
        else:
            self.scaler = None
            self.logger.info("Using standard precision training")
            
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}", leave=False)
        
        for batch_idx, (data, target) in enumerate(pbar):
            try:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                
                # Forward pass with AMP if enabled
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                predictions = output.argmax(dim=1)
                correct_predictions += (predictions == target).sum().item()
                total_samples += target.size(0)
                
                # Update progress bar
                current_acc = 100. * correct_predictions / total_samples
                current_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.2f}%',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
            except Exception as e:
                self.logger.warning(f"Training batch {batch_idx} failed: {e}")
                continue
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self) -> Tuple[float, float, Dict, Tuple]:
        """Validate for one epoch with detailed metrics"""
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validation {self.current_epoch}", leave=False)
            
            for batch_idx, (data, target) in enumerate(pbar):
                try:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Forward pass
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    # Statistics
                    running_loss += loss.item()
                    
                    # Store predictions and probabilities
                    probabilities = torch.softmax(output, dim=1)
                    predictions = output.argmax(dim=1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                    
                    # Update progress bar
                    current_loss = running_loss / (batch_idx + 1)
                    pbar.set_postfix({'Loss': f'{current_loss:.4f}'})
                    
                except Exception as e:
                    self.logger.warning(f"Validation batch {batch_idx} failed: {e}")
                    continue
        
        epoch_loss = running_loss / len(self.val_loader)
        
        # Calculate comprehensive metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # Basic accuracy
        epoch_acc = 100. * accuracy_score(all_targets, all_predictions)
        
        # Detailed metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        detailed_metrics = {
            'accuracy': epoch_acc / 100,
            'precision': precision_score(all_targets, all_predictions, average='weighted', zero_division=0),
            'recall': recall_score(all_targets, all_predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        }
        
        return epoch_loss, epoch_acc, detailed_metrics, (all_targets, all_predictions, all_probabilities)
    
    def train(self, num_epochs: Optional[int] = None, resume_from_checkpoint: bool = False) -> Dict:
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train (uses config if None)
            resume_from_checkpoint: Whether to resume from latest checkpoint
            
        Returns:
            Training history dictionary
        """
        if num_epochs is None:
            num_epochs = self.training_config['epochs']
            
        # Resume from checkpoint if requested
        if resume_from_checkpoint:
            self._load_latest_checkpoint()
            
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}, Mixed Precision: {self.use_amp}")
        
        training_start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch + 1, num_epochs + 1):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # Training
                train_loss, train_acc = self.train_epoch()
                
                # Validation
                val_loss, val_acc, detailed_metrics, validation_data = self.validate_epoch()
                
                # Update scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        metric = detailed_metrics.get(self.training_config.get('monitor_metric', 'f1_score'), val_acc/100)
                        self.scheduler.step(metric)
                    else:
                        self.scheduler.step()
                
                # Store history
                self.train_history['loss'].append(train_loss)
                self.train_history['accuracy'].append(train_acc)
                self.val_history['loss'].append(val_loss)
                self.val_history['accuracy'].append(val_acc)
                self.val_history['f1_score'].append(detailed_metrics['f1_score'])
                self.val_history['precision'].append(detailed_metrics['precision'])
                self.val_history['recall'].append(detailed_metrics['recall'])
                
                # Print epoch results
                epoch_time = time.time() - epoch_start_time
                self._print_epoch_results(epoch, train_loss, train_acc, val_loss, val_acc, 
                                        detailed_metrics, epoch_time)
                
                # Check for best model
                monitor_metric = self.training_config.get('monitor_metric', 'f1_score')
                current_metric = detailed_metrics.get(monitor_metric, val_acc/100)
                
                is_best = current_metric > self.best_metric
                if is_best:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    self._save_best_model(epoch, validation_data)
                else:
                    self.patience_counter += 1
                
                # Early stopping
                early_stopping_patience = self.training_config.get('early_stopping_patience', 10)
                if self.patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
                
                # Save checkpoint
                if epoch % 5 == 0:
                    self._save_checkpoint(epoch)
                
                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
            
        training_time = time.time() - training_start_time
        self.logger.info(f"Training completed in {training_time/60:.2f} minutes")
        self.logger.info(f"Best {monitor_metric}: {self.best_metric:.4f}")
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_metric': self.best_metric,
            'training_time': training_time
        }
    
    def _print_epoch_results(self, epoch: int, train_loss: float, train_acc: float,
                           val_loss: float, val_acc: float, detailed_metrics: Dict, epoch_time: float):
        """Print formatted epoch results"""
        print(f"\\nEpoch {epoch} Results ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Val F1: {detailed_metrics['f1_score']:.4f} | Val Precision: {detailed_metrics['precision']:.4f} | Val Recall: {detailed_metrics['recall']:.4f}")
        print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Patience: {self.patience_counter}/{self.training_config.get('early_stopping_patience', 10)}")
        
    def _save_best_model(self, epoch: int, validation_data: Tuple):
        """Save the best model"""
        model_path = self.checkpoint_dir / 'best_bird_detector.pth'
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'model_class': self.model.__class__.__name__
        }
        
        torch.save(checkpoint, model_path)
        self.logger.info(f"Best model saved: {model_path}")
        
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint"""
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_metric': self.best_metric,
            'patience_counter': self.patience_counter
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path.name}")
        
    def _load_latest_checkpoint(self):
        """Load the latest checkpoint for resuming training"""
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
            self._load_checkpoint(latest_checkpoint)
            self.logger.info(f"Resumed from {latest_checkpoint.name}")
        else:
            self.logger.warning("No checkpoint found to resume from")
            
    def _load_checkpoint(self, checkpoint_path: Path):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.current_epoch = checkpoint['epoch']
        self.train_history = checkpoint.get('train_history', {'loss': [], 'accuracy': []})
        self.val_history = checkpoint.get('val_history', {'loss': [], 'accuracy': [], 'f1_score': [], 'precision': [], 'recall': []})
        self.best_metric = checkpoint.get('best_metric', 0.0)
        self.patience_counter = checkpoint.get('patience_counter', 0)
        
    def plot_training_history(self, save_path: Optional[Path] = None):
        """Plot training history"""
        if not self.train_history['loss']:
            self.logger.warning("No training history to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.train_history['loss']) + 1)
        
        # Loss
        axes[0,0].plot(epochs, self.train_history['loss'], 'b-', label='Train Loss')
        axes[0,0].plot(epochs, self.val_history['loss'], 'r-', label='Val Loss')
        axes[0,0].set_title('Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Accuracy
        axes[0,1].plot(epochs, self.train_history['accuracy'], 'b-', label='Train Acc')
        axes[0,1].plot(epochs, self.val_history['accuracy'], 'r-', label='Val Acc')
        axes[0,1].set_title('Accuracy')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Accuracy (%)')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # F1 Score
        axes[1,0].plot(epochs, self.val_history['f1_score'], 'g-', label='F1 Score')
        axes[1,0].plot(epochs, self.val_history['precision'], 'orange', label='Precision')
        axes[1,0].plot(epochs, self.val_history['recall'], 'purple', label='Recall')
        axes[1,0].set_title('Validation Metrics')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Score')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Learning rate
        if hasattr(self.scheduler, 'get_last_lr'):
            lr_history = [self.optimizer.param_groups[0]['lr']] * len(epochs)
            axes[1,1].plot(epochs, lr_history, 'm-')
            axes[1,1].set_title('Learning Rate')
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Learning Rate')
            axes[1,1].set_yscale('log')
            axes[1,1].grid(True)
        else:
            axes[1,1].text(0.5, 0.5, 'Learning Rate\\nHistory\\nNot Available', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Learning Rate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training plots saved to {save_path}")
        
        plt.show()
        
    def evaluate_model(self, test_loader, class_names: List[str]) -> Dict:
        """Evaluate model on test set"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating"):
                data = data.to(self.device)
                output = self.model(data)
                
                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        accuracy = accuracy_score(all_targets, all_predictions)
        report = classification_report(all_targets, all_predictions, 
                                     target_names=class_names, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }