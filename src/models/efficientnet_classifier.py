"""
EfficientNet-B1 Bird Call Classifier
Professional implementation with transfer learning capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from typing import Dict, Optional, List, Tuple
import logging

class BirdCallClassifier(nn.Module):
    """
    EfficientNet-B1 based classifier for bird call detection
    Optimized for endangered species identification
    """
    
    def __init__(self, 
                 num_classes: int = 5,
                 architecture: str = 'efficientnet_b1',
                 pretrained: bool = True,
                 dropout_rate: float = 0.5,
                 use_attention: bool = False,
                 freeze_backbone: bool = False):
        """
        Initialize the bird call classifier
        
        Args:
            num_classes: Number of bird species to classify
            architecture: Model architecture (currently supports efficientnet_b1)
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
            use_attention: Whether to use attention mechanism
            freeze_backbone: Whether to freeze backbone initially
        """
        super(BirdCallClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.architecture = architecture
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize backbone
        self._build_backbone(pretrained, freeze_backbone)
        
        # Build classifier head
        self._build_classifier_head()
        
        # Initialize weights
        self._initialize_weights()
        
        self.logger.info(f"Initialized {architecture} classifier for {num_classes} classes")
        
    def _build_backbone(self, pretrained: bool, freeze_backbone: bool):
        """Build the backbone network"""
        if self.architecture == 'efficientnet_b1':
            weights = EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = efficientnet_b1(weights=weights)
            
            # Remove the original classifier
            self.backbone.classifier = nn.Identity()
            
            # Get feature dimension
            self.feature_dim = 1280  # EfficientNet-B1 feature dimension
            
            # Freeze backbone if requested
            if freeze_backbone:
                self._freeze_backbone()
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
    
    def _freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.logger.info("Backbone frozen")
    
    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.logger.info("Backbone unfrozen")
    
    def _build_classifier_head(self):
        """Build the classification head"""
        layers = []
        
        # Global Average Pooling (already included in backbone)
        
        # Feature processing layers
        layers.extend([
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate)
        ])
        
        # Intermediate layer
        layers.extend([
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate * 0.5)
        ])
        
        # Attention mechanism (optional)
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            layers.append(nn.LayerNorm(256))
        
        # Final classification layer
        layers.append(nn.Linear(256, self.num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights for new layers"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def _convert_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert single-channel mel-spectrogram to 3-channel for EfficientNet
        
        Args:
            x: Input tensor [B, 1, H, W]
            
        Returns:
            RGB tensor [B, 3, H, W]
        """
        # Convert single channel to 3 channels by repeating
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Ensure proper input size for EfficientNet (224x224 minimum)
        if x.shape[2] < 224 or x.shape[3] < 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        return x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from backbone without classification
        
        Args:
            x: Input tensor [B, 1, H, W]
            
        Returns:
            Feature tensor [B, feature_dim]
        """
        # Convert input format
        x = self._convert_input(x)
        
        # Extract features using backbone
        features = self.backbone(x)
        
        return features
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input mel-spectrogram tensor [B, 1, H, W]
            return_features: Whether to return intermediate features
            
        Returns:
            Classification logits [B, num_classes] or tuple with features
        """
        # Extract features
        features = self.extract_features(x)
        
        # Apply attention if enabled
        if self.use_attention:
            # Reshape for attention: [B, 1, feature_dim]
            attn_input = features.unsqueeze(1)
            attn_output, _ = self.attention(attn_input, attn_input, attn_input)
            features = attn_output.squeeze(1)
        
        # Classification head
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities
        
        Args:
            x: Input tensor [B, 1, H, W]
            
        Returns:
            Probability tensor [B, num_classes]
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def predict(self, x: torch.Tensor, return_confidence: bool = False) -> torch.Tensor:
        """
        Make predictions
        
        Args:
            x: Input tensor [B, 1, H, W]
            return_confidence: Whether to return confidence scores
            
        Returns:
            Predicted class indices [B] or tuple with confidence
        """
        probabilities = self.predict_proba(x)
        confidence, predicted = torch.max(probabilities, dim=1)
        
        if return_confidence:
            return predicted, confidence
        return predicted
    
    def get_cam(self, x: torch.Tensor, target_class: Optional[int] = None) -> torch.Tensor:
        """
        Generate Class Activation Map (CAM) for interpretation
        
        Args:
            x: Input tensor [B, 1, H, W]
            target_class: Target class for CAM (uses predicted class if None)
            
        Returns:
            CAM tensor [B, H, W]
        """
        self.eval()
        
        # Register hook to capture feature maps
        features_maps = []
        def hook_feature(module, input, output):
            features_maps.append(output)
        
        # Find the last convolutional layer
        last_conv_layer = None
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv_layer = module
        
        if last_conv_layer is None:
            raise ValueError("No convolutional layer found for CAM generation")
        
        handle = last_conv_layer.register_forward_hook(hook_feature)
        
        try:
            # Forward pass
            logits = self.forward(x)
            
            if target_class is None:
                target_class = torch.argmax(logits, dim=1)
            
            # Get feature maps
            feature_maps = features_maps[-1]  # [B, C, H, W]
            
            # Get weights from classifier
            classifier_weights = self.classifier[-1].weight  # [num_classes, feature_dim]
            
            # Generate CAM
            batch_size = x.shape[0]
            cams = []
            
            for i in range(batch_size):
                class_idx = target_class[i] if torch.is_tensor(target_class) else target_class
                weights = classifier_weights[class_idx]  # [feature_dim]
                
                # Weighted combination of feature maps
                feature_map = feature_maps[i]  # [C, H, W]
                cam = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * feature_map, dim=0)
                
                # Normalize
                cam = F.relu(cam)
                cam = cam - cam.min()
                cam = cam / (cam.max() + 1e-8)
                
                cams.append(cam)
            
            cam_tensor = torch.stack(cams, dim=0)
            
        finally:
            handle.remove()
        
        return cam_tensor


class MultiScaleBirdClassifier(BirdCallClassifier):
    """
    Multi-scale version of the bird classifier for handling varying audio lengths
    """
    
    def __init__(self, scales: List[int] = [224, 256, 288], **kwargs):
        """
        Initialize multi-scale classifier
        
        Args:
            scales: List of input scales to process
            **kwargs: Arguments for base classifier
        """
        super().__init__(**kwargs)
        self.scales = scales
        
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """Multi-scale forward pass"""
        scale_outputs = []
        
        for scale in self.scales:
            # Resize input to current scale
            if x.shape[2] != scale or x.shape[3] != scale:
                scaled_x = F.interpolate(x, size=(scale, scale), mode='bilinear', align_corners=False)
            else:
                scaled_x = x
            
            # Forward pass at current scale
            output = super().forward(scaled_x, return_features=False)
            scale_outputs.append(output)
        
        # Ensemble the outputs
        ensemble_output = torch.mean(torch.stack(scale_outputs, dim=0), dim=0)
        
        if return_features:
            # Return features from the largest scale
            features = self.extract_features(
                F.interpolate(x, size=(max(self.scales), max(self.scales)), 
                            mode='bilinear', align_corners=False)
            )
            return ensemble_output, features
        
        return ensemble_output


def create_model(config: Dict) -> BirdCallClassifier:
    """
    Factory function to create model from configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    model_config = config['model']
    
    model = BirdCallClassifier(
        num_classes=model_config['num_classes'],
        architecture=model_config['architecture'],
        pretrained=model_config['pretrained'],
        dropout_rate=model_config['dropout_rate'],
        use_attention=model_config.get('use_attention', False),
        freeze_backbone=model_config.get('freeze_backbone', False)
    )
    
    return model


def load_pretrained_model(checkpoint_path: str, config: Dict, device: str = 'cpu') -> BirdCallClassifier:
    """
    Load a pretrained model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Model configuration
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    # Create model
    model = create_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    
    return model


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def model_summary(model: torch.nn.Module, input_shape: Tuple[int, ...] = (1, 128, 157)) -> str:
    """
    Generate model summary
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (excluding batch dimension)
        
    Returns:
        Model summary string
    """
    total_params, trainable_params = count_parameters(model)
    
    summary = f"""
    Model: {model.__class__.__name__}
    Total parameters: {total_params:,}
    Trainable parameters: {trainable_params:,}
    Non-trainable parameters: {total_params - trainable_params:,}
    Input shape: {input_shape}
    """
    
    return summary