# Smart Industrial Defect Detection - EfficientNet Classifier
"""
EfficientNet-based defect classifier.
Used for fine-grained classification after detection.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-based defect classifier.
    Supports multiple EfficientNet variants and custom heads.
    """
    
    def __init__(self,
                 model_name: str = 'efficientnet_b4',
                 num_classes: int = 5,
                 pretrained: bool = True,
                 dropout_rate: float = 0.4,
                 use_attention: bool = True):
        """
        Initialize classifier.
        
        Args:
            model_name: EfficientNet variant (b0-b7, efficientnetv2_s/m/l)
            num_classes: Number of defect classes
            pretrained: Use ImageNet pretrained weights
            dropout_rate: Dropout rate before classifier
            use_attention: Add attention mechanism
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        self.use_attention = use_attention
        
        # Load pretrained model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classifier head
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Attention module (SE-style)
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim // 16),
                nn.ReLU(),
                nn.Linear(self.feature_dim // 16, self.feature_dim),
                nn.Sigmoid()
            )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )
        
        # Class names
        self.class_names = [
            'scratch',
            'crack',
            'dent',
            'missing_component',
            'contamination'
        ]
        
        logger.info(f"Initialized {model_name} classifier with {num_classes} classes")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            Class logits (B, num_classes)
        """
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        if self.use_attention:
            attention_weights = self.attention(features)
            features = features * attention_weights
        
        # Classify
        logits = self.classifier(features)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> Dict:
        """
        Predict with probabilities.
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            Dictionary with predictions and probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            predictions = probs.argmax(dim=1)
            confidences = probs.max(dim=1).values
        
        results = []
        for i in range(len(predictions)):
            results.append({
                'class_id': int(predictions[i]),
                'class_name': self.class_names[predictions[i]],
                'confidence': float(confidences[i]),
                'probabilities': probs[i].cpu().numpy().tolist()
            })
        
        return results
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification.
        
        Args:
            x: Input tensor
        
        Returns:
            Feature tensor
        """
        self.eval()
        with torch.no_grad():
            features = self.backbone(x)
            if self.use_attention:
                attention_weights = self.attention(features)
                features = features * attention_weights
        return features
    
    def get_cam(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate Class Activation Map (CAM) for interpretability.
        
        Args:
            x: Input tensor (1, 3, H, W)
            class_idx: Target class index (None = predicted class)
        
        Returns:
            CAM heatmap
        """
        self.eval()
        
        # Get feature maps before global pooling
        features = None
        def hook_fn(module, input, output):
            nonlocal features
            features = output
        
        # Register hook on last conv layer
        handle = self.backbone.conv_head.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            logits = self.forward(x)
            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()
        
        handle.remove()
        
        # Get weights from classifier
        weights = self.classifier[-1].weight[class_idx]  # (feature_dim,)
        
        # Compute CAM
        cam = torch.zeros(features.shape[2:], device=features.device)
        for i, w in enumerate(weights):
            cam += w * features[0, i]
        
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()


class DefectClassificationPipeline:
    """
    Complete classification pipeline with preprocessing and postprocessing.
    """
    
    def __init__(self,
                 model_path: str = None,
                 model_name: str = 'efficientnet_b4',
                 num_classes: int = 5,
                 device: str = 'cuda:0',
                 input_size: int = 224):
        """
        Initialize pipeline.
        
        Args:
            model_path: Path to trained model weights
            model_name: EfficientNet variant
            num_classes: Number of classes
            device: Device to run on
            input_size: Input image size
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        
        # Initialize model
        self.model = EfficientNetClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=model_path is None
        ).to(self.device)
        
        # Load weights if provided
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded classifier weights from {model_path}")
        
        self.model.eval()
        
        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image (BGR or RGB)
        
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (self.input_size, self.input_size))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        
        # To tensor (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return tensor.unsqueeze(0).to(self.device)
    
    def classify(self, image: np.ndarray) -> Dict:
        """
        Classify a single image.
        
        Args:
            image: Input image
        
        Returns:
            Classification result
        """
        tensor = self.preprocess(image)
        results = self.model.predict(tensor)
        return results[0]
    
    def classify_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Classify multiple images.
        
        Args:
            images: List of images
        
        Returns:
            List of classification results
        """
        # Preprocess all images
        tensors = [self.preprocess(img) for img in images]
        batch = torch.cat(tensors, dim=0)
        
        # Classify
        results = self.model.predict(batch)
        return results
    
    def classify_roi(self, 
                     image: np.ndarray,
                     bbox: List[float]) -> Dict:
        """
        Classify a region of interest (ROI) from detection.
        
        Args:
            image: Full image
            bbox: Bounding box [x1, y1, x2, y2]
        
        Returns:
            Classification result
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Extract ROI with padding
        h, w = image.shape[:2]
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return {'class_id': -1, 'class_name': 'unknown', 'confidence': 0.0}
        
        return self.classify(roi)
    
    def get_explanation(self, image: np.ndarray) -> Tuple[Dict, np.ndarray]:
        """
        Get classification with CAM explanation.
        
        Args:
            image: Input image
        
        Returns:
            Tuple of (classification result, CAM heatmap overlay)
        """
        tensor = self.preprocess(image)
        
        # Get prediction
        results = self.model.predict(tensor)
        result = results[0]
        
        # Get CAM
        cam = self.model.get_cam(tensor, result['class_id'])
        
        # Resize CAM to image size
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # Create heatmap overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        
        return result, overlay


def train_classifier(model: EfficientNetClassifier,
                     train_loader,
                     val_loader,
                     epochs: int = 50,
                     learning_rate: float = 0.001,
                     weight_decay: float = 0.0001,
                     class_weights: List[float] = None,
                     device: str = 'cuda:0',
                     save_dir: str = 'runs/classifier') -> Dict:
    """
    Train EfficientNet classifier.
    
    Args:
        model: Classifier model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        class_weights: Class weights for imbalanced data
        device: Device to train on
        save_dir: Save directory
    
    Returns:
        Training history
    """
    device = torch.device(device)
    model = model.to(device)
    
    # Loss function with class weights
    if class_weights:
        weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        scheduler.step()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, save_path / 'best_classifier.pt')
            logger.info(f"Saved best model with val_acc={val_acc:.2f}%")
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'history': history
    }, save_path / 'final_classifier.pt')
    
    return history
