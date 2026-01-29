# Smart Industrial Defect Detection - Core Models
"""
YOLOv8 wrapper, EfficientNet classifier, and Autoencoder anomaly detector.
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
import timm
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class YOLOv8Detector:
    """Wrapper for YOLOv8 detection model."""
    
    def __init__(self, model_path: str = None, device: str = 'cuda:0'):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_path: Path to pre-trained model
            device: Device to run on (cuda:0, cpu, etc.)
        """
        self.device = device
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLOv8 from {model_path}")
        else:
            # Load pretrained YOLOv8n (nano) as default
            self.model = YOLO('yolov8n.pt')
            logger.info("Loaded default YOLOv8n pretrained model")
        
        self.model.to(device)
    
    def predict(self, image: torch.Tensor, 
                conf_threshold: float = 0.6,
                nms_threshold: float = 0.45) -> Dict:
        """
        Perform inference on image.
        
        Args:
            image: Input image (numpy or torch tensor)
            conf_threshold: Confidence threshold
            nms_threshold: NMS threshold
        
        Returns:
            Dictionary with detections
        """
        results = self.model.predict(
            image,
            conf=conf_threshold,
            iou=nms_threshold,
            device=self.device,
            verbose=False
        )
        
        if results:
            result = results[0]
            detections = {
                'boxes': result.boxes.xyxy.cpu().numpy(),  # [N, 4]
                'confidences': result.boxes.conf.cpu().numpy(),  # [N]
                'class_ids': result.boxes.cls.cpu().numpy().astype(int),  # [N]
            }
        else:
            detections = {
                'boxes': np.array([]),
                'confidences': np.array([]),
                'class_ids': np.array([])
            }
        
        return detections


class EfficientNetClassifier(nn.Module):
    """EfficientNet-based defect classifier."""
    
    def __init__(self, model_name: str = 'efficientnet_b4',
                 num_classes: int = 5,
                 pretrained: bool = True):
        """
        Initialize classifier.
        
        Args:
            model_name: EfficientNet variant
            num_classes: Number of defect classes
            pretrained: Use ImageNet pretrained weights
        """
        super().__init__()
        
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        logger.info(f"Loaded {model_name} with {num_classes} classes")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


class AnomalyAutoencoder(nn.Module):
    """Autoencoder for anomaly detection."""
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 32):
        """
        Initialize autoencoder.
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            latent_dim: Latent representation dimension
        """
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc_encode = nn.Linear(512, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * 4 * 4)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        self.latent_dim = latent_dim
        logger.info(f"Initialized Autoencoder with latent_dim={latent_dim}")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        latent = self.fc_encode(features)
        return latent
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        features = self.fc_decode(z)
        features = features.view(features.size(0), 512, 4, 4)
        reconstruction = self.decoder(features)
        return reconstruction
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass (returns reconstruction and latent)."""
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z


class EnsembleDefectDetector:
    """
    Ensemble of detection, classification, and anomaly detection models.
    Combines predictions for robust decision making.
    """
    
    def __init__(self, config):
        """
        Initialize ensemble.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device(config.inference.device)
        
        # Load models
        self.detector = YOLOv8Detector(
            model_path=config.model.detection_model_path,
            device=config.inference.device
        )
        
        self.classifier = EfficientNetClassifier(
            num_classes=config.data.num_classes,
            pretrained=False
        ).to(self.device)
        
        if os.path.exists(config.model.classification_model_path):
            checkpoint = torch.load(config.model.classification_model_path,
                                  map_location=self.device)
            self.classifier.load_state_dict(checkpoint)
            logger.info("Loaded classifier weights")
        
        self.classifier.eval()
        
        # Anomaly detector
        self.anomaly_detector = AnomalyAutoencoder().to(self.device)
        if os.path.exists(config.model.anomaly_model_path):
            checkpoint = torch.load(config.model.anomaly_model_path,
                                  map_location=self.device)
            self.anomaly_detector.load_state_dict(checkpoint)
            logger.info("Loaded anomaly detector weights")
        
        self.anomaly_detector.eval()
        
        logger.info("Ensemble model initialized successfully")
    
    @torch.no_grad()
    def detect_and_classify(self, image: np.ndarray) -> Dict:
        """
        Perform detection and classification.
        
        Args:
            image: Input image (numpy array, BGR)
        
        Returns:
            Dictionary with detection and classification results
        """
        # Detection
        detections = self.detector.predict(
            image,
            conf_threshold=self.config.model.detection_confidence_threshold,
            nms_threshold=self.config.model.detection_nms_threshold
        )
        
        results = {
            'detections': detections,
            'classifications': [],
            'anomaly_scores': []
        }
        
        if len(detections['boxes']) == 0:
            return results
        
        # Classification and anomaly detection for each detection
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.to(self.device)
        
        for box in detections['boxes']:
            x1, y1, x2, y2 = box.astype(int)
            crop = image_tensor[:, y1:y2, x1:x2]
            crop = torch.nn.functional.interpolate(
                crop.unsqueeze(0),
                size=(224, 224),
                mode='bilinear'
            )
            
            # Classification
            logits = self.classifier(crop)
            probs = torch.softmax(logits, dim=1)
            class_id = probs.argmax(dim=1).item()
            confidence = probs[0, class_id].item()
            
            results['classifications'].append({
                'class_id': class_id,
                'class_name': self.config.data.defect_classes[class_id],
                'confidence': confidence
            })
            
            # Anomaly detection
            if self.config.model.anomaly_enable:
                reconstruction, _ = self.anomaly_detector(crop)
                mse = torch.mean((crop - reconstruction) ** 2)
                results['anomaly_scores'].append(mse.item())
        
        return results


# Import required modules
import os
import numpy as np
