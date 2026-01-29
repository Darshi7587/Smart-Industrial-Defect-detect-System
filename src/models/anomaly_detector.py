# Smart Industrial Defect Detection - Anomaly Detection
"""
Anomaly detection for unknown/novel defects.
Uses Autoencoder reconstruction error and Isolation Forest.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import joblib
import logging

logger = logging.getLogger(__name__)


class AnomalyAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for anomaly detection.
    Trained on normal products, high reconstruction error indicates anomaly.
    """
    
    def __init__(self,
                 input_channels: int = 3,
                 input_size: int = 224,
                 latent_dim: int = 128,
                 base_channels: int = 32):
        """
        Initialize autoencoder.
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            input_size: Input image size
            latent_dim: Latent representation dimension
            base_channels: Base number of channels in conv layers
        """
        super().__init__()
        
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        
        # Encoder
        self.encoder = nn.Sequential(
            # 224 -> 112
            nn.Conv2d(input_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2),
            
            # 112 -> 56
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2),
            
            # 56 -> 28
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2),
            
            # 28 -> 14
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2),
            
            # 14 -> 7
            nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 16),
            nn.LeakyReLU(0.2),
        )
        
        # Calculate flattened size
        self.flatten_size = base_channels * 16 * 7 * 7
        
        # Latent space
        self.fc_encode = nn.Linear(self.flatten_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            # 7 -> 14
            nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(),
            
            # 14 -> 28
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(),
            
            # 28 -> 56
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            
            # 56 -> 112
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            
            # 112 -> 224
            nn.ConvTranspose2d(base_channels, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
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
        features = features.view(features.size(0), -1, 7, 7)
        reconstruction = self.decoder(features)
        return reconstruction
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Tuple of (reconstruction, latent)
        """
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z
    
    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample reconstruction error.
        
        Args:
            x: Input tensor
        
        Returns:
            Reconstruction error per sample
        """
        reconstruction, _ = self.forward(x)
        
        # Per-sample MSE
        error = F.mse_loss(reconstruction, x, reduction='none')
        error = error.view(error.size(0), -1).mean(dim=1)
        
        return error


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) for anomaly detection.
    Provides probabilistic latent space for better anomaly detection.
    """
    
    def __init__(self,
                 input_channels: int = 3,
                 input_size: int = 224,
                 latent_dim: int = 128,
                 base_channels: int = 32):
        """Initialize VAE."""
        super().__init__()
        
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        # Encoder (same as autoencoder)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 16),
            nn.LeakyReLU(0.2),
        )
        
        self.flatten_size = base_channels * 16 * 7 * 7
        
        # VAE latent layers
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent distribution parameters."""
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent vector."""
        features = self.fc_decode(z)
        features = features.view(features.size(0), -1, 7, 7)
        return self.decoder(features)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly score based on reconstruction error + KL divergence.
        """
        reconstruction, mu, logvar = self.forward(x)
        
        # Reconstruction error
        recon_error = F.mse_loss(reconstruction, x, reduction='none')
        recon_error = recon_error.view(recon_error.size(0), -1).mean(dim=1)
        
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        # Combined anomaly score
        anomaly_score = recon_error + 0.01 * kl_div
        
        return anomaly_score


class IsolationForestAnomalyDetector:
    """
    Isolation Forest for feature-based anomaly detection.
    Uses features from classification model.
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 contamination: float = 0.1,
                 random_state: int = 42):
        """
        Initialize Isolation Forest.
        
        Args:
            n_estimators: Number of trees
            contamination: Expected proportion of outliers
            random_state: Random seed
        """
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        logger.info(f"Initialized IsolationForest with n_estimators={n_estimators}")
    
    def fit(self, features: np.ndarray) -> 'IsolationForestAnomalyDetector':
        """
        Fit on normal product features.
        
        Args:
            features: Feature array (N, D)
        
        Returns:
            self
        """
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit model
        self.model.fit(features_scaled)
        self.is_fitted = True
        
        logger.info(f"Fitted IsolationForest on {len(features)} samples")
        return self
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            features: Feature array
        
        Returns:
            Labels: 1 for normal, -1 for anomaly
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)
    
    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (lower = more anomalous).
        
        Args:
            features: Feature array
        
        Returns:
            Anomaly scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        features_scaled = self.scaler.transform(features)
        return self.model.score_samples(features_scaled)
    
    def save(self, path: str):
        """Save model to file."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }, path)
        logger.info(f"Saved IsolationForest to {path}")
    
    def load(self, path: str) -> 'IsolationForestAnomalyDetector':
        """Load model from file."""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_fitted = data['is_fitted']
        logger.info(f"Loaded IsolationForest from {path}")
        return self


class AnomalyDetectionPipeline:
    """
    Combined anomaly detection pipeline.
    Uses both autoencoder and isolation forest.
    """
    
    def __init__(self,
                 autoencoder_path: str = None,
                 isolation_forest_path: str = None,
                 device: str = 'cuda:0',
                 input_size: int = 224,
                 anomaly_threshold: float = 2.0):
        """
        Initialize pipeline.
        
        Args:
            autoencoder_path: Path to trained autoencoder
            isolation_forest_path: Path to trained isolation forest
            device: Device to run on
            input_size: Input image size
            anomaly_threshold: Threshold for anomaly detection (std devs from mean)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.anomaly_threshold = anomaly_threshold
        
        # Initialize autoencoder
        self.autoencoder = AnomalyAutoencoder(
            input_size=input_size,
            latent_dim=128
        ).to(self.device)
        
        if autoencoder_path and os.path.exists(autoencoder_path):
            checkpoint = torch.load(autoencoder_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.autoencoder.load_state_dict(checkpoint)
            logger.info(f"Loaded autoencoder from {autoencoder_path}")
        
        self.autoencoder.eval()
        
        # Initialize isolation forest
        self.isolation_forest = IsolationForestAnomalyDetector()
        if isolation_forest_path and os.path.exists(isolation_forest_path):
            self.isolation_forest.load(isolation_forest_path)
        
        # Baseline stats (computed during calibration)
        self.baseline_mean = None
        self.baseline_std = None
        
        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = image.astype(np.float32) / 255.0
        
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        return tensor.unsqueeze(0).to(self.device)
    
    def calibrate(self, normal_images: List[np.ndarray]):
        """
        Calibrate anomaly detection on normal images.
        Computes baseline reconstruction error statistics.
        
        Args:
            normal_images: List of normal product images
        """
        errors = []
        features = []
        
        with torch.no_grad():
            for img in normal_images:
                tensor = self.preprocess(img)
                error = self.autoencoder.compute_reconstruction_error(tensor)
                errors.append(error.item())
                
                # Get latent features for isolation forest
                latent = self.autoencoder.encode(tensor)
                features.append(latent.cpu().numpy().flatten())
        
        errors = np.array(errors)
        self.baseline_mean = errors.mean()
        self.baseline_std = errors.std()
        
        # Fit isolation forest
        features = np.array(features)
        self.isolation_forest.fit(features)
        
        logger.info(f"Calibrated on {len(normal_images)} normal images. "
                   f"Mean error: {self.baseline_mean:.4f}, Std: {self.baseline_std:.4f}")
    
    def detect_anomaly(self, image: np.ndarray) -> Dict:
        """
        Detect if image is anomalous.
        
        Args:
            image: Input image
        
        Returns:
            Dictionary with anomaly detection results
        """
        tensor = self.preprocess(image)
        
        with torch.no_grad():
            # Autoencoder-based detection
            error = self.autoencoder.compute_reconstruction_error(tensor).item()
            reconstruction, latent = self.autoencoder(tensor)
            
            # Z-score based anomaly
            if self.baseline_mean is not None:
                z_score = (error - self.baseline_mean) / (self.baseline_std + 1e-8)
                ae_is_anomaly = z_score > self.anomaly_threshold
            else:
                z_score = 0.0
                ae_is_anomaly = False
            
            # Isolation forest based detection
            if self.isolation_forest.is_fitted:
                if_score = self.isolation_forest.score_samples(latent.cpu().numpy())[0]
                if_is_anomaly = if_score < -0.5  # More negative = more anomalous
            else:
                if_score = 0.0
                if_is_anomaly = False
        
        # Combined decision (either method flags anomaly)
        is_anomaly = ae_is_anomaly or if_is_anomaly
        
        return {
            'is_anomaly': is_anomaly,
            'reconstruction_error': error,
            'z_score': z_score,
            'isolation_forest_score': if_score,
            'ae_anomaly': ae_is_anomaly,
            'if_anomaly': if_is_anomaly,
            'confidence': min(1.0, abs(z_score) / self.anomaly_threshold) if z_score > 0 else 0.0
        }
    
    def get_reconstruction(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get reconstruction and difference map.
        
        Args:
            image: Input image
        
        Returns:
            Tuple of (reconstruction, difference_map)
        """
        tensor = self.preprocess(image)
        
        with torch.no_grad():
            reconstruction, _ = self.autoencoder(tensor)
        
        # Convert back to image
        recon_np = reconstruction[0].cpu().numpy().transpose(1, 2, 0)
        recon_np = (recon_np * 255).astype(np.uint8)
        
        # Resize to original
        original_resized = cv2.resize(image, (self.input_size, self.input_size))
        if len(original_resized.shape) == 3:
            original_resized = cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB)
        
        # Compute difference map
        diff = np.abs(original_resized.astype(np.float32) - recon_np.astype(np.float32))
        diff = diff.mean(axis=2) if len(diff.shape) == 3 else diff
        diff = (diff / diff.max() * 255).astype(np.uint8)
        
        # Apply colormap
        diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        
        return recon_np, diff_colored


def train_autoencoder(model: AnomalyAutoencoder,
                      train_loader,
                      val_loader,
                      epochs: int = 100,
                      learning_rate: float = 0.001,
                      device: str = 'cuda:0',
                      save_dir: str = 'runs/anomaly') -> Dict:
    """
    Train autoencoder on normal images.
    
    Args:
        model: Autoencoder model
        train_loader: Training data loader (normal images only)
        val_loader: Validation data loader
        epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Save directory
    
    Returns:
        Training history
    """
    device = torch.device(device)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            
            optimizer.zero_grad()
            reconstruction, _ = model(images)
            loss = F.mse_loss(reconstruction, images)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                reconstruction, _ = model(images)
                loss = F.mse_loss(reconstruction, images)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss
            }, save_path / 'best_autoencoder.pt')
    
    return history
