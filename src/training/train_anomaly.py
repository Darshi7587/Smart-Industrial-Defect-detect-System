# Smart Industrial Defect Detection - Anomaly Detection Training
"""
Training script for autoencoder-based anomaly detection.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np
import cv2
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.anomaly_detector import AnomalyAutoencoder, VariationalAutoencoder

logger = logging.getLogger(__name__)


class NormalImageDataset(Dataset):
    """Dataset for normal (non-defective) images only."""
    
    def __init__(self, images_dir: str, image_size: int = 224):
        """
        Initialize dataset.
        
        Args:
            images_dir: Directory with normal images
            image_size: Target image size
        """
        self.image_size = image_size
        self.image_paths = []
        
        images_path = Path(images_dir)
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            self.image_paths.extend(list(images_path.glob(ext)))
        
        logger.info(f"Loaded {len(self.image_paths)} normal images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # To tensor (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image.transpose(2, 0, 1))
        
        return {'image': image, 'path': str(self.image_paths[idx])}


def train_autoencoder(normal_images_dir: str,
                      val_images_dir: str = None,
                      model_type: str = 'ae',  # 'ae' or 'vae'
                      latent_dim: int = 128,
                      epochs: int = 100,
                      batch_size: int = 32,
                      learning_rate: float = 0.001,
                      device: str = 'cuda:0',
                      save_dir: str = 'runs/anomaly',
                      name: str = None) -> dict:
    """
    Train autoencoder for anomaly detection.
    
    Args:
        normal_images_dir: Directory with normal images for training
        val_images_dir: Directory with validation images
        model_type: 'ae' for Autoencoder, 'vae' for Variational Autoencoder
        latent_dim: Latent space dimension
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Training device
        save_dir: Save directory
        name: Run name
    
    Returns:
        Training results
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Setup run name
    if name is None:
        name = f"anomaly_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    save_path = Path(save_dir) / name
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training {model_type.upper()} anomaly detector")
    logger.info(f"Device: {device}")
    
    # Create datasets
    train_dataset = NormalImageDataset(normal_images_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    if val_images_dir:
        val_dataset = NormalImageDataset(val_images_dir)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
    else:
        val_loader = None
    
    # Create model
    if model_type == 'vae':
        model = VariationalAutoencoder(
            latent_dim=latent_dim,
            input_size=224
        ).to(device)
    else:
        model = AnomalyAutoencoder(
            latent_dim=latent_dim,
            input_size=224
        ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in pbar:
            images = batch['image'].to(device)
            
            optimizer.zero_grad()
            
            if model_type == 'vae':
                reconstruction, mu, logvar = model(images)
                
                # VAE loss = reconstruction + KL divergence
                recon_loss = F.mse_loss(reconstruction, images, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = (recon_loss + 0.1 * kl_loss) / images.size(0)
            else:
                reconstruction, _ = model(images)
                loss = F.mse_loss(reconstruction, images)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    
                    if model_type == 'vae':
                        reconstruction, mu, logvar = model(images)
                        recon_loss = F.mse_loss(reconstruction, images, reduction='sum')
                        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        loss = (recon_loss + 0.1 * kl_loss) / images.size(0)
                    else:
                        reconstruction, _ = model(images)
                        loss = F.mse_loss(reconstruction, images)
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)
            scheduler.step(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'model_type': model_type,
                    'latent_dim': latent_dim
                }, save_path / 'best_autoencoder.pt')
        else:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'latent_dim': latent_dim,
        'history': history
    }, save_path / 'final_autoencoder.pt')
    
    # Compute baseline statistics on training set
    logger.info("Computing baseline statistics...")
    model.eval()
    errors = []
    
    with torch.no_grad():
        for batch in train_loader:
            images = batch['image'].to(device)
            error = model.compute_reconstruction_error(images)
            errors.extend(error.cpu().numpy())
    
    baseline_stats = {
        'mean': float(np.mean(errors)),
        'std': float(np.std(errors)),
        'min': float(np.min(errors)),
        'max': float(np.max(errors)),
        'threshold_2std': float(np.mean(errors) + 2 * np.std(errors)),
        'threshold_3std': float(np.mean(errors) + 3 * np.std(errors))
    }
    
    # Save baseline stats
    import json
    with open(save_path / 'baseline_stats.json', 'w') as f:
        json.dump(baseline_stats, f, indent=2)
    
    logger.info(f"Baseline mean error: {baseline_stats['mean']:.6f}")
    logger.info(f"Baseline std error: {baseline_stats['std']:.6f}")
    logger.info(f"Anomaly threshold (2σ): {baseline_stats['threshold_2std']:.6f}")
    
    return {
        'best_model_path': str(save_path / 'best_autoencoder.pt'),
        'final_model_path': str(save_path / 'final_autoencoder.pt'),
        'baseline_stats': baseline_stats,
        'history': history
    }


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train Anomaly Detector')
    parser.add_argument('--data', type=str, required=True, help='Normal images directory')
    parser.add_argument('--val_data', type=str, default=None, help='Validation images directory')
    parser.add_argument('--model_type', type=str, default='ae', choices=['ae', 'vae'])
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_dir', type=str, default='runs/anomaly')
    parser.add_argument('--name', type=str, default=None)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    results = train_autoencoder(
        normal_images_dir=args.data,
        val_images_dir=args.val_data,
        model_type=args.model_type,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        name=args.name
    )
    
    print(f"\nTraining Complete!")
    print(f"Best Model: {results['best_model_path']}")
    print(f"Anomaly Threshold (2σ): {results['baseline_stats']['threshold_2std']:.6f}")


if __name__ == '__main__':
    main()
