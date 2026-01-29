# Smart Industrial Defect Detection - Classification Training Script
"""
Training script for EfficientNet defect classifier.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.efficientnet_classifier import EfficientNetClassifier
from src.data.dataset import DefectClassificationDataset
from src.training.losses import FocalLoss, LabelSmoothingLoss
from src.training.metrics import compute_classification_metrics

logger = logging.getLogger(__name__)


def train_classifier(data_dir: str,
                     model_name: str = 'efficientnet_b4',
                     num_classes: int = 5,
                     epochs: int = 50,
                     batch_size: int = 32,
                     learning_rate: float = 0.001,
                     weight_decay: float = 0.0001,
                     class_weights: list = None,
                     use_focal_loss: bool = True,
                     use_label_smoothing: bool = False,
                     device: str = 'cuda:0',
                     save_dir: str = 'runs/classifier',
                     name: str = None) -> dict:
    """
    Train EfficientNet classifier.
    
    Args:
        data_dir: Root directory with train/val subdirectories
        model_name: EfficientNet variant
        num_classes: Number of classes
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        class_weights: Class weights for imbalanced data
        use_focal_loss: Use focal loss
        use_label_smoothing: Use label smoothing
        device: Training device
        save_dir: Save directory
        name: Run name
    
    Returns:
        Training results
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Setup run name
    if name is None:
        name = f"classifier_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    save_path = Path(save_dir) / name
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training {model_name} classifier")
    logger.info(f"Device: {device}")
    logger.info(f"Save path: {save_path}")
    
    # Create datasets
    train_dataset = DefectClassificationDataset(
        root_dir=os.path.join(data_dir, 'train'),
        image_size=224,
        augment=True
    )
    
    val_dataset = DefectClassificationDataset(
        root_dir=os.path.join(data_dir, 'val'),
        image_size=224,
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    model = EfficientNetClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True,
        dropout_rate=0.4,
        use_attention=True
    ).to(device)
    
    # Loss function
    if class_weights:
        weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    else:
        weights = None
    
    if use_focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    elif use_label_smoothing:
        criterion = LabelSmoothingLoss(num_classes=num_classes, smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Training loop
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Compute metrics
        import numpy as np
        metrics = compute_classification_metrics(
            np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs),
            class_names=train_dataset.class_names
        )
        
        val_acc = metrics['accuracy'] * 100
        val_f1 = metrics['f1_macro']
        
        scheduler.step()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}"
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'history': history,
                'class_names': train_dataset.class_names
            }, save_path / 'best_classifier.pt')
            logger.info(f"Saved best model with val_acc={val_acc:.2f}%")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'history': history,
        'class_names': train_dataset.class_names
    }, save_path / 'final_classifier.pt')
    
    return {
        'best_model_path': str(save_path / 'best_classifier.pt'),
        'final_model_path': str(save_path / 'final_classifier.pt'),
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1,
        'history': history
    }


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train Defect Classifier')
    parser.add_argument('--data', type=str, required=True, help='Data directory')
    parser.add_argument('--model', type=str, default='efficientnet_b4')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_dir', type=str, default='runs/classifier')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--focal_loss', action='store_true')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    results = train_classifier(
        data_dir=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        name=args.name,
        use_focal_loss=args.focal_loss
    )
    
    print(f"\nTraining Complete!")
    print(f"Best Model: {results['best_model_path']}")
    print(f"Best Accuracy: {results['best_val_acc']:.2f}%")
    print(f"Best F1: {results['best_val_f1']:.4f}")


if __name__ == '__main__':
    main()
