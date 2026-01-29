# Smart Industrial Defect Detection - YOLOv8 Training Script
"""
Training script for YOLOv8 defect detection.
"""

import os
import argparse
import yaml
from pathlib import Path
from datetime import datetime
import logging

from ultralytics import YOLO

logger = logging.getLogger(__name__)


def train_yolov8_detector(data_yaml: str,
                          model_size: str = 'n',
                          epochs: int = 100,
                          batch_size: int = 16,
                          image_size: int = 640,
                          pretrained: bool = True,
                          resume: bool = False,
                          device: str = 'cuda:0',
                          project: str = 'runs/detect',
                          name: str = None,
                          **kwargs) -> dict:
    """
    Train YOLOv8 detection model.
    
    Args:
        data_yaml: Path to dataset YAML configuration
        model_size: Model size (n, s, m, l, x)
        epochs: Number of training epochs
        batch_size: Batch size
        image_size: Input image size
        pretrained: Use pretrained weights
        resume: Resume from checkpoint
        device: Training device
        project: Project directory
        name: Run name
        **kwargs: Additional training arguments
    
    Returns:
        Training results
    """
    # Setup run name
    if name is None:
        name = f"defect_yolov8{model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Load model
    model = YOLO(f'yolov8{model_size}.pt')
    
    logger.info(f"Starting YOLOv8{model_size} training")
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {image_size}")
    
    # Training configuration
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': image_size,
        'pretrained': pretrained,
        'device': device,
        'project': project,
        'name': name,
        'resume': resume,
        
        # Optimization
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        
        # Augmentation (for defect detection)
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 15.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 5.0,
        'perspective': 0.0,
        'flipud': 0.1,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,
        
        # Loss weights
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # Regularization
        'label_smoothing': 0.0,
        'dropout': 0.0,
        
        # Early stopping
        'patience': 50,
        
        # Saving
        'save': True,
        'save_period': 10,
        
        # Validation
        'val': True,
        
        # Logging
        'verbose': True,
        'plots': True,
    }
    
    # Override with custom arguments
    train_args.update(kwargs)
    
    # Train
    results = model.train(**train_args)
    
    # Log results
    logger.info(f"Training completed!")
    logger.info(f"Best mAP50: {results.box.map50:.4f}")
    logger.info(f"Best mAP50-95: {results.box.map:.4f}")
    logger.info(f"Model saved to: {project}/{name}")
    
    return {
        'best_model_path': str(Path(project) / name / 'weights' / 'best.pt'),
        'last_model_path': str(Path(project) / name / 'weights' / 'last.pt'),
        'results_path': str(Path(project) / name),
        'mAP50': float(results.box.map50),
        'mAP50_95': float(results.box.map),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr)
    }


def validate_yolov8(model_path: str,
                    data_yaml: str,
                    split: str = 'val',
                    batch_size: int = 16,
                    image_size: int = 640,
                    device: str = 'cuda:0') -> dict:
    """
    Validate YOLOv8 model.
    
    Args:
        model_path: Path to model weights
        data_yaml: Path to dataset YAML
        split: Dataset split ('val' or 'test')
        batch_size: Batch size
        image_size: Image size
        device: Device
    
    Returns:
        Validation metrics
    """
    model = YOLO(model_path)
    
    results = model.val(
        data=data_yaml,
        split=split,
        batch=batch_size,
        imgsz=image_size,
        device=device,
        verbose=True
    )
    
    metrics = {
        'mAP50': float(results.box.map50),
        'mAP50_95': float(results.box.map),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr)
    }
    
    # Per-class metrics
    if hasattr(results.box, 'ap50'):
        metrics['per_class_AP50'] = results.box.ap50.tolist()
    
    logger.info(f"Validation Results: mAP50={metrics['mAP50']:.4f}, "
                f"mAP50-95={metrics['mAP50_95']:.4f}")
    
    return metrics


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train YOLOv8 Defect Detector')
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML path')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--project', type=str, default='runs/detect')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Train
    results = train_yolov8_detector(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume
    )
    
    print(f"\nTraining Complete!")
    print(f"Best Model: {results['best_model_path']}")
    print(f"mAP50: {results['mAP50']:.4f}")


if __name__ == '__main__':
    main()
