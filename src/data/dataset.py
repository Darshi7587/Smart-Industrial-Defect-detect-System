# Smart Industrial Defect Detection - Data Pipeline
"""
Data loading, preprocessing, and augmentation pipeline.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logger = logging.getLogger(__name__)


class DefectDetectionDataset(Dataset):
    """
    PyTorch Dataset for defect detection (YOLOv8).
    Loads images and YOLO format labels (normalized bbox coordinates).
    """
    
    def __init__(self, 
                 images_dir: str,
                 labels_dir: str,
                 image_size: int = 640,
                 augment: bool = False,
                 class_names: List[str] = None):
        """
        Initialize dataset.
        
        Args:
            images_dir: Path to images directory
            labels_dir: Path to YOLO labels directory
            image_size: Target image size
            augment: Enable augmentation
            class_names: List of class names
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_size = image_size
        self.augment = augment
        self.class_names = class_names or []
        
        # Get image files
        self.image_files = sorted([
            f for f in self.images_dir.glob('*') 
            if f.suffix.lower() in ['.jpg', '.png', '.jpeg']
        ])
        
        logger.info(f"Loaded {len(self.image_files)} images from {images_dir}")
        
        # Setup augmentation
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.Blur(blur_limit=3, p=0.2),
                A.CoarseDropout(max_holes=5, max_height=20, max_width=20, p=0.3),
                A.Affine(scale=(0.8, 1.2), p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get single sample."""
        image_path = self.image_files[idx]
        label_path = self.labels_dir / (image_path.stem + '.txt')
        
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Load labels (YOLO format: normalized coordinates)
        bboxes = []
        class_labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        bbox_w = float(parts[3])
                        bbox_h = float(parts[4])
                        
                        # Convert from YOLO (normalized) to Pascal VOC format
                        x1 = max(0, (x_center - bbox_w/2) * w)
                        y1 = max(0, (y_center - bbox_h/2) * h)
                        x2 = min(w, (x_center + bbox_w/2) * w)
                        y2 = min(h, (y_center + bbox_h/2) * h)
                        
                        bboxes.append([x1, y1, x2, y2])
                        class_labels.append(class_id)
        
        # Resize image
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Apply augmentation and convert to tensor
        if self.augment and bboxes:
            augmented = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image = augmented['image']
            bboxes = augmented['bboxes']
            class_labels = augmented['class_labels']
        else:
            augmented = self.transform(image=image, bboxes=[], class_labels=[])
            image = augmented['image']
        
        # Convert bboxes back to YOLO format (normalized)
        yolo_bboxes = []
        for bbox, class_id in zip(bboxes, class_labels):
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                x_center = (x1 + x2) / (2 * self.image_size)
                y_center = (y1 + y2) / (2 * self.image_size)
                bbox_w = (x2 - x1) / self.image_size
                bbox_h = (y2 - y1) / self.image_size
                yolo_bboxes.append([class_id, x_center, y_center, bbox_w, bbox_h])
        
        # Convert to tensor
        targets = torch.zeros((len(yolo_bboxes), 5))
        if yolo_bboxes:
            targets = torch.tensor(yolo_bboxes, dtype=torch.float32)
        
        return {
            'image': image,
            'targets': targets,
            'image_path': str(image_path),
            'image_size': self.image_size
        }


class DefectClassificationDataset(Dataset):
    """
    PyTorch Dataset for defect classification (EfficientNet).
    Loads images from class subdirectories.
    """
    
    def __init__(self,
                 root_dir: str,
                 image_size: int = 224,
                 augment: bool = False,
                 class_names: List[str] = None):
        """
        Initialize dataset.
        
        Args:
            root_dir: Root directory with class subdirectories
            image_size: Target image size
            augment: Enable augmentation
            class_names: List of class names (in order)
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.augment = augment
        
        # Discover classes from subdirectories
        self.class_names = class_names or sorted([
            d.name for d in self.root_dir.iterdir() if d.is_dir()
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # Get all images with their labels
        self.images = []
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_file in class_dir.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                        self.images.append((str(img_file), self.class_to_idx[class_name]))
        
        logger.info(f"Loaded {len(self.images)} classification images from {root_dir}")
        
        # Setup augmentation
        if self.augment:
            self.transform = A.Compose([
                A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.Blur(blur_limit=3, p=0.2),
                A.Rotate(limit=10, p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get single sample."""
        image_path, class_idx = self.images[idx]
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation
        augmented = self.transform(image=image)
        image = augmented['image']
        
        return {
            'image': image,
            'label': torch.tensor(class_idx, dtype=torch.long),
            'image_path': image_path
        }


def create_dataloaders(config,
                      train_images_dir: str = None,
                      train_labels_dir: str = None,
                      val_images_dir: str = None,
                      val_labels_dir: str = None,
                      dataset_type: str = 'detection') -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders.
    
    Args:
        config: Configuration object
        train_images_dir: Training images directory
        train_labels_dir: Training labels directory
        val_images_dir: Validation images directory
        val_labels_dir: Validation labels directory
        dataset_type: 'detection' or 'classification'
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if dataset_type == 'detection':
        train_dataset = DefectDetectionDataset(
            images_dir=train_images_dir or config.data.train_dir,
            labels_dir=train_labels_dir or config.data.train_dir.replace('images', 'labels'),
            image_size=config.model.input_size,
            augment=True,
            class_names=config.data.defect_classes
        )
        
        val_dataset = DefectDetectionDataset(
            images_dir=val_images_dir or config.data.val_dir,
            labels_dir=val_labels_dir or config.data.val_dir.replace('images', 'labels'),
            image_size=config.model.input_size,
            augment=False,
            class_names=config.data.defect_classes
        )
    
    elif dataset_type == 'classification':
        train_dataset = DefectClassificationDataset(
            root_dir=train_images_dir or config.data.train_dir,
            image_size=224,
            augment=True,
            class_names=config.data.defect_classes
        )
        
        val_dataset = DefectClassificationDataset(
            root_dir=val_images_dir or config.data.val_dir,
            image_size=224,
            augment=False,
            class_names=config.data.defect_classes
        )
    
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    logger.info(f"Created DataLoaders - Train: {len(train_loader)} batches, "
                f"Val: {len(val_loader)} batches")
    
    return train_loader, val_loader
