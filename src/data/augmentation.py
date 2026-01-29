# Smart Industrial Defect Detection - Data Augmentation Pipeline
"""
Albumentations-based augmentation for defect detection.
Industry-standard augmentations for manufacturing images.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Dict, List, Optional


def get_training_augmentation(image_size: int = 640,
                               augmentation_strength: str = 'medium') -> A.Compose:
    """
    Get training augmentation pipeline.
    
    Args:
        image_size: Target image size
        augmentation_strength: 'light', 'medium', 'heavy'
    
    Returns:
        Albumentations Compose object
    """
    if augmentation_strength == 'light':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    elif augmentation_strength == 'medium':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.4
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
            ], p=0.2),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                fill_value=0,
                p=0.3
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    else:  # heavy
        return A.Compose([
            A.RandomResizedCrop(image_size, image_size, scale=(0.7, 1.0), p=0.5),
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=30, p=0.6, border_mode=cv2.BORDER_REFLECT_101),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(-0.1, 0.1),
                rotate=(-15, 15),
                shear=(-10, 10),
                p=0.4
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.3
            ),
            A.GaussNoise(var_limit=(10.0, 80.0), p=0.4),
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MedianBlur(blur_limit=5, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.1, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, p=0.5),
            ], p=0.2),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.CoarseDropout(
                max_holes=12,
                max_height=40,
                max_width=40,
                fill_value=0,
                p=0.4
            ),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_limit=(1, 3),
                shadow_dimension=5,
                p=0.2
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def get_validation_augmentation(image_size: int = 640) -> A.Compose:
    """
    Get validation augmentation (no augmentation, just preprocessing).
    
    Args:
        image_size: Target image size
    
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def get_inference_transforms(image_size: int = 640) -> A.Compose:
    """
    Get inference transforms (no bbox params needed).
    
    Args:
        image_size: Target image size
    
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


class DefectSpecificAugmentation:
    """
    Defect-specific augmentation strategies.
    Different defect types benefit from different augmentations.
    """
    
    @staticmethod
    def get_scratch_augmentation(image_size: int = 640) -> A.Compose:
        """Scratches: emphasize linear features, blur to simulate motion."""
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=45, p=0.6),  # Scratches can be at any angle
            A.MotionBlur(blur_limit=5, p=0.3),  # Simulate camera motion
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.25,  # Higher contrast to emphasize scratches
                p=0.5
            ),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    @staticmethod
    def get_crack_augmentation(image_size: int = 640) -> A.Compose:
        """Cracks: emphasize texture, add noise for realistic conditions."""
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=90, p=0.5),  # Cracks can be any orientation
            A.ElasticTransform(alpha=50, sigma=50 * 0.05, p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
            A.GaussNoise(var_limit=(10.0, 40.0), p=0.4),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.3,
                p=0.4
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    @staticmethod
    def get_dent_augmentation(image_size: int = 640) -> A.Compose:
        """Dents: emphasize depth/shadow, use lighting variations."""
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.4),
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.2,
                p=0.5
            ),
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_limit=(1, 4),
                shadow_dimension=5,
                p=0.4
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


class MixupAugmentation:
    """
    Mixup and CutMix augmentation for classification.
    Improves model calibration and generalization.
    """
    
    @staticmethod
    def mixup(image1: np.ndarray, label1: int,
              image2: np.ndarray, label2: int,
              alpha: float = 0.2) -> tuple:
        """
        Apply Mixup augmentation.
        
        Args:
            image1, image2: Input images
            label1, label2: Class labels
            alpha: Mixup interpolation strength
        
        Returns:
            Mixed image and interpolated label weights
        """
        lam = np.random.beta(alpha, alpha)
        mixed_image = lam * image1 + (1 - lam) * image2
        return mixed_image.astype(np.uint8), (label1, label2, lam)
    
    @staticmethod
    def cutmix(image1: np.ndarray, label1: int,
               image2: np.ndarray, label2: int,
               alpha: float = 1.0) -> tuple:
        """
        Apply CutMix augmentation.
        
        Args:
            image1, image2: Input images
            label1, label2: Class labels
            alpha: CutMix beta distribution parameter
        
        Returns:
            CutMixed image and label weights
        """
        lam = np.random.beta(alpha, alpha)
        H, W = image1.shape[:2]
        
        # Calculate cut size
        cut_ratio = np.sqrt(1 - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        
        # Random center point
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Calculate bounding box
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_image = image1.copy()
        mixed_image[bby1:bby2, bbx1:bbx2] = image2[bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual cut area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return mixed_image, (label1, label2, lam)


def create_mosaic_augmentation(images: List[np.ndarray],
                               bboxes_list: List[List],
                               labels_list: List[List],
                               output_size: int = 640) -> tuple:
    """
    Create mosaic augmentation (4 images combined).
    Used in YOLOv8 training.
    
    Args:
        images: List of 4 images
        bboxes_list: List of bounding boxes for each image
        labels_list: List of labels for each image
        output_size: Output image size
    
    Returns:
        Mosaic image, combined bboxes, combined labels
    """
    assert len(images) == 4, "Mosaic requires exactly 4 images"
    
    output_image = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    output_bboxes = []
    output_labels = []
    
    # Random center point
    cx = np.random.randint(output_size // 4, 3 * output_size // 4)
    cy = np.random.randint(output_size // 4, 3 * output_size // 4)
    
    # Quadrant positions: top-left, top-right, bottom-left, bottom-right
    positions = [
        (0, 0, cx, cy),
        (cx, 0, output_size, cy),
        (0, cy, cx, output_size),
        (cx, cy, output_size, output_size)
    ]
    
    for i, (img, bboxes, labels) in enumerate(zip(images, bboxes_list, labels_list)):
        x1, y1, x2, y2 = positions[i]
        h, w = y2 - y1, x2 - x1
        
        # Resize image to fit quadrant
        resized = cv2.resize(img, (w, h))
        output_image[y1:y2, x1:x2] = resized
        
        # Transform bounding boxes
        for bbox, label in zip(bboxes, labels):
            # Scale bbox to quadrant
            bx, by, bw, bh = bbox  # YOLO format
            new_bx = (bx * w + x1) / output_size
            new_by = (by * h + y1) / output_size
            new_bw = bw * w / output_size
            new_bh = bh * h / output_size
            
            output_bboxes.append([new_bx, new_by, new_bw, new_bh])
            output_labels.append(label)
    
    return output_image, output_bboxes, output_labels
