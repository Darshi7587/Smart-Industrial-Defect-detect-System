# Smart Industrial Defect Detection - YOLOv8 Wrapper
"""
YOLOv8 model wrapper for defect detection.
Provides unified interface for training, inference, and export.
"""

import os
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)


class YOLOv8Detector:
    """
    YOLOv8 detector wrapper for industrial defect detection.
    Supports training, inference, and model export.
    """
    
    def __init__(self, 
                 model_path: str = None,
                 model_size: str = 'n',  # n, s, m, l, x
                 device: str = 'cuda:0',
                 num_classes: int = 5):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_path: Path to custom trained model (optional)
            model_size: YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)
            device: Device to run on
            num_classes: Number of defect classes
        """
        self.device = device
        self.num_classes = num_classes
        self.model_size = model_size
        
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            logger.info(f"Loaded custom YOLOv8 model from {model_path}")
        else:
            # Load pretrained model
            model_name = f'yolov8{model_size}.pt'
            self.model = YOLO(model_name)
            logger.info(f"Loaded pretrained YOLOv8{model_size} model")
        
        # Move to device
        self.model.to(device)
        
        # Default class names
        self.class_names = [
            'scratch',
            'crack', 
            'dent',
            'missing_component',
            'contamination'
        ]
    
    def train(self,
              data_yaml: str,
              epochs: int = 100,
              batch_size: int = 16,
              image_size: int = 640,
              pretrained: bool = True,
              optimizer: str = 'AdamW',
              lr0: float = 0.01,
              lrf: float = 0.01,
              momentum: float = 0.937,
              weight_decay: float = 0.0005,
              warmup_epochs: int = 3,
              save_dir: str = 'runs/detect',
              project_name: str = 'defect_detection',
              resume: bool = False,
              patience: int = 50,
              **kwargs) -> Dict:
        """
        Train YOLOv8 model.
        
        Args:
            data_yaml: Path to dataset YAML configuration
            epochs: Number of training epochs
            batch_size: Batch size
            image_size: Input image size
            pretrained: Use pretrained weights
            optimizer: Optimizer (SGD, Adam, AdamW)
            lr0: Initial learning rate
            lrf: Final learning rate factor
            momentum: SGD momentum
            weight_decay: Weight decay
            warmup_epochs: Warmup epochs
            save_dir: Save directory
            project_name: Project name for logging
            resume: Resume from checkpoint
            patience: Early stopping patience
            **kwargs: Additional YOLO training arguments
        
        Returns:
            Training results dictionary
        """
        logger.info(f"Starting YOLOv8 training with {epochs} epochs")
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            pretrained=pretrained,
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            project=save_dir,
            name=project_name,
            resume=resume,
            patience=patience,
            device=self.device,
            verbose=True,
            **kwargs
        )
        
        logger.info(f"Training completed. Results saved to {save_dir}/{project_name}")
        return results
    
    def predict(self,
                image: Union[np.ndarray, str, List],
                conf_threshold: float = 0.5,
                iou_threshold: float = 0.45,
                max_detections: int = 100,
                classes: List[int] = None,
                verbose: bool = False) -> Dict:
        """
        Perform inference on image(s).
        
        Args:
            image: Input image(s) - numpy array, file path, or list
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
            max_detections: Maximum number of detections
            classes: Filter by class indices
            verbose: Print verbose output
        
        Returns:
            Dictionary with detection results
        """
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=max_detections,
            classes=classes,
            device=self.device,
            verbose=verbose
        )
        
        # Process results
        detections = []
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # [N, 4]
                confidences = result.boxes.conf.cpu().numpy()  # [N]
                class_ids = result.boxes.cls.cpu().numpy().astype(int)  # [N]
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    detection = {
                        'bbox': box.tolist(),  # [x1, y1, x2, y2]
                        'confidence': float(conf),
                        'class_id': int(cls_id),
                        'class_name': self.class_names[cls_id] if cls_id < len(self.class_names) else f'class_{cls_id}'
                    }
                    detections.append(detection)
        
        return {
            'detections': detections,
            'num_detections': len(detections),
            'image_shape': results[0].orig_shape if results else None
        }
    
    def predict_batch(self,
                      images: List[np.ndarray],
                      conf_threshold: float = 0.5,
                      iou_threshold: float = 0.45) -> List[Dict]:
        """
        Batch inference on multiple images.
        
        Args:
            images: List of images
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
        
        Returns:
            List of detection results
        """
        all_results = []
        
        # Process in batches
        batch_results = self.model.predict(
            source=images,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )
        
        for result in batch_results:
            detections = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    detections.append({
                        'bbox': box.tolist(),
                        'confidence': float(conf),
                        'class_id': int(cls_id),
                        'class_name': self.class_names[cls_id] if cls_id < len(self.class_names) else f'class_{cls_id}'
                    })
            
            all_results.append({
                'detections': detections,
                'num_detections': len(detections)
            })
        
        return all_results
    
    def validate(self,
                 data_yaml: str,
                 split: str = 'val',
                 batch_size: int = 16,
                 image_size: int = 640,
                 conf_threshold: float = 0.001,
                 iou_threshold: float = 0.6) -> Dict:
        """
        Validate model on dataset.
        
        Args:
            data_yaml: Path to dataset YAML
            split: Dataset split ('val' or 'test')
            batch_size: Batch size
            image_size: Image size
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
        
        Returns:
            Validation metrics dictionary
        """
        results = self.model.val(
            data=data_yaml,
            split=split,
            batch=batch_size,
            imgsz=image_size,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=True
        )
        
        metrics = {
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'per_class_AP50': results.box.ap50.tolist() if hasattr(results.box, 'ap50') else None
        }
        
        logger.info(f"Validation results: mAP50={metrics['mAP50']:.4f}, mAP50-95={metrics['mAP50-95']:.4f}")
        return metrics
    
    def export(self,
               format: str = 'onnx',
               output_path: str = None,
               image_size: int = 640,
               half: bool = False,
               dynamic: bool = False,
               simplify: bool = True,
               opset: int = 12) -> str:
        """
        Export model to deployment format.
        
        Args:
            format: Export format (onnx, torchscript, tflite, engine, etc.)
            output_path: Output path
            image_size: Input image size
            half: FP16 quantization
            dynamic: Dynamic input shapes
            simplify: Simplify ONNX model
            opset: ONNX opset version
        
        Returns:
            Path to exported model
        """
        logger.info(f"Exporting model to {format} format")
        
        export_path = self.model.export(
            format=format,
            imgsz=image_size,
            half=half,
            dynamic=dynamic,
            simplify=simplify,
            opset=opset
        )
        
        if output_path and export_path:
            import shutil
            shutil.move(str(export_path), output_path)
            export_path = output_path
        
        logger.info(f"Model exported to {export_path}")
        return str(export_path)
    
    def visualize_predictions(self,
                              image: np.ndarray,
                              detections: Dict,
                              output_path: str = None,
                              show: bool = False) -> np.ndarray:
        """
        Visualize predictions on image.
        
        Args:
            image: Input image (BGR)
            detections: Detection results from predict()
            output_path: Path to save visualization
            show: Display image
        
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Color palette for different classes
        colors = [
            (0, 0, 255),    # Red - scratch
            (0, 165, 255),  # Orange - crack
            (0, 255, 255),  # Yellow - dent
            (255, 0, 0),    # Blue - missing_component
            (255, 0, 255)   # Magenta - contamination
        ]
        
        for det in detections['detections']:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            cls_id = det['class_id']
            conf = det['confidence']
            cls_name = det['class_name']
            
            color = colors[cls_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{cls_name}: {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if output_path:
            cv2.imwrite(output_path, annotated)
        
        if show:
            cv2.imshow('Defect Detection', annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return annotated
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'model_type': 'YOLOv8',
            'model_size': self.model_size,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'device': self.device,
            'input_size': 640
        }


def create_yolo_data_yaml(train_path: str,
                          val_path: str,
                          test_path: str = None,
                          class_names: List[str] = None,
                          output_path: str = 'data/defect_data.yaml') -> str:
    """
    Create YOLO dataset configuration YAML file.
    
    Args:
        train_path: Path to training images
        val_path: Path to validation images
        test_path: Path to test images (optional)
        class_names: List of class names
        output_path: Output YAML path
    
    Returns:
        Path to created YAML file
    """
    import yaml
    
    if class_names is None:
        class_names = ['scratch', 'crack', 'dent', 'missing_component', 'contamination']
    
    data = {
        'path': str(Path(train_path).parent.parent),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    if test_path:
        data['test'] = 'test/images'
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    logger.info(f"Created YOLO data config at {output_path}")
    return output_path
