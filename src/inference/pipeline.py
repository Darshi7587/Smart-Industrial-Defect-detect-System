# Smart Industrial Defect Detection - Inference Pipeline
"""
Complete inference pipeline for real-time defect detection.
Combines detection, classification, and anomaly detection.
"""

import os
import time
import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Detection result for a single image."""
    image_id: str
    timestamp: float
    decision: str  # 'ACCEPT', 'REJECT', 'ALERT'
    defects: List[Dict]
    anomaly_score: float
    confidence: float
    latency_ms: float
    annotated_image: Optional[np.ndarray] = None


class InferencePipeline:
    """
    Complete inference pipeline for defect detection.
    Integrates YOLOv8 detection, EfficientNet classification, and anomaly detection.
    """
    
    def __init__(self,
                 detection_model_path: str = None,
                 classification_model_path: str = None,
                 anomaly_model_path: str = None,
                 config_path: str = None,
                 device: str = 'cuda:0'):
        """
        Initialize inference pipeline.
        
        Args:
            detection_model_path: Path to YOLOv8 model
            classification_model_path: Path to classifier model
            anomaly_model_path: Path to anomaly detector
            config_path: Path to configuration file
            device: Device to run on
        """
        self.device = device
        
        # Load configuration
        if config_path:
            from src.config import get_config
            self.config = get_config(config_path)
        else:
            self.config = None
        
        # Initialize models
        self._init_detection_model(detection_model_path)
        self._init_classification_model(classification_model_path)
        self._init_anomaly_model(anomaly_model_path)
        
        # Class names
        self.class_names = [
            'scratch',
            'crack',
            'dent',
            'missing_component',
            'contamination'
        ]
        
        # Decision thresholds
        self.detection_threshold = 0.5
        self.classification_threshold = 0.85
        self.anomaly_threshold = 2.0
        
        # Latency tracking
        self.latency_history = []
        
        logger.info("Inference pipeline initialized")
    
    def _init_detection_model(self, model_path: str):
        """Initialize detection model."""
        try:
            from src.models.yolov8_wrapper import YOLOv8Detector
            self.detector = YOLOv8Detector(
                model_path=model_path,
                device=self.device
            )
            self.detector_available = True
            logger.info("Detection model loaded")
        except Exception as e:
            logger.warning(f"Detection model not available: {e}")
            self.detector = None
            self.detector_available = False
    
    def _init_classification_model(self, model_path: str):
        """Initialize classification model."""
        try:
            from src.models.efficientnet_classifier import DefectClassificationPipeline
            self.classifier = DefectClassificationPipeline(
                model_path=model_path,
                device=self.device
            )
            self.classifier_available = True
            logger.info("Classification model loaded")
        except Exception as e:
            logger.warning(f"Classification model not available: {e}")
            self.classifier = None
            self.classifier_available = False
    
    def _init_anomaly_model(self, model_path: str):
        """Initialize anomaly detection model."""
        try:
            from src.models.anomaly_detector import AnomalyDetectionPipeline
            self.anomaly_detector = AnomalyDetectionPipeline(
                autoencoder_path=model_path,
                device=self.device
            )
            self.anomaly_available = True
            logger.info("Anomaly detection model loaded")
        except Exception as e:
            logger.warning(f"Anomaly model not available: {e}")
            self.anomaly_detector = None
            self.anomaly_available = False
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (BGR)
        
        Returns:
            Preprocessed image
        """
        # Ensure correct format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Resize if needed
        if image.shape[0] != 640 or image.shape[1] != 640:
            image = cv2.resize(image, (640, 640))
        
        return image
    
    def predict(self, 
                image: np.ndarray,
                image_id: str = None,
                return_annotated: bool = True) -> DetectionResult:
        """
        Run complete inference pipeline.
        
        Args:
            image: Input image (BGR)
            image_id: Optional image identifier
            return_annotated: Return annotated image
        
        Returns:
            DetectionResult with all detections and decision
        """
        start_time = time.time()
        
        if image_id is None:
            image_id = f"img_{int(time.time() * 1000)}"
        
        # Preprocess
        processed_image = self.preprocess(image)
        
        defects = []
        anomaly_score = 0.0
        max_confidence = 0.0
        
        # Step 1: Object Detection
        if self.detector_available:
            detection_result = self.detector.predict(
                processed_image,
                conf_threshold=self.detection_threshold
            )
            
            for det in detection_result['detections']:
                defect = {
                    'bbox': det['bbox'],
                    'detection_confidence': det['confidence'],
                    'detection_class': det['class_name'],
                    'classification': None,
                    'classification_confidence': 0.0
                }
                
                # Step 2: Classification for each detection
                if self.classifier_available:
                    classification = self.classifier.classify_roi(
                        processed_image,
                        det['bbox']
                    )
                    defect['classification'] = classification['class_name']
                    defect['classification_confidence'] = classification['confidence']
                
                defects.append(defect)
                max_confidence = max(max_confidence, det['confidence'])
        
        # Step 3: Anomaly Detection
        if self.anomaly_available:
            anomaly_result = self.anomaly_detector.detect_anomaly(processed_image)
            anomaly_score = anomaly_result.get('z_score', 0.0)
        
        # Step 4: Decision Logic
        decision = self._make_decision(defects, anomaly_score, max_confidence)
        
        # Calculate overall confidence
        if defects:
            confidence = np.mean([d['detection_confidence'] for d in defects])
        else:
            confidence = 1.0 - min(1.0, anomaly_score / self.anomaly_threshold)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        self.latency_history.append(latency_ms)
        
        # Create annotated image
        annotated_image = None
        if return_annotated:
            annotated_image = self._annotate_image(
                processed_image.copy(),
                defects,
                decision,
                anomaly_score
            )
        
        return DetectionResult(
            image_id=image_id,
            timestamp=time.time(),
            decision=decision,
            defects=defects,
            anomaly_score=anomaly_score,
            confidence=confidence,
            latency_ms=latency_ms,
            annotated_image=annotated_image
        )
    
    def _make_decision(self,
                      defects: List[Dict],
                      anomaly_score: float,
                      max_confidence: float) -> str:
        """
        Make accept/reject/alert decision.
        
        Args:
            defects: List of detected defects
            anomaly_score: Anomaly detection score
            max_confidence: Maximum detection confidence
        
        Returns:
            Decision string: 'ACCEPT', 'REJECT', or 'ALERT'
        """
        # No defects detected
        if not defects:
            # Check anomaly score
            if anomaly_score > self.anomaly_threshold:
                return 'ALERT'  # Unknown anomaly detected
            return 'ACCEPT'
        
        # Defects detected
        if max_confidence > self.detection_threshold:
            # High confidence defect
            if any(d['detection_confidence'] > 0.8 for d in defects):
                return 'REJECT'
            
            # Medium confidence - check classification
            if self.classifier_available:
                if any(d['classification_confidence'] > self.classification_threshold for d in defects):
                    return 'REJECT'
            
            # Lower confidence - alert for review
            return 'ALERT'
        
        # Low confidence detections
        return 'ACCEPT'
    
    def _annotate_image(self,
                       image: np.ndarray,
                       defects: List[Dict],
                       decision: str,
                       anomaly_score: float) -> np.ndarray:
        """
        Annotate image with detections and decision.
        
        Args:
            image: Input image
            defects: List of defects
            decision: Decision string
            anomaly_score: Anomaly score
        
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Color scheme
        colors = {
            'scratch': (0, 0, 255),      # Red
            'crack': (0, 165, 255),      # Orange
            'dent': (0, 255, 255),       # Yellow
            'missing_component': (255, 0, 0),  # Blue
            'contamination': (255, 0, 255)     # Magenta
        }
        
        decision_colors = {
            'ACCEPT': (0, 255, 0),   # Green
            'REJECT': (0, 0, 255),   # Red
            'ALERT': (0, 255, 255)   # Yellow
        }
        
        # Draw defects
        for defect in defects:
            bbox = defect['bbox']
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            class_name = defect.get('classification') or defect.get('detection_class', 'unknown')
            confidence = defect.get('classification_confidence') or defect.get('detection_confidence', 0)
            
            color = colors.get(class_name, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw decision banner
        h, w = annotated.shape[:2]
        banner_color = decision_colors.get(decision, (128, 128, 128))
        cv2.rectangle(annotated, (0, 0), (w, 50), banner_color, -1)
        
        decision_text = f"Decision: {decision}"
        if anomaly_score > 0:
            decision_text += f" | Anomaly: {anomaly_score:.2f}"
        
        cv2.putText(annotated, decision_text, (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return annotated
    
    def predict_batch(self, 
                      images: List[np.ndarray],
                      image_ids: List[str] = None) -> List[DetectionResult]:
        """
        Batch inference on multiple images.
        
        Args:
            images: List of images
            image_ids: Optional list of image identifiers
        
        Returns:
            List of DetectionResults
        """
        if image_ids is None:
            image_ids = [f"img_{i}" for i in range(len(images))]
        
        results = []
        for img, img_id in zip(images, image_ids):
            result = self.predict(img, image_id=img_id, return_annotated=False)
            results.append(result)
        
        return results
    
    def get_latency_stats(self) -> Dict:
        """Get latency statistics."""
        if not self.latency_history:
            return {}
        
        latencies = np.array(self.latency_history[-100:])  # Last 100
        
        return {
            'mean_ms': float(np.mean(latencies)),
            'std_ms': float(np.std(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99))
        }
    
    def warmup(self, num_iterations: int = 10):
        """
        Warmup models with dummy inputs.
        
        Args:
            num_iterations: Number of warmup iterations
        """
        logger.info(f"Warming up models with {num_iterations} iterations...")
        
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        for _ in range(num_iterations):
            self.predict(dummy_image, return_annotated=False)
        
        # Clear warmup latencies
        self.latency_history = []
        
        logger.info("Warmup complete")


class RealTimeInference:
    """
    Real-time inference from camera or video stream.
    """
    
    def __init__(self,
                 pipeline: InferencePipeline,
                 camera_source: int = 0,
                 frame_rate: int = 30):
        """
        Initialize real-time inference.
        
        Args:
            pipeline: Inference pipeline
            camera_source: Camera index or video path
            frame_rate: Target frame rate
        """
        self.pipeline = pipeline
        self.camera_source = camera_source
        self.frame_rate = frame_rate
        self.running = False
        
        # Statistics
        self.total_frames = 0
        self.defect_frames = 0
        self.decisions = {'ACCEPT': 0, 'REJECT': 0, 'ALERT': 0}
    
    def start(self, 
              callback=None,
              display: bool = True,
              save_defects: bool = True,
              save_dir: str = 'output/defects'):
        """
        Start real-time inference loop.
        
        Args:
            callback: Callback function for each result
            display: Display video with annotations
            save_defects: Save images with defects
            save_dir: Directory to save defect images
        """
        cap = cv2.VideoCapture(self.camera_source)
        
        if not cap.isOpened():
            logger.error(f"Cannot open camera source: {self.camera_source}")
            return
        
        if save_defects:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        self.running = True
        frame_delay = 1.0 / self.frame_rate
        
        logger.info(f"Starting real-time inference at {self.frame_rate} FPS")
        
        try:
            while self.running:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break
                
                # Run inference
                result = self.pipeline.predict(frame, return_annotated=display)
                
                # Update statistics
                self.total_frames += 1
                self.decisions[result.decision] += 1
                if result.defects:
                    self.defect_frames += 1
                
                # Callback
                if callback:
                    callback(result)
                
                # Save defect images
                if save_defects and result.decision in ['REJECT', 'ALERT']:
                    save_path = Path(save_dir) / f"{result.image_id}_{result.decision}.jpg"
                    cv2.imwrite(str(save_path), result.annotated_image or frame)
                
                # Display
                if display and result.annotated_image is not None:
                    # Add FPS counter
                    fps = 1000.0 / result.latency_ms
                    cv2.putText(result.annotated_image, f"FPS: {fps:.1f}",
                               (10, result.annotated_image.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imshow('Defect Detection', result.annotated_image)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Frame rate control
                elapsed = time.time() - start_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            self.running = False
        
        logger.info(f"Inference stopped. Total frames: {self.total_frames}")
        logger.info(f"Decisions: {self.decisions}")
    
    def stop(self):
        """Stop inference loop."""
        self.running = False
    
    def get_statistics(self) -> Dict:
        """Get inference statistics."""
        defect_rate = self.defect_frames / max(1, self.total_frames)
        
        return {
            'total_frames': self.total_frames,
            'defect_frames': self.defect_frames,
            'defect_rate': defect_rate,
            'decisions': self.decisions,
            'latency_stats': self.pipeline.get_latency_stats()
        }
