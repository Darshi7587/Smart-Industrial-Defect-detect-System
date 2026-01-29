# Smart Industrial Defect Detection - Evaluation Metrics
"""
Comprehensive metrics for defect detection and classification.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, precision_recall_curve
)
import logging

logger = logging.getLogger(__name__)


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-8)


def compute_detection_metrics(predictions: List[Dict],
                             ground_truths: List[Dict],
                             iou_threshold: float = 0.5,
                             class_names: List[str] = None) -> Dict:
    """
    Compute detection metrics (precision, recall, mAP).
    
    Args:
        predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
        ground_truths: List of ground truth dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for matching
        class_names: List of class names
    
    Returns:
        Dictionary with detection metrics
    """
    all_detections = []
    all_gt = []
    
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred.get('boxes', [])
        pred_scores = pred.get('scores', [])
        pred_labels = pred.get('labels', [])
        
        gt_boxes = gt.get('boxes', [])
        gt_labels = gt.get('labels', [])
        
        # Match predictions to ground truths
        gt_matched = [False] * len(gt_boxes)
        
        for i, (pred_box, pred_score, pred_label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
            best_iou = 0
            best_gt_idx = -1
            
            for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_matched[j]:
                    continue
                if pred_label != gt_label:
                    continue
                
                iou = compute_iou(np.array(pred_box), np.array(gt_box))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold:
                all_detections.append({
                    'score': pred_score,
                    'label': pred_label,
                    'tp': True
                })
                gt_matched[best_gt_idx] = True
            else:
                all_detections.append({
                    'score': pred_score,
                    'label': pred_label,
                    'tp': False
                })
        
        # Count unmatched ground truths (false negatives)
        for j, matched in enumerate(gt_matched):
            if not matched:
                all_gt.append({'label': gt_labels[j]})
    
    # Calculate metrics
    if not all_detections:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'mAP50': 0.0,
            'f1_score': 0.0
        }
    
    # Sort by score
    all_detections = sorted(all_detections, key=lambda x: x['score'], reverse=True)
    
    tp = sum(1 for d in all_detections if d['tp'])
    fp = sum(1 for d in all_detections if not d['tp'])
    fn = len(all_gt)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Compute AP (simplified)
    cumsum_tp = np.cumsum([1 if d['tp'] else 0 for d in all_detections])
    cumsum_fp = np.cumsum([0 if d['tp'] else 1 for d in all_detections])
    
    precisions = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-8)
    recalls = cumsum_tp / (tp + fn + 1e-8)
    
    # Interpolated AP
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        prec_at_rec = [p for p, r in zip(precisions, recalls) if r >= t]
        if prec_at_rec:
            ap += max(prec_at_rec) / 11
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'mAP50': float(ap),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }


def compute_classification_metrics(predictions: np.ndarray,
                                   labels: np.ndarray,
                                   probabilities: np.ndarray = None,
                                   class_names: List[str] = None) -> Dict:
    """
    Compute classification metrics.
    
    Args:
        predictions: Predicted class indices
        labels: True class indices
        probabilities: Class probabilities (for AUC)
        class_names: List of class names
    
    Returns:
        Dictionary with classification metrics
    """
    if class_names is None:
        class_names = [f'class_{i}' for i in range(len(np.unique(labels)))]
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    
    precision_weighted = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall_weighted = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(labels, predictions, average=None, zero_division=0)
    recall_per_class = recall_score(labels, predictions, average=None, zero_division=0)
    f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # AUC (if probabilities provided)
    auc_scores = {}
    if probabilities is not None and len(np.unique(labels)) > 1:
        try:
            if len(np.unique(labels)) == 2:
                auc_scores['roc_auc'] = roc_auc_score(labels, probabilities[:, 1])
            else:
                auc_scores['roc_auc_ovr'] = roc_auc_score(
                    labels, probabilities, multi_class='ovr', average='macro'
                )
        except:
            pass
    
    # Per-class report
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            per_class_metrics[class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i])
            }
    
    return {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'confusion_matrix': cm.tolist(),
        'per_class': per_class_metrics,
        **auc_scores
    }


def compute_anomaly_metrics(anomaly_scores: np.ndarray,
                           labels: np.ndarray,
                           threshold: float = None) -> Dict:
    """
    Compute anomaly detection metrics.
    
    Args:
        anomaly_scores: Anomaly scores (higher = more anomalous)
        labels: True labels (1 = anomaly, 0 = normal)
        threshold: Decision threshold (if None, uses optimal)
    
    Returns:
        Dictionary with anomaly detection metrics
    """
    # Find optimal threshold if not provided
    if threshold is None:
        precisions, recalls, thresholds = precision_recall_curve(labels, anomaly_scores)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        threshold = thresholds[np.argmax(f1_scores[:-1])]
    
    # Binary predictions
    predictions = (anomaly_scores >= threshold).astype(int)
    
    # Metrics
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    # AUC
    try:
        roc_auc = roc_auc_score(labels, anomaly_scores)
        pr_auc = average_precision_score(labels, anomaly_scores)
    except:
        roc_auc = 0.0
        pr_auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    return {
        'threshold': float(threshold),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'false_positive_rate': float(fp / (fp + tn + 1e-8)),
        'false_negative_rate': float(fn / (fn + tp + 1e-8))
    }


class MetricsTracker:
    """
    Track metrics during training.
    """
    
    def __init__(self, metric_names: List[str] = None):
        """Initialize tracker."""
        self.metric_names = metric_names or ['loss', 'accuracy', 'f1']
        self.history = {name: [] for name in self.metric_names}
        self.best_values = {}
    
    def update(self, metrics: Dict):
        """Update with new metrics."""
        for name, value in metrics.items():
            if name in self.history:
                self.history[name].append(value)
                
                # Update best
                if name not in self.best_values:
                    self.best_values[name] = value
                elif 'loss' in name.lower():
                    self.best_values[name] = min(self.best_values[name], value)
                else:
                    self.best_values[name] = max(self.best_values[name], value)
    
    def get_latest(self) -> Dict:
        """Get latest metrics."""
        return {name: values[-1] if values else None 
                for name, values in self.history.items()}
    
    def get_best(self) -> Dict:
        """Get best metrics."""
        return self.best_values.copy()
    
    def get_history(self) -> Dict:
        """Get full history."""
        return self.history.copy()
