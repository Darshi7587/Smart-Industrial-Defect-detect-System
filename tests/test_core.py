# Smart Industrial Defect Detection - Unit Tests
"""
Test suite for defect detection system.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataAugmentation:
    """Test data augmentation pipeline."""
    
    def test_training_augmentation(self):
        """Test training augmentation returns correct format."""
        from src.data.augmentation import get_training_augmentation
        
        transform = get_training_augmentation(image_size=640)
        
        # Create dummy image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bboxes = [[0.5, 0.5, 0.1, 0.1]]
        class_labels = [0]
        
        result = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        
        assert 'image' in result
        assert result['image'].shape == (3, 640, 640)
    
    def test_validation_augmentation(self):
        """Test validation augmentation (no augmentation)."""
        from src.data.augmentation import get_validation_augmentation
        
        transform = get_validation_augmentation(image_size=640)
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bboxes = [[0.5, 0.5, 0.1, 0.1]]
        class_labels = [0]
        
        result = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        
        assert result['image'].shape == (3, 640, 640)


class TestLossFunctions:
    """Test custom loss functions."""
    
    def test_focal_loss(self):
        """Test focal loss computation."""
        from src.training.losses import FocalLoss
        
        loss_fn = FocalLoss()
        
        inputs = torch.randn(8, 5)  # Batch of 8, 5 classes
        targets = torch.randint(0, 5, (8,))
        
        loss = loss_fn(inputs, targets)
        
        assert loss.shape == ()
        assert loss >= 0
    
    def test_label_smoothing_loss(self):
        """Test label smoothing loss."""
        from src.training.losses import LabelSmoothingLoss
        
        loss_fn = LabelSmoothingLoss(num_classes=5, smoothing=0.1)
        
        inputs = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        
        loss = loss_fn(inputs, targets)
        
        assert loss.shape == ()
        assert loss >= 0


class TestMetrics:
    """Test metric computation."""
    
    def test_classification_metrics(self):
        """Test classification metrics computation."""
        from src.training.metrics import compute_classification_metrics
        
        predictions = np.array([0, 1, 2, 0, 1, 2])
        labels = np.array([0, 1, 2, 0, 2, 1])
        
        metrics = compute_classification_metrics(predictions, labels)
        
        assert 'accuracy' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 'f1_macro' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_detection_metrics(self):
        """Test detection metrics computation."""
        from src.training.metrics import compute_detection_metrics
        
        predictions = [
            {'boxes': [[10, 10, 50, 50]], 'scores': [0.9], 'labels': [0]}
        ]
        ground_truths = [
            {'boxes': [[10, 10, 50, 50]], 'labels': [0]}
        ]
        
        metrics = compute_detection_metrics(predictions, ground_truths)
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics


class TestDecisionEngine:
    """Test decision engine."""
    
    def test_accept_no_defects(self):
        """Test accept decision when no defects."""
        from src.factory_integration.decision_engine import DecisionEngine, Decision
        
        engine = DecisionEngine()
        
        result = engine.make_decision(defects=[], anomaly_score=0.5)
        
        assert result.decision == Decision.ACCEPT
    
    def test_reject_high_confidence_defect(self):
        """Test reject decision for high confidence defect."""
        from src.factory_integration.decision_engine import DecisionEngine, Decision
        
        engine = DecisionEngine()
        
        defects = [{
            'detection_class': 'crack',
            'detection_confidence': 0.95,
            'classification': 'crack',
            'classification_confidence': 0.92
        }]
        
        result = engine.make_decision(defects=defects, max_confidence=0.95)
        
        assert result.decision == Decision.REJECT
    
    def test_alert_high_anomaly(self):
        """Test alert decision for high anomaly score."""
        from src.factory_integration.decision_engine import DecisionEngine, Decision
        
        engine = DecisionEngine(anomaly_threshold=2.0)
        
        result = engine.make_decision(defects=[], anomaly_score=3.5)
        
        assert result.decision == Decision.ALERT


class TestSyntheticDataGenerator:
    """Test synthetic data generation."""
    
    def test_generate_scratch(self):
        """Test scratch generation."""
        from src.data.synthetic_data import SyntheticDefectGenerator
        
        generator = SyntheticDefectGenerator()
        
        image = np.ones((640, 640, 3), dtype=np.uint8) * 200
        result_image, bboxes = generator.generate_scratch(image, num_scratches=2)
        
        assert result_image.shape == image.shape
        assert len(bboxes) == 2
        assert all(len(bbox) == 4 for bbox in bboxes)
    
    def test_generate_crack(self):
        """Test crack generation."""
        from src.data.synthetic_data import SyntheticDefectGenerator
        
        generator = SyntheticDefectGenerator()
        
        image = np.ones((640, 640, 3), dtype=np.uint8) * 200
        result_image, bboxes = generator.generate_crack(image, num_cracks=1)
        
        assert result_image.shape == image.shape
        assert len(bboxes) == 1


class TestConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration."""
        from src.config import get_config
        
        config = get_config()
        
        assert config.model.detection_threshold == 0.5
        assert config.training.epochs == 100
        assert config.data.image_size == 640
    
    def test_config_save_load(self, tmp_path):
        """Test config save and load."""
        from src.config import get_config, save_config
        
        config = get_config()
        config.model.detection_threshold = 0.7
        
        config_path = tmp_path / "test_config.yaml"
        save_config(config, str(config_path))
        
        loaded_config = get_config(str(config_path))
        
        assert loaded_config.model.detection_threshold == 0.7


class TestAlertManager:
    """Test alert manager."""
    
    def test_create_alert(self):
        """Test alert creation."""
        from src.factory_integration.alerts import AlertManager, AlertSeverity
        
        manager = AlertManager()
        
        alert = manager.create_alert(
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test",
            source="test"
        )
        
        assert alert.alert_id is not None
        assert alert.severity == AlertSeverity.WARNING
        assert len(manager.active_alerts) == 1
    
    def test_acknowledge_alert(self):
        """Test alert acknowledgment."""
        from src.factory_integration.alerts import AlertManager, AlertSeverity
        
        manager = AlertManager()
        
        alert = manager.create_alert(
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test message",
            source="test"
        )
        
        result = manager.acknowledge_alert(alert.alert_id, "operator1")
        
        assert result is True
        assert len(manager.active_alerts) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
