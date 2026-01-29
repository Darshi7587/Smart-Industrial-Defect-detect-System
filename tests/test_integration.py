# Smart Industrial Defect Detection - Integration Tests
"""
Integration tests for the complete inference pipeline.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestInferencePipeline:
    """Test complete inference pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline can be initialized without models."""
        from src.inference.pipeline import InferencePipeline
        
        # Should initialize without errors
        pipeline = InferencePipeline(
            detection_model=None,
            classification_model=None,
            anomaly_model=None
        )
        
        assert pipeline is not None
    
    def test_preprocessing(self):
        """Test image preprocessing."""
        from src.inference.pipeline import InferencePipeline
        
        pipeline = InferencePipeline()
        
        # Create dummy image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Should handle preprocessing
        processed = pipeline.preprocess_image(image)
        
        assert processed.shape == (640, 640, 3)


class TestFactoryIntegration:
    """Test factory integration components."""
    
    def test_decision_engine_integration(self):
        """Test decision engine with factory components."""
        from src.factory_integration.decision_engine import DecisionEngine, Decision
        from src.factory_integration.plc_controller import PLCController
        
        engine = DecisionEngine()
        plc = PLCController(simulate=True)
        
        # Test accept path
        result = engine.make_decision(defects=[], anomaly_score=0.5)
        
        assert result.decision == Decision.ACCEPT
        
        # Should be able to send signal
        success = plc.accept_product()
        assert success is True
    
    def test_database_logging(self):
        """Test database logging integration."""
        from src.factory_integration.database import DatabaseLogger
        
        logger = DatabaseLogger(db_type='sqlite', db_path=':memory:')
        
        # Log detection
        detection = {
            'image_id': 'test_001',
            'defects': [{'class': 'scratch', 'confidence': 0.85}],
            'decision': 'REJECT',
            'latency_ms': 32.5
        }
        
        result = logger.log_detection(detection)
        
        assert result is not None


class TestEnsemble:
    """Test ensemble model."""
    
    def test_ensemble_initialization(self):
        """Test ensemble can be initialized."""
        from src.models.ensemble import EnsembleDefectDetector
        
        ensemble = EnsembleDefectDetector()
        
        assert ensemble.weights['detection'] == 0.5
        assert ensemble.weights['classification'] == 0.3
        assert ensemble.weights['anomaly'] == 0.2
    
    def test_ensemble_prediction_empty(self):
        """Test ensemble prediction without models."""
        from src.models.ensemble import EnsembleDefectDetector
        
        ensemble = EnsembleDefectDetector()
        
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        result = ensemble.predict(image)
        
        assert 'detections' in result
        assert 'decision' in result
        assert result['decision'] == 'ACCEPT'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
