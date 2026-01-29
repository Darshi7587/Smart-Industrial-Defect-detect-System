# Smart Industrial Defect Detection - Configuration Module
"""
Configuration loader for the entire system.
Supports YAML-based configuration with environment variable overrides.
"""

import os
import yaml
from typing import Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Camera configuration."""
    camera_type: str = "gige"  # gige, usb, rtsp
    camera_ip: str = "192.168.1.100"
    camera_port: int = 3956
    frame_width: int = 1920
    frame_height: int = 1080
    fps: int = 30
    exposure_time: float = 20000  # microseconds
    gain: float = 0.0
    frame_buffer_size: int = 3


@dataclass
class ModelConfig:
    """Model configuration."""
    detection_model_path: str = "models/detection_best.pt"
    classification_model_path: str = "models/classifier_best.pt"
    anomaly_model_path: str = "models/anomaly_autoencoder.pt"
    
    # Detection settings
    detection_confidence_threshold: float = 0.6
    detection_nms_threshold: float = 0.45
    
    # Classification settings
    classification_confidence_threshold: float = 0.85
    
    # Anomaly detection settings
    anomaly_threshold: float = 2.0  # Standard deviations from mean
    anomaly_enable: bool = True
    
    # Input preprocessing
    input_size: int = 640
    normalize_mean: list = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: list = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class InferenceConfig:
    """Inference configuration."""
    batch_size: int = 1
    device: str = "cuda:0"  # cuda:0, cuda:1, cpu
    num_workers: int = 4
    mixed_precision: bool = True  # FP16
    tensorrt_optimization: bool = True
    max_latency_ms: float = 100.0
    enable_profiling: bool = False


@dataclass
class DecisionConfig:
    """Decision engine configuration."""
    accept_threshold: float = 0.95  # High confidence for accept
    reject_threshold: float = 0.6   # Moderate confidence for reject
    alert_threshold: float = 0.5    # Low confidence for alert
    use_ensemble: bool = True
    anomaly_weight: float = 0.3
    fallback_to_manual: bool = True
    manual_review_threshold: float = 0.50


@dataclass
class FactoryConfig:
    """Factory integration configuration."""
    plc_enabled: bool = True
    plc_ip: str = "192.168.1.50"
    plc_port: int = 502
    modbus_timeout: float = 5.0
    
    # Rejection mechanism
    reject_signal_address: int = 100
    reject_signal_duration_ms: int = 50
    conveyor_control_enabled: bool = True
    
    # Database
    db_type: str = "postgresql"  # postgresql, sqlite
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "factory_defect_db"
    db_user: str = "postgres"
    db_password: str = "password"
    
    # Logging
    log_images: bool = True
    image_storage_path: str = "data/logged_images/"
    max_image_storage_gb: float = 100.0
    compress_images: bool = True


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    streamlit_host: str = "0.0.0.0"
    streamlit_port: int = 8501
    refresh_interval_seconds: int = 2
    max_history_hours: int = 24
    show_raw_camera: bool = True
    show_detections: bool = True
    show_anomalies: bool = True


@dataclass
class DataConfig:
    """Data configuration."""
    data_root: str = "data/"
    train_dir: str = "data/raw/train/"
    val_dir: str = "data/raw/val/"
    test_dir: str = "data/raw/test/"
    
    # Defect classes
    defect_classes: list = field(default_factory=lambda: [
        "scratch",
        "crack",
        "dent",
        "missing_component",
        "contamination"
    ])
    
    num_classes: int = 5
    class_weights: list = field(default_factory=lambda: [1.0, 1.5, 1.2, 2.0, 1.3])


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    momentum: float = 0.9
    warmup_epochs: int = 10
    
    # Augmentation
    enable_augmentation: bool = True
    augmentation_probability: float = 0.8
    
    # Checkpointing
    save_interval: int = 10  # epochs
    best_model_metric: str = "mAP50"
    patience: int = 15  # Early stopping


@dataclass
class Config:
    """Main configuration class."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    factory: FactoryConfig = field(default_factory=FactoryConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Project settings
    project_root: str = str(Path(__file__).parent.parent.parent)
    experiment_name: str = "defect_detection_v1"
    seed: int = 42
    debug: bool = False


class ConfigLoader:
    """Load and manage configuration from YAML files."""
    
    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f) or {}
            logger.info(f"Loaded config from: {config_path}")
            return config_dict
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    @staticmethod
    def merge_configs(default_config: Dict[str, Any], 
                     loaded_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge loaded config with defaults."""
        merged = default_config.copy()
        for key, value in loaded_config.items():
            if key in merged and isinstance(merged[key], dict):
                merged[key].update(value)
            else:
                merged[key] = value
        return merged
    
    @staticmethod
    def load_config(config_path: str = None) -> Config:
        """Load complete configuration."""
        if config_path is None:
            config_path = "configs/default_config.yaml"
        
        # Load from file
        loaded = ConfigLoader.load_yaml(config_path)
        
        # Create config objects
        config = Config()
        
        # Update from loaded config
        if 'camera' in loaded:
            for key, value in loaded['camera'].items():
                setattr(config.camera, key, value)
        
        if 'model' in loaded:
            for key, value in loaded['model'].items():
                setattr(config.model, key, value)
        
        if 'inference' in loaded:
            for key, value in loaded['inference'].items():
                setattr(config.inference, key, value)
        
        if 'decision' in loaded:
            for key, value in loaded['decision'].items():
                setattr(config.decision, key, value)
        
        if 'factory' in loaded:
            for key, value in loaded['factory'].items():
                setattr(config.factory, key, value)
        
        if 'dashboard' in loaded:
            for key, value in loaded['dashboard'].items():
                setattr(config.dashboard, key, value)
        
        if 'data' in loaded:
            for key, value in loaded['data'].items():
                setattr(config.data, key, value)
        
        if 'training' in loaded:
            for key, value in loaded['training'].items():
                setattr(config.training, key, value)
        
        # Override with environment variables
        if os.getenv('DEVICE'):
            config.inference.device = os.getenv('DEVICE')
        
        if os.getenv('DEBUG'):
            config.debug = os.getenv('DEBUG').lower() == 'true'
        
        logger.info(f"Configuration loaded successfully. Debug mode: {config.debug}")
        return config


def get_config(config_path: str = None) -> Config:
    """Convenience function to load configuration."""
    return ConfigLoader.load_config(config_path)
