# Training module
from .train_detection import train_yolov8_detector
from .train_classification import train_classifier
from .train_anomaly import train_autoencoder
from .losses import FocalLoss, DiceLoss, CombinedLoss
from .metrics import compute_detection_metrics, compute_classification_metrics
