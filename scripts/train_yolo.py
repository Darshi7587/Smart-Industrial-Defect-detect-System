# Train YOLOv8 on Real Industrial Defect Dataset
"""
Trains YOLOv8 detector on the generated defect dataset.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
import yaml


def train_detector():
    print("="*60)
    print("TRAINING YOLOV8 DEFECT DETECTOR")
    print("="*60)
    
    # Create data.yaml
    data_dir = Path(__file__).parent.parent / "data"
    config_dir = Path(__file__).parent.parent / "configs"
    
    data_yaml = {
        'path': str(data_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 4,
        'names': ['scratch', 'crack', 'dent', 'contamination']
    }
    
    yaml_path = config_dir / "train_data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    
    print(f"\nData config: {yaml_path}")
    print(f"Training images: {data_dir / 'train/images'}")
    
    # Load YOLOv8 nano model (fast training)
    print("\nLoading YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    
    # Train
    print("\nStarting training...")
    results = model.train(
        data=str(yaml_path),
        epochs=20,  # Quick training for demo
        imgsz=640,
        batch=8,
        patience=10,
        save=True,
        project=str(Path(__file__).parent.parent / "runs"),
        name="defect_detection",
        exist_ok=True,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    # Validate
    print("\nRunning validation...")
    val_results = model.val()
    
    print(f"\nResults saved to: runs/defect_detection/")
    print(f"Best model: runs/defect_detection/weights/best.pt")
    
    return model


if __name__ == "__main__":
    train_detector()
