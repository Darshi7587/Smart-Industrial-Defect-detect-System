# Train YOLOv8 on Real MVTec Dataset
"""
Trains on real MVTec industrial defect data.
"""

from ultralytics import YOLO
from pathlib import Path

def train_on_mvtec():
    print("="*60)
    print("TRAINING ON REAL MVTEC AD DATASET")
    print("="*60)
    
    yaml_path = Path("configs/mvtec_data.yaml")
    
    if not yaml_path.exists():
        print("❌ MVTec dataset not found!")
        print("Run: python scripts/download_mvtec.py")
        return
    
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data=str(yaml_path),
        epochs=30,
        imgsz=640,
        batch=8,
        project="runs",
        name="mvtec_detection",
        exist_ok=True,
        verbose=True
    )
    
    print("\n✅ Training complete!")
    print("Model: runs/mvtec_detection/weights/best.pt")

if __name__ == "__main__":
    train_on_mvtec()
