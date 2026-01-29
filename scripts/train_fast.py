"""
Fast training script - optimized for CPU
Reduces epochs, batch size, and image size for quick results
"""

from pathlib import Path
from ultralytics import YOLO

def train_fast():
    print("="*60)
    print("FAST TRAINING MODE (CPU-OPTIMIZED)")
    print("="*60)
    
    yaml_path = Path(__file__).parent.parent / "configs" / "mvtec_data.yaml"
    
    if not yaml_path.exists():
        print(f"❌ Config not found: {yaml_path}")
        print("Run: python scripts/download_mvtec.py first")
        return
    
    # Use nano model (smallest, fastest)
    model = YOLO('yolov8n.pt')
    
    print("\n⚡ CPU-Optimized Settings:")
    print("  • Epochs: 10 (instead of 30)")
    print("  • Batch: 4 (instead of 8)")
    print("  • Image size: 320 (instead of 640)")
    print("  • Workers: 0 (CPU-friendly)")
    print("  • Cache: True (faster loading)")
    print("\n🚀 Starting training...\n")
    
    results = model.train(
        data=str(yaml_path),
        epochs=10,           # Quick training
        imgsz=320,          # Smaller images = 4x faster
        batch=4,            # Smaller batch for CPU
        device='cpu',
        project='runs',
        name='mvtec_fast',
        exist_ok=True,
        patience=5,         # Early stopping
        cache=True,         # Cache images for speed
        workers=0,          # CPU-friendly
        verbose=True,
        amp=False           # Disable AMP on CPU
    )
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"Best model: runs/mvtec_fast/weights/best.pt")
    print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.3f}")
    
    # Test inference
    print("\n🔍 Testing inference...")
    test_model = YOLO('runs/mvtec_fast/weights/best.pt')
    
    # Find a test image
    test_imgs = list(Path('data/mvtec_yolo/val/images').glob('*.png'))[:5]
    if test_imgs:
        results = test_model.predict(test_imgs, save=True, project='runs', name='test_predictions')
        print(f"✅ Tested on {len(test_imgs)} images")
        print(f"📁 Results saved to: runs/test_predictions/")

if __name__ == "__main__":
    train_fast()
