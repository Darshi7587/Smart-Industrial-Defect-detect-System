# Download and Setup Real Industrial Defect Datasets
"""
Downloads real-world industrial defect detection datasets:
1. MVTec AD (Anomaly Detection) - Industry standard
2. NEU Surface Defect Database - Steel surface defects
3. DAGM - German industrial defects
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import shutil

# Dataset directory
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url, dest_path, desc="Downloading"):
    """Download file with progress."""
    print(f"\n📥 {desc}...")
    print(f"   URL: {url}")
    print(f"   Destination: {dest_path}")
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r   Progress: {percent}%")
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        print("\n   ✅ Download complete!")
        return True
    except Exception as e:
        print(f"\n   ❌ Download failed: {e}")
        return False


def setup_neu_surface_defect():
    """
    Download NEU Surface Defect Database.
    Contains 6 types of steel surface defects:
    - Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches
    1800 grayscale images (300 per class)
    """
    print("\n" + "="*60)
    print("📦 NEU SURFACE DEFECT DATABASE")
    print("="*60)
    print("Steel surface defects - 6 classes, 1800 images")
    
    dataset_dir = DATA_DIR / "NEU-DET"
    
    if dataset_dir.exists():
        print("✅ Dataset already exists!")
        return dataset_dir
    
    # NEU dataset from Kaggle mirror or direct link
    # Using a commonly available mirror
    url = "https://github.com/abin24/Surface-Defect-Detection/raw/master/NEU-DET.zip"
    zip_path = DATA_DIR / "NEU-DET.zip"
    
    # Alternative: Manual download instructions
    print("\n⚠️ Automatic download may fail. If so, download manually:")
    print("   1. Go to: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database")
    print("   2. Download and extract to: data/NEU-DET/")
    print("   3. Or use: https://github.com/abin24/Surface-Defect-Detection")
    
    # Try download
    if download_file(url, zip_path, "Downloading NEU Surface Defect Database"):
        print("📂 Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        os.remove(zip_path)
        print("✅ NEU dataset ready!")
    
    return dataset_dir


def setup_casting_defect():
    """
    Download Casting Product Defect Dataset.
    Binary classification: defective vs ok
    Real manufacturing inspection images
    """
    print("\n" + "="*60)
    print("📦 CASTING PRODUCT DEFECT DATASET")
    print("="*60)
    print("Manufacturing casting defects - Binary classification")
    
    dataset_dir = DATA_DIR / "casting_data"
    
    if dataset_dir.exists():
        print("✅ Dataset already exists!")
        return dataset_dir
    
    print("\n📋 Download from Kaggle:")
    print("   https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product")
    print("\n   Extract to: data/casting_data/")
    
    return dataset_dir


def setup_pcb_defect():
    """
    PCB (Printed Circuit Board) Defect Dataset.
    Common defects: missing hole, mouse bite, open circuit, short, spur, spurious copper
    """
    print("\n" + "="*60)
    print("📦 PCB DEFECT DATASET")
    print("="*60)
    print("PCB manufacturing defects - 6 defect types")
    
    dataset_dir = DATA_DIR / "PCB_DATASET"
    
    print("\n📋 Download from:")
    print("   https://www.kaggle.com/datasets/akhatova/pcb-defects")
    print("   https://github.com/tangsanli5201/DeepPCB")
    print("\n   Extract to: data/PCB_DATASET/")
    
    return dataset_dir


def create_sample_dataset():
    """Create a sample dataset structure with synthetic defects for immediate use."""
    print("\n" + "="*60)
    print("🔧 CREATING SAMPLE DATASET FOR IMMEDIATE USE")
    print("="*60)
    
    import numpy as np
    import cv2
    
    # Create directories
    for split in ['train', 'val', 'test']:
        for dtype in ['images', 'labels']:
            (DATA_DIR / split / dtype).mkdir(parents=True, exist_ok=True)
        
        # Classification structure
        for cls in ['scratch', 'crack', 'dent', 'contamination', 'good']:
            (DATA_DIR / 'classification' / split / cls).mkdir(parents=True, exist_ok=True)
    
    defect_classes = ['scratch', 'crack', 'dent', 'contamination']
    
    def generate_base_image():
        """Generate realistic product surface."""
        # Metal-like texture
        base = np.random.randint(160, 200, (640, 640, 3), dtype=np.uint8)
        # Add subtle texture
        noise = np.random.normal(0, 8, base.shape).astype(np.int16)
        base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # Slight gradient for realism
        gradient = np.linspace(0.95, 1.05, 640).reshape(1, -1, 1)
        base = np.clip(base * gradient, 0, 255).astype(np.uint8)
        return base
    
    def add_scratch(img):
        """Add scratch defect."""
        h, w = img.shape[:2]
        x1 = np.random.randint(50, w-150)
        y1 = np.random.randint(50, h-150)
        length = np.random.randint(80, 200)
        angle = np.random.uniform(0, np.pi)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        
        color = tuple(np.random.randint(40, 100, 3).tolist())
        thickness = np.random.randint(1, 4)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        
        # Add secondary lines
        for _ in range(3):
            ox, oy = np.random.randint(-8, 8, 2)
            cv2.line(img, (x1+ox, y1+oy), (x2+ox, y2+oy), color, 1)
        
        # Bounding box in YOLO format
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = abs(x2 - x1 + 20) / w
        bh = abs(y2 - y1 + 20) / h
        return img, (0, cx, cy, bw, bh)  # class 0 = scratch
    
    def add_crack(img):
        """Add crack defect."""
        h, w = img.shape[:2]
        x, y = np.random.randint(100, w-100), np.random.randint(100, h-100)
        points = [(x, y)]
        
        for _ in range(np.random.randint(8, 20)):
            dx = np.random.randint(-25, 25)
            dy = np.random.randint(-25, 25)
            x = np.clip(x + dx, 10, w-10)
            y = np.clip(y + dy, 10, h-10)
            points.append((x, y))
        
        color = tuple(np.random.randint(20, 70, 3).tolist())
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i+1], color, np.random.randint(1, 3))
        
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        cx = (min(xs) + max(xs)) / 2 / w
        cy = (min(ys) + max(ys)) / 2 / h
        bw = (max(xs) - min(xs) + 20) / w
        bh = (max(ys) - min(ys) + 20) / h
        return img, (1, cx, cy, bw, bh)  # class 1 = crack
    
    def add_dent(img):
        """Add dent defect."""
        h, w = img.shape[:2]
        cx = np.random.randint(80, w-80)
        cy = np.random.randint(80, h-80)
        radius = np.random.randint(25, 60)
        
        # Create dent effect
        for r in range(radius, 0, -2):
            alpha = (radius - r) / radius * 0.4
            overlay = img.copy()
            cv2.circle(overlay, (cx, cy), r, (90, 90, 90), -1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        bcx = cx / w
        bcy = cy / h
        bw = radius * 2 / w
        bh = radius * 2 / h
        return img, (2, bcx, bcy, bw, bh)  # class 2 = dent
    
    def add_contamination(img):
        """Add contamination spots."""
        h, w = img.shape[:2]
        cx = np.random.randint(50, w-50)
        cy = np.random.randint(50, h-50)
        size = np.random.randint(15, 45)
        
        color = tuple(np.random.randint(0, 80, 3).tolist())
        cv2.circle(img, (cx, cy), size, color, -1)
        
        # Add irregular edges
        for _ in range(5):
            ox = np.random.randint(-size//2, size//2)
            oy = np.random.randint(-size//2, size//2)
            s = np.random.randint(5, size//2)
            cv2.circle(img, (cx+ox, cy+oy), s, color, -1)
        
        bcx = cx / w
        bcy = cy / h
        bw = size * 2.5 / w
        bh = size * 2.5 / h
        return img, (3, bcx, bcy, bw, bh)  # class 3 = contamination
    
    generators = [add_scratch, add_crack, add_dent, add_contamination]
    
    # Generate dataset
    counts = {'train': 200, 'val': 50, 'test': 50}
    
    for split, count in counts.items():
        print(f"\n📁 Generating {split} set ({count} images)...")
        
        for i in range(count):
            img = generate_base_image()
            labels = []
            
            # 30% chance of defect
            if np.random.random() < 0.35:
                # Add 1-3 defects
                num_defects = np.random.randint(1, 4)
                selected = np.random.choice(len(generators), min(num_defects, len(generators)), replace=False)
                
                defect_class = None
                for idx in selected:
                    img, label = generators[idx](img)
                    labels.append(label)
                    defect_class = defect_classes[idx]
                
                # Save for classification
                cls_path = DATA_DIR / 'classification' / split / defect_class / f"{i:04d}.jpg"
                cv2.imwrite(str(cls_path), img)
            else:
                # Good product
                cls_path = DATA_DIR / 'classification' / split / 'good' / f"{i:04d}.jpg"
                cv2.imwrite(str(cls_path), img)
            
            # Save for detection
            img_path = DATA_DIR / split / 'images' / f"{i:04d}.jpg"
            label_path = DATA_DIR / split / 'labels' / f"{i:04d}.txt"
            
            cv2.imwrite(str(img_path), img)
            
            with open(label_path, 'w') as f:
                for label in labels:
                    f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")
            
            if (i + 1) % 50 == 0:
                print(f"   Generated {i+1}/{count} images")
    
    print("\n✅ Sample dataset created!")
    print(f"   Detection: data/train/images, data/val/images, data/test/images")
    print(f"   Classification: data/classification/train, val, test")
    
    return DATA_DIR


def create_data_yaml():
    """Create YOLO data.yaml for training."""
    yaml_content = """# Industrial Defect Detection Dataset
path: ../data
train: train/images
val: val/images
test: test/images

nc: 4
names:
  0: scratch
  1: crack
  2: dent
  3: contamination
"""
    
    yaml_path = DATA_DIR.parent / "configs" / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Created {yaml_path}")


def main():
    print("="*60)
    print("🏭 INDUSTRIAL DEFECT DETECTION - DATASET SETUP")
    print("="*60)
    
    print("\n📋 Available Datasets:")
    print("1. NEU Surface Defect Database (Steel defects)")
    print("2. Casting Product Defect Dataset")
    print("3. PCB Defect Dataset")
    print("4. Create synthetic sample dataset (for immediate use)")
    print("5. All of the above")
    
    choice = input("\nSelect option (1-5) [4]: ").strip() or "4"
    
    if choice in ['1', '5']:
        setup_neu_surface_defect()
    
    if choice in ['2', '5']:
        setup_casting_defect()
    
    if choice in ['3', '5']:
        setup_pcb_defect()
    
    if choice in ['4', '5']:
        create_sample_dataset()
        create_data_yaml()
    
    print("\n" + "="*60)
    print("✅ DATASET SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Train detection model: python -m src.training.train_detection")
    print("2. Train classifier: python -m src.training.train_classification")
    print("3. Run demo: python scripts/run_demo.py")


if __name__ == "__main__":
    main()
