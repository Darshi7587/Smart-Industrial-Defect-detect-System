# Download MVTec AD - Real Industrial Defect Dataset
"""
Downloads the MVTec Anomaly Detection dataset.
Real-world industrial defect detection benchmark.
"""

import urllib.request
import tarfile
import zipfile
from pathlib import Path
import os
import shutil
import kagglehub

DATA_DIR = Path(__file__).parent.parent / "data" / "mvtec_ad"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# MVTec AD dataset categories (picking most relevant for manufacturing)
CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

BASE_URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download"

def download_mvtec():
    print("="*60)
    print("DOWNLOADING MVTEC AD DATASET (REAL)")
    print("="*60)
    print("Using Kaggle automatic download...")
    
    # Check if already downloaded
    if (DATA_DIR / "bottle").exists():
        print(f"\n✅ Dataset already exists at: {DATA_DIR}")
        return True
    
    try:
        print("\n🔄 Downloading from Kaggle (requires kagglehub)...")
        print("This will take a few minutes (4.9 GB)...")
        
        # Download using kagglehub
        path = kagglehub.dataset_download("ipythonx/mvtec-ad")
        print(f"\n✅ Downloaded to: {path}")
        
        # Copy to our data directory
        source_path = Path(path)
        if source_path.exists():
            print(f"\n📦 Copying to project directory...")
            for item in source_path.iterdir():
                dest = DATA_DIR / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
            print(f"✅ Copied to: {DATA_DIR}")
            return True
        else:
            raise Exception("Download path not found")
            
    except Exception as e:
        print(f"\n❌ Auto-download failed: {e}")
        print("\n📋 MANUAL DOWNLOAD INSTRUCTIONS:")
        print("\nOption 1 - Official Source:")
        print("  1. Visit: https://www.mvtec.com/company/research/datasets/mvtec-ad")
        print("  2. Fill the form and download")
        print(f"  3. Extract to: {DATA_DIR}")
        
        print("\nOption 2 - Kaggle Manual:")
        print("  1. Visit: https://www.kaggle.com/datasets/ipythonx/mvtec-ad")
        print("  2. Click 'Download' (requires Kaggle account)")
        print(f"  3. Extract to: {DATA_DIR}")
        
        print("\nOption 3 - Use Synthetic Data:")
        print("  The system has 300 high-quality synthetic images ready!")
        return False
    
    # Extract
    if not (DATA_DIR / "bottle").exists():
        print(f"\n📂 Extracting to {DATA_DIR}...")
        import lzma
        
        with lzma.open(tar_path) as f:
            with tarfile.open(fileobj=f) as tar:
                tar.extractall(DATA_DIR.parent)
        
        print("✅ Extraction complete!")
    else:
        print("✅ Dataset already extracted!")
    
    return True

def convert_to_yolo_format():
    """Convert MVTec to YOLO detection format."""
    print("\n📝 Converting to YOLO format...")
    
    yolo_dir = DATA_DIR.parent / "mvtec_yolo"
    
    for split in ['train', 'val', 'test']:
        (yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    import cv2
    import numpy as np
    
    img_count = 0
    
    for cat in CATEGORIES[:5]:  # Use first 5 categories
        cat_dir = DATA_DIR / cat
        if not cat_dir.exists():
            continue
        
        # Good images (train)
        good_dir = cat_dir / 'train' / 'good'
        if good_dir.exists():
            for img_path in list(good_dir.glob('*.png'))[:30]:
                dest = yolo_dir / 'train' / 'images' / f"{cat}_{img_path.name}"
                shutil.copy(img_path, dest)
                # Empty label file for good images
                label_path = yolo_dir / 'train' / 'labels' / f"{cat}_{img_path.stem}.txt"
                label_path.touch()
                img_count += 1
        
        # Defect images (val/test)
        test_dir = cat_dir / 'test'
        if test_dir.exists():
            for defect_type in test_dir.iterdir():
                if not defect_type.is_dir() or defect_type.name == 'good':
                    continue
                
                imgs = list(defect_type.glob('*.png'))[:10]
                for i, img_path in enumerate(imgs):
                    split = 'val' if i < 5 else 'test'
                    dest = yolo_dir / split / 'images' / f"{cat}_{defect_type.name}_{img_path.name}"
                    shutil.copy(img_path, dest)
                    
                    # Create bbox label (full image as defective)
                    label_path = yolo_dir / split / 'labels' / f"{cat}_{defect_type.name}_{img_path.stem}.txt"
                    with open(label_path, 'w') as f:
                        # Class 0 = defect, centered bbox covering most of image
                        f.write("0 0.5 0.5 0.8 0.8\n")
                    
                    img_count += 1
    
    print(f"✅ Converted {img_count} images to YOLO format")
    print(f"   Location: {yolo_dir}")
    
    # Create data.yaml
    yaml_content = f"""path: {yolo_dir.absolute()}
train: train/images
val: val/images
test: test/images

nc: 1
names:
  0: defect
"""
    
    yaml_path = Path(__file__).parent.parent / "configs" / "mvtec_data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Created config: {yaml_path}")
    return yolo_dir

if __name__ == "__main__":
    print("🏭 INDUSTRIAL DEFECT DETECTION - Dataset Setup\n")
    
    if download_mvtec():
        convert_to_yolo_format()
        print("\n" + "="*60)
        print("✅ MVTEC AD DATASET READY!")
        print("="*60)
        print("\nTo train:")
        print("  python scripts/train_on_mvtec.py")
    else:
        print("\n" + "="*60)
        print("⚡ QUICK START WITH SYNTHETIC DATA")
        print("="*60)
        print("You have 300 high-quality synthetic images ready!")
        print("\n🚀 Start training: python scripts/train_yolo.py")
        print("📊 View dashboard: streamlit run src/dashboard/app.py")
