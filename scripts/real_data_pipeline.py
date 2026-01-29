# Real Dataset Pipeline - Downloads and trains on real industrial data
"""
Downloads a real industrial defect dataset and runs the complete pipeline:
1. Downloads Casting Defect Dataset from direct URL
2. Prepares data in YOLO format
3. Trains YOLOv8 detector
4. Runs inference and logs to dashboard
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
import shutil
import cv2
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"


def download_casting_dataset():
    """
    Download Casting Product Image Dataset.
    Real manufacturing images - binary classification (def_front, ok_front)
    """
    print("="*60)
    print("DOWNLOADING CASTING DEFECT DATASET")
    print("="*60)
    
    casting_dir = DATA_DIR / "casting_data"
    
    if casting_dir.exists() and len(list(casting_dir.rglob("*.jpeg"))) > 100:
        print("Dataset already exists!")
        return casting_dir
    
    casting_dir.mkdir(parents=True, exist_ok=True)
    
    # Direct download links for casting dataset
    # This is a real manufacturing dataset
    print("\nThis dataset contains real casting product images.")
    print("Categories: defective (def_front) and good (ok_front)")
    
    # Alternative: Download from GitHub mirror
    url = "https://github.com/ravirajsinh45/Casting_Product_Image_data/archive/refs/heads/main.zip"
    
    zip_path = DATA_DIR / "casting.zip"
    
    print(f"\nDownloading from: {url}")
    
    try:
        def progress(count, block_size, total_size):
            percent = min(100, int(count * block_size * 100 / max(total_size, 1)))
            sys.stdout.write(f"\rProgress: {percent}%")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, zip_path, progress)
        print("\nDownload complete!")
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        
        # Move to correct location
        extracted = DATA_DIR / "Casting_Product_Image_data-main" / "casting_data"
        if extracted.exists():
            if casting_dir.exists():
                shutil.rmtree(casting_dir)
            shutil.move(str(extracted), str(casting_dir))
        
        # Cleanup
        shutil.rmtree(DATA_DIR / "Casting_Product_Image_data-main", ignore_errors=True)
        os.remove(zip_path)
        
        print("Extraction complete!")
        return casting_dir
        
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("\nPlease download manually from:")
        print("https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product")
        return None


def convert_casting_to_yolo(casting_dir):
    """Convert casting dataset to YOLO detection format."""
    print("\n" + "="*60)
    print("CONVERTING TO YOLO FORMAT")
    print("="*60)
    
    casting_dir = Path(casting_dir)
    output_dir = DATA_DIR / "casting_yolo"
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Process images
    train_count = val_count = test_count = 0
    
    for split_name in ['train', 'test']:
        split_dir = casting_dir / split_name
        if not split_dir.exists():
            continue
        
        for category in ['def_front', 'ok_front']:
            cat_dir = split_dir / category
            if not cat_dir.exists():
                continue
            
            is_defect = category == 'def_front'
            
            images = list(cat_dir.glob('*.jpeg')) + list(cat_dir.glob('*.jpg')) + list(cat_dir.glob('*.png'))
            
            for i, img_path in enumerate(images):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                
                # Determine split (80% train, 10% val, 10% test)
                if split_name == 'test':
                    out_split = 'test'
                    idx = test_count
                    test_count += 1
                elif i % 10 == 0:
                    out_split = 'val'
                    idx = val_count
                    val_count += 1
                else:
                    out_split = 'train'
                    idx = train_count
                    train_count += 1
                
                # Save image
                out_img = output_dir / out_split / 'images' / f'{category}_{idx:04d}.jpg'
                cv2.imwrite(str(out_img), img)
                
                # Create label
                out_lbl = output_dir / out_split / 'labels' / f'{category}_{idx:04d}.txt'
                
                if is_defect:
                    # For defects, create approximate bounding box (center of image)
                    # In real scenario, you'd have actual annotations
                    with open(out_lbl, 'w') as f:
                        f.write("0 0.5 0.5 0.6 0.6\n")  # class 0 = defect
                else:
                    # Good images - empty label
                    open(out_lbl, 'w').close()
    
    # Create data.yaml
    yaml_content = f"""# Casting Defect Dataset - YOLO Format
path: {output_dir.absolute()}
train: train/images
val: val/images
test: test/images

nc: 1
names: ['defect']
"""
    
    with open(output_dir / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print(f"\nConversion complete!")
    print(f"Train: {train_count} images")
    print(f"Val: {val_count} images")
    print(f"Test: {test_count} images")
    print(f"Config: {output_dir / 'data.yaml'}")
    
    return output_dir


def train_on_real_data(data_dir):
    """Train YOLOv8 on real casting data."""
    print("\n" + "="*60)
    print("TRAINING YOLOV8 ON REAL DATA")
    print("="*60)
    
    from ultralytics import YOLO
    
    data_yaml = data_dir / 'data.yaml'
    
    print(f"Data config: {data_yaml}")
    
    # Load model
    print("\nLoading YOLOv8n...")
    model = YOLO('yolov8n.pt')
    
    # Train
    print("\nStarting training (10 epochs for demo)...")
    results = model.train(
        data=str(data_yaml),
        epochs=10,
        imgsz=512,
        batch=8,
        patience=5,
        save=True,
        project=str(Path(__file__).parent.parent / "runs"),
        name="casting_defect",
        exist_ok=True
    )
    
    print("\nTraining complete!")
    
    return model


def run_inference_demo(model, data_dir):
    """Run inference on test images and log to database."""
    print("\n" + "="*60)
    print("RUNNING INFERENCE ON TEST SET")
    print("="*60)
    
    import sqlite3
    from datetime import datetime
    import time
    
    test_dir = data_dir / 'test' / 'images'
    db_path = DATA_DIR / 'defect_detection.db'
    
    # Setup database
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            image_id TEXT,
            defect_type TEXT,
            confidence REAL,
            bbox_x REAL,
            bbox_y REAL,
            bbox_w REAL,
            bbox_h REAL,
            decision TEXT,
            latency_ms REAL,
            line_id INTEGER
        )
    ''')
    conn.commit()
    
    stats = {'accept': 0, 'reject': 0, 'total': 0}
    
    test_images = list(test_dir.glob('*.jpg'))[:50]  # Limit for demo
    
    print(f"\nProcessing {len(test_images)} test images...")
    
    for img_path in test_images:
        start = time.time()
        
        # Run inference
        results = model.predict(str(img_path), verbose=False)
        
        latency = (time.time() - start) * 1000
        
        # Process results
        detections = results[0].boxes
        
        image_id = img_path.stem
        timestamp = datetime.now().isoformat()
        
        if len(detections) > 0:
            # Defect detected
            for box in detections:
                conf = float(box.conf[0])
                x, y, w, h = box.xywhn[0].tolist()
                
                decision = 'REJECT' if conf > 0.5 else 'ALERT'
                
                cursor.execute('''
                    INSERT INTO detections 
                    (timestamp, image_id, defect_type, confidence, bbox_x, bbox_y, bbox_w, bbox_h, decision, latency_ms, line_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (timestamp, image_id, 'defect', conf, x, y, w, h, decision, latency, 1))
                
                stats['reject'] += 1
        else:
            # No defect
            cursor.execute('''
                INSERT INTO detections 
                (timestamp, image_id, defect_type, confidence, bbox_x, bbox_y, bbox_w, bbox_h, decision, latency_ms, line_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, image_id, 'none', 0.0, 0, 0, 0, 0, 'ACCEPT', latency, 1))
            
            stats['accept'] += 1
        
        stats['total'] += 1
        
        status = 'REJECT' if len(detections) > 0 else 'ACCEPT'
        icon = '❌' if status == 'REJECT' else '✅'
        print(f"[{stats['total']:3d}] {image_id}: {icon} {status} | {latency:.1f}ms")
    
    conn.commit()
    conn.close()
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE!")
    print("="*60)
    print(f"Total: {stats['total']}")
    print(f"Accepted: {stats['accept']}")
    print(f"Rejected: {stats['reject']}")
    print(f"\nResults logged to: {db_path}")
    print("\nRefresh the dashboard to see real inference results!")


def main():
    print("="*60)
    print("REAL INDUSTRIAL DEFECT DETECTION PIPELINE")
    print("="*60)
    
    # Step 1: Download dataset
    casting_dir = download_casting_dataset()
    
    if casting_dir is None:
        print("\nFailed to download dataset. Exiting.")
        return
    
    # Step 2: Convert to YOLO format
    yolo_dir = convert_casting_to_yolo(casting_dir)
    
    # Step 3: Train model
    print("\n" + "="*60)
    print("READY TO TRAIN")
    print("="*60)
    
    train_now = input("\nTrain YOLOv8 now? (y/n) [y]: ").strip().lower() or 'y'
    
    if train_now == 'y':
        model = train_on_real_data(yolo_dir)
        
        # Step 4: Run inference
        run_inference_demo(model, yolo_dir)
    else:
        print("\nTo train later, run:")
        print(f"  python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt').train(data='{yolo_dir / 'data.yaml'}', epochs=10)\"")


if __name__ == "__main__":
    main()
