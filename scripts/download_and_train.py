# Download and prepare real industrial defect datasets
"""
Uses multiple sources to download real industrial defect data.
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import shutil
import cv2
import numpy as np
import json

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url, dest, desc="Downloading"):
    """Download with progress."""
    print(f"\n{desc}...")
    print(f"URL: {url[:80]}...")
    
    try:
        def hook(count, block_size, total_size):
            if total_size > 0:
                pct = min(100, int(count * block_size * 100 / total_size))
                sys.stdout.write(f"\r  {pct}%")
                sys.stdout.flush()
        
        # Add headers to avoid 403
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, dest, hook)
        print(" Done!")
        return True
    except Exception as e:
        print(f" Failed: {e}")
        return False


def download_pcb_defect():
    """
    Download PCB Defect Dataset from GitHub.
    6 defect types: missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper
    """
    print("="*60)
    print("DOWNLOADING PCB DEFECT DATASET")
    print("="*60)
    
    pcb_dir = DATA_DIR / "PCB_DATASET"
    
    if pcb_dir.exists() and len(list(pcb_dir.rglob("*.jpg"))) > 50:
        print("PCB dataset already exists!")
        return pcb_dir
    
    pcb_dir.mkdir(parents=True, exist_ok=True)
    
    # DeepPCB dataset from GitHub
    url = "https://github.com/tangsanli5201/DeepPCB/archive/refs/heads/master.zip"
    zip_path = DATA_DIR / "deeppcb.zip"
    
    if download_file(url, zip_path, "Downloading DeepPCB dataset"):
        print("Extracting...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(DATA_DIR)
            
            # Move files
            src = DATA_DIR / "DeepPCB-master" / "PCBData"
            if src.exists():
                for item in src.iterdir():
                    shutil.move(str(item), str(pcb_dir / item.name))
            
            shutil.rmtree(DATA_DIR / "DeepPCB-master", ignore_errors=True)
            os.remove(zip_path)
            print("PCB dataset ready!")
            return pcb_dir
        except Exception as e:
            print(f"Extraction failed: {e}")
    
    return None


def download_textile_defect():
    """
    Download AITEX Textile Defect Dataset.
    Fabric defect detection.
    """
    print("\n" + "="*60)
    print("DOWNLOADING TEXTILE DEFECT DATASET")
    print("="*60)
    
    textile_dir = DATA_DIR / "AITEX"
    
    if textile_dir.exists() and len(list(textile_dir.rglob("*.png"))) > 20:
        print("Textile dataset already exists!")
        return textile_dir
    
    textile_dir.mkdir(parents=True, exist_ok=True)
    
    # AITEX dataset
    url = "https://github.com/shshafin/Fabric-Defect-Detection-Dataset/archive/refs/heads/main.zip"
    zip_path = DATA_DIR / "textile.zip"
    
    if download_file(url, zip_path, "Downloading Textile Defect dataset"):
        print("Extracting...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(DATA_DIR)
            
            src = DATA_DIR / "Fabric-Defect-Detection-Dataset-main"
            if src.exists():
                for item in src.iterdir():
                    if item.is_dir():
                        shutil.move(str(item), str(textile_dir / item.name))
            
            shutil.rmtree(src, ignore_errors=True)
            os.remove(zip_path)
            print("Textile dataset ready!")
            return textile_dir
        except Exception as e:
            print(f"Extraction failed: {e}")
    
    return None


def create_unified_dataset():
    """
    Create unified YOLO dataset from all downloaded data.
    """
    print("\n" + "="*60)
    print("CREATING UNIFIED YOLO DATASET")
    print("="*60)
    
    output_dir = DATA_DIR / "industrial_defects"
    
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    class_names = ['defect']  # Binary for simplicity
    
    train_count = val_count = test_count = 0
    
    # Process PCB data
    pcb_dir = DATA_DIR / "PCB_DATASET"
    if pcb_dir.exists():
        print("\nProcessing PCB data...")
        
        for group_dir in pcb_dir.iterdir():
            if not group_dir.is_dir():
                continue
            
            for img_file in group_dir.glob("*_test.jpg"):
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                
                # Look for annotation
                txt_file = img_file.with_suffix('.txt')
                labels = []
                
                if txt_file.exists():
                    with open(txt_file) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 4:
                                x1, y1, x2, y2 = map(int, parts[:4])
                                cx = (x1 + x2) / 2 / w
                                cy = (y1 + y2) / 2 / h
                                bw = (x2 - x1) / w
                                bh = (y2 - y1) / h
                                labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                
                # Determine split
                r = np.random.random()
                if r < 0.1:
                    split = 'test'
                    idx = test_count
                    test_count += 1
                elif r < 0.2:
                    split = 'val'
                    idx = val_count
                    val_count += 1
                else:
                    split = 'train'
                    idx = train_count
                    train_count += 1
                
                out_img = output_dir / split / 'images' / f'pcb_{idx:04d}.jpg'
                out_lbl = output_dir / split / 'labels' / f'pcb_{idx:04d}.txt'
                
                # Resize to standard size
                img_resized = cv2.resize(img, (640, 640))
                cv2.imwrite(str(out_img), img_resized)
                
                with open(out_lbl, 'w') as f:
                    f.write('\n'.join(labels))
    
    # Process Textile data
    textile_dir = DATA_DIR / "AITEX"
    if textile_dir.exists():
        print("Processing Textile data...")
        
        for defect_dir in textile_dir.iterdir():
            if not defect_dir.is_dir():
                continue
            
            for img_file in defect_dir.glob("*.png"):
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                
                # Check if defect or mask
                if 'mask' in img_file.name.lower():
                    continue
                
                is_defect = 'defect' in defect_dir.name.lower() or 'nok' in defect_dir.name.lower()
                
                labels = []
                if is_defect:
                    # Look for corresponding mask
                    mask_file = img_file.parent / f"{img_file.stem}_mask.png"
                    if mask_file.exists():
                        mask = cv2.imread(str(mask_file), 0)
                        if mask is not None:
                            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for cnt in contours:
                                x, y, bw, bh = cv2.boundingRect(cnt)
                                cx = (x + bw/2) / w
                                cy = (y + bh/2) / h
                                labels.append(f"0 {cx:.6f} {cy:.6f} {bw/w:.6f} {bh/h:.6f}")
                    else:
                        labels.append("0 0.5 0.5 0.3 0.3")
                
                r = np.random.random()
                if r < 0.1:
                    split = 'test'
                    idx = test_count
                    test_count += 1
                elif r < 0.2:
                    split = 'val'
                    idx = val_count
                    val_count += 1
                else:
                    split = 'train'
                    idx = train_count
                    train_count += 1
                
                out_img = output_dir / split / 'images' / f'textile_{idx:04d}.jpg'
                out_lbl = output_dir / split / 'labels' / f'textile_{idx:04d}.txt'
                
                img_resized = cv2.resize(img, (640, 640))
                cv2.imwrite(str(out_img), img_resized)
                
                with open(out_lbl, 'w') as f:
                    f.write('\n'.join(labels))
    
    # Also include synthetic data
    synth_train = DATA_DIR / "train" / "images"
    if synth_train.exists():
        print("Adding synthetic data...")
        
        for img_file in synth_train.glob("*.jpg"):
            label_file = DATA_DIR / "train" / "labels" / f"{img_file.stem}.txt"
            
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            r = np.random.random()
            if r < 0.1:
                split = 'test'
                idx = test_count
                test_count += 1
            elif r < 0.2:
                split = 'val'
                idx = val_count
                val_count += 1
            else:
                split = 'train'
                idx = train_count
                train_count += 1
            
            out_img = output_dir / split / 'images' / f'synth_{idx:04d}.jpg'
            out_lbl = output_dir / split / 'labels' / f'synth_{idx:04d}.txt'
            
            cv2.imwrite(str(out_img), img)
            if label_file.exists():
                shutil.copy(label_file, out_lbl)
            else:
                open(out_lbl, 'w').close()
    
    # Create data.yaml
    yaml_content = f"""# Industrial Defect Detection Dataset
path: {output_dir.absolute()}
train: train/images
val: val/images
test: test/images

nc: 1
names: ['defect']
"""
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    # Also save to configs
    configs_dir = DATA_DIR.parent / "configs"
    configs_dir.mkdir(exist_ok=True)
    with open(configs_dir / 'real_data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print(f"\nDataset created!")
    print(f"Train: {train_count} images")
    print(f"Val: {val_count} images")
    print(f"Test: {test_count} images")
    print(f"Config: {yaml_path}")
    
    return output_dir, yaml_path


def train_yolo(data_yaml):
    """Train YOLOv8 on the dataset."""
    print("\n" + "="*60)
    print("TRAINING YOLOV8 DETECTOR")
    print("="*60)
    
    from ultralytics import YOLO
    
    model = YOLO('yolov8n.pt')
    
    print(f"\nData: {data_yaml}")
    print("Training for 15 epochs...")
    
    results = model.train(
        data=str(data_yaml),
        epochs=15,
        imgsz=640,
        batch=8,
        patience=5,
        save=True,
        project=str(DATA_DIR.parent / "runs"),
        name="real_defect_detector",
        exist_ok=True,
        verbose=True
    )
    
    print("\nTraining complete!")
    print(f"Best model: runs/real_defect_detector/weights/best.pt")
    
    return model


def run_inference(model, data_dir):
    """Run inference and log to database."""
    print("\n" + "="*60)
    print("RUNNING REAL INFERENCE")
    print("="*60)
    
    import sqlite3
    from datetime import datetime
    import time
    
    test_dir = Path(data_dir) / 'test' / 'images'
    db_path = DATA_DIR / 'defect_detection.db'
    
    # Clear old data and create fresh table
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS detections')
    cursor.execute('''
        CREATE TABLE detections (
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
    
    test_images = list(test_dir.glob('*.jpg'))
    print(f"\nProcessing {len(test_images)} test images...")
    
    stats = {'accept': 0, 'reject': 0, 'alert': 0, 'total': 0}
    
    for img_path in test_images:
        start = time.time()
        
        results = model.predict(str(img_path), verbose=False, conf=0.25)
        
        latency = (time.time() - start) * 1000
        
        detections = results[0].boxes
        image_id = img_path.stem
        timestamp = datetime.now().isoformat()
        
        if len(detections) > 0:
            for box in detections:
                conf = float(box.conf[0])
                x, y, w, h = box.xywhn[0].tolist()
                
                if conf > 0.7:
                    decision = 'REJECT'
                    stats['reject'] += 1
                else:
                    decision = 'ALERT'
                    stats['alert'] += 1
                
                cursor.execute('''
                    INSERT INTO detections 
                    (timestamp, image_id, defect_type, confidence, bbox_x, bbox_y, bbox_w, bbox_h, decision, latency_ms, line_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (timestamp, image_id, 'defect', conf, x, y, w, h, decision, latency, 1))
        else:
            cursor.execute('''
                INSERT INTO detections 
                (timestamp, image_id, defect_type, confidence, bbox_x, bbox_y, bbox_w, bbox_h, decision, latency_ms, line_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, image_id, 'none', 0.0, 0, 0, 0, 0, 'ACCEPT', latency, 1))
            stats['accept'] += 1
        
        stats['total'] += 1
        
        icon = {'ACCEPT': '✅', 'REJECT': '❌', 'ALERT': '⚠️'}
        status = 'REJECT' if len(detections) > 0 and float(detections[0].conf[0]) > 0.7 else ('ALERT' if len(detections) > 0 else 'ACCEPT')
        print(f"[{stats['total']:3d}] {image_id}: {icon[status]} {status} | {latency:.1f}ms")
    
    conn.commit()
    conn.close()
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE!")
    print("="*60)
    print(f"Total: {stats['total']}")
    print(f"✅ Accept: {stats['accept']}")
    print(f"❌ Reject: {stats['reject']}")
    print(f"⚠️ Alert: {stats['alert']}")
    print(f"\nResults saved to: {db_path}")


def main():
    print("="*60)
    print("REAL INDUSTRIAL DEFECT DETECTION PIPELINE")
    print("="*60)
    
    # Download datasets
    download_pcb_defect()
    download_textile_defect()
    
    # Create unified dataset
    data_dir, yaml_path = create_unified_dataset()
    
    # Train
    print("\n" + "="*60)
    train_choice = input("Train YOLOv8 now? (y/n) [y]: ").strip().lower() or 'y'
    
    if train_choice == 'y':
        model = train_yolo(yaml_path)
        
        # Run inference
        run_inference(model, data_dir)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
        print("\nRefresh dashboard at http://localhost:8501 to see results!")


if __name__ == "__main__":
    main()
