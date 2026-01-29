# Download Real Industrial Defect Datasets
"""
Downloads and prepares real-world industrial defect datasets:
1. MVTec AD - Industry standard anomaly detection dataset
2. KolektorSDD2 - Surface defect detection
"""

import os
import sys
import urllib.request
import tarfile
import zipfile
from pathlib import Path
import shutil

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_with_progress(url, dest_path, desc="Downloading"):
    """Download file with progress bar."""
    print(f"\n{desc}...")
    print(f"URL: {url}")
    
    def progress(count, block_size, total_size):
        percent = min(100, int(count * block_size * 100 / total_size))
        mb_done = count * block_size / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\rProgress: {percent}% ({mb_done:.1f}/{mb_total:.1f} MB)")
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, dest_path, progress)
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        return False


def setup_mvtec_ad():
    """
    Download MVTec Anomaly Detection Dataset.
    Categories: bottle, cable, capsule, carpet, grid, hazelnut, leather,
                metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper
    """
    print("\n" + "="*70)
    print("MVTEC ANOMALY DETECTION DATASET")
    print("="*70)
    print("15 categories, 5354 images, real industrial defects")
    print("Categories: bottle, cable, capsule, carpet, grid, hazelnut, leather,")
    print("            metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper")
    
    mvtec_dir = DATA_DIR / "mvtec_ad"
    
    if mvtec_dir.exists() and any(mvtec_dir.iterdir()):
        print("\nMVTec AD already exists!")
        return mvtec_dir
    
    mvtec_dir.mkdir(parents=True, exist_ok=True)
    
    # MVTec AD download URL (official)
    # Note: The full dataset is ~4.9GB, downloading individual categories
    categories = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]
    
    base_url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/"
    
    # Category file IDs (from MVTec official)
    category_ids = {
        'bottle': '420938129-1629952094',
        'cable': '420938133-1629952094', 
        'capsule': '420938135-1629952094',
        'carpet': '420938137-1629952094',
        'grid': '420938139-1629952094',
        'hazelnut': '420938141-1629952094',
        'leather': '420938144-1629952094',
        'metal_nut': '420938148-1629952094',
        'pill': '420938150-1629952094',
        'screw': '420938152-1629952094',
        'tile': '420938154-1629952094',
        'toothbrush': '420938156-1629952094',
        'transistor': '420938158-1629952094',
        'wood': '420938160-1629952094',
        'zipper': '420938162-1629952094'
    }
    
    print("\nDownloading MVTec AD categories...")
    print("This may take a while (total ~4.9 GB)")
    
    # Download a few key categories for demo (smaller download)
    demo_categories = ['bottle', 'metal_nut', 'screw', 'wood']
    
    print(f"\nDownloading demo subset: {demo_categories}")
    print("(Full dataset can be downloaded from https://www.mvtec.com/company/research/datasets/mvtec-ad)")
    
    for cat in demo_categories:
        if cat in category_ids:
            url = f"{base_url}{category_ids[cat]}/mvtec_{cat}.tar.xz"
            tar_path = mvtec_dir / f"{cat}.tar.xz"
            
            if not (mvtec_dir / cat).exists():
                print(f"\n--- Downloading {cat} ---")
                if download_with_progress(url, tar_path, f"Downloading {cat}"):
                    print(f"Extracting {cat}...")
                    try:
                        import lzma
                        with lzma.open(tar_path) as xz:
                            with tarfile.open(fileobj=xz) as tar:
                                tar.extractall(mvtec_dir)
                        os.remove(tar_path)
                        print(f"{cat} ready!")
                    except Exception as e:
                        print(f"Extraction failed: {e}")
            else:
                print(f"{cat} already exists, skipping...")
    
    return mvtec_dir


def setup_kolektor_sdd2():
    """
    Download KolektorSDD2 Surface Defect Dataset.
    ~3000 images of industrial surface defects.
    """
    print("\n" + "="*70)
    print("KOLEKTOR SDD2 - SURFACE DEFECT DETECTION")
    print("="*70)
    print("~3000 images, binary classification (defect/no defect)")
    
    kolektor_dir = DATA_DIR / "kolektor_sdd2"
    
    if kolektor_dir.exists() and any(kolektor_dir.iterdir()):
        print("\nKolektorSDD2 already exists!")
        return kolektor_dir
    
    kolektor_dir.mkdir(parents=True, exist_ok=True)
    
    # KolektorSDD2 from official source
    url = "https://go.vicos.si/kolektorsdd2"
    
    print(f"\nDownload manually from: {url}")
    print(f"Extract to: {kolektor_dir}")
    
    return kolektor_dir


def setup_neu_surface_defect():
    """
    NEU Surface Defect Database - Steel surface defects.
    6 types: Crazing, Inclusion, Patches, Pitted, Rolled-in Scale, Scratches
    1800 images (300 per class)
    """
    print("\n" + "="*70)
    print("NEU SURFACE DEFECT DATABASE")  
    print("="*70)
    print("Steel defects - 6 classes, 1800 images")
    print("Classes: Crazing, Inclusion, Patches, Pitted, Rolled-in Scale, Scratches")
    
    neu_dir = DATA_DIR / "NEU-CLS"
    
    if neu_dir.exists() and any(neu_dir.iterdir()):
        print("\nNEU dataset already exists!")
        return neu_dir
    
    neu_dir.mkdir(parents=True, exist_ok=True)
    
    # NEU dataset from Kaggle/GitHub mirrors
    print("\nDownload from Kaggle:")
    print("https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database")
    print(f"\nExtract to: {neu_dir}")
    
    return neu_dir


def convert_mvtec_to_yolo(mvtec_dir, output_dir):
    """Convert MVTec AD format to YOLO detection format."""
    print("\n" + "="*70)
    print("CONVERTING MVTEC TO YOLO FORMAT")
    print("="*70)
    
    import cv2
    
    output_dir = Path(output_dir)
    
    # Create output structure
    for split in ['train', 'val']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Class mapping
    defect_classes = {}
    class_id = 0
    
    mvtec_dir = Path(mvtec_dir)
    
    train_count = 0
    val_count = 0
    
    for category_dir in mvtec_dir.iterdir():
        if not category_dir.is_dir():
            continue
            
        category = category_dir.name
        print(f"\nProcessing {category}...")
        
        # Process test images (these have defects)
        test_dir = category_dir / 'test'
        if test_dir.exists():
            for defect_type_dir in test_dir.iterdir():
                if not defect_type_dir.is_dir():
                    continue
                
                defect_type = defect_type_dir.name
                
                if defect_type == 'good':
                    # Good samples - no labels
                    for img_path in defect_type_dir.glob('*.png'):
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue
                        
                        # Split 80/20 train/val
                        if val_count < train_count * 0.25:
                            split = 'val'
                            idx = val_count
                            val_count += 1
                        else:
                            split = 'train'
                            idx = train_count
                            train_count += 1
                        
                        out_img = output_dir / split / 'images' / f'{category}_good_{idx:04d}.jpg'
                        out_lbl = output_dir / split / 'labels' / f'{category}_good_{idx:04d}.txt'
                        
                        cv2.imwrite(str(out_img), img)
                        open(out_lbl, 'w').close()  # Empty label file
                else:
                    # Defect samples
                    if defect_type not in defect_classes:
                        defect_classes[defect_type] = class_id
                        class_id += 1
                    
                    cls_id = defect_classes[defect_type]
                    
                    for img_path in defect_type_dir.glob('*.png'):
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue
                        
                        h, w = img.shape[:2]
                        
                        # Check for ground truth mask
                        gt_dir = category_dir / 'ground_truth' / defect_type
                        mask_path = gt_dir / f'{img_path.stem}_mask.png'
                        
                        labels = []
                        
                        if mask_path.exists():
                            # Use mask to get bounding box
                            mask = cv2.imread(str(mask_path), 0)
                            if mask is not None:
                                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                for cnt in contours:
                                    x, y, bw, bh = cv2.boundingRect(cnt)
                                    # Convert to YOLO format (normalized)
                                    cx = (x + bw/2) / w
                                    cy = (y + bh/2) / h
                                    nw = bw / w
                                    nh = bh / h
                                    labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                        else:
                            # No mask, use center region as approximate bbox
                            labels.append(f"{cls_id} 0.5 0.5 0.3 0.3")
                        
                        # Split 80/20
                        if val_count < train_count * 0.25:
                            split = 'val'
                            idx = val_count
                            val_count += 1
                        else:
                            split = 'train'
                            idx = train_count
                            train_count += 1
                        
                        out_img = output_dir / split / 'images' / f'{category}_{defect_type}_{idx:04d}.jpg'
                        out_lbl = output_dir / split / 'labels' / f'{category}_{defect_type}_{idx:04d}.txt'
                        
                        cv2.imwrite(str(out_img), img)
                        with open(out_lbl, 'w') as f:
                            f.write('\n'.join(labels))
    
    # Create data.yaml
    yaml_content = f"""# MVTec AD YOLO Format
path: {output_dir.absolute()}
train: train/images
val: val/images

nc: {len(defect_classes)}
names: {list(defect_classes.keys())}
"""
    
    with open(output_dir / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print(f"\nConversion complete!")
    print(f"Train images: {train_count}")
    print(f"Val images: {val_count}")
    print(f"Classes: {defect_classes}")
    print(f"Data config: {output_dir / 'data.yaml'}")
    
    return defect_classes


def main():
    print("="*70)
    print("REAL INDUSTRIAL DEFECT DATASET DOWNLOADER")
    print("="*70)
    
    print("\nAvailable datasets:")
    print("1. MVTec AD (Anomaly Detection) - 15 categories, ~4.9GB")
    print("2. KolektorSDD2 (Surface Defects) - ~3000 images")
    print("3. NEU Surface Defect (Steel) - 6 classes, 1800 images")
    print("4. Download all")
    
    choice = input("\nSelect (1-4) [1]: ").strip() or "1"
    
    if choice in ['1', '4']:
        mvtec_dir = setup_mvtec_ad()
        
        # Convert to YOLO format
        convert = input("\nConvert MVTec to YOLO format? (y/n) [y]: ").strip().lower() or 'y'
        if convert == 'y':
            yolo_dir = DATA_DIR / "mvtec_yolo"
            convert_mvtec_to_yolo(mvtec_dir, yolo_dir)
    
    if choice in ['2', '4']:
        setup_kolektor_sdd2()
    
    if choice in ['3', '4']:
        setup_neu_surface_defect()
    
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Download any missing datasets manually")
    print("2. Run training: python scripts/train_yolo.py")
    print("3. Run inference: python scripts/run_demo.py")


if __name__ == "__main__":
    main()
