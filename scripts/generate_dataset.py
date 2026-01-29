# Generate Real Industrial Defect Dataset
"""
Creates a realistic industrial defect detection dataset.
"""

import os
import numpy as np
import cv2
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

def generate_base_image():
    """Generate realistic product surface."""
    base = np.random.randint(160, 200, (640, 640, 3), dtype=np.uint8)
    noise = np.random.normal(0, 8, base.shape).astype(np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    gradient = np.linspace(0.95, 1.05, 640).reshape(1, -1, 1)
    base = np.clip(base * gradient, 0, 255).astype(np.uint8)
    return base

def add_scratch(img):
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
    
    for _ in range(3):
        ox, oy = np.random.randint(-8, 8, 2)
        cv2.line(img, (x1+ox, y1+oy), (x2+ox, y2+oy), color, 1)
    
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = abs(x2 - x1 + 20) / w
    bh = abs(y2 - y1 + 20) / h
    return img, (0, cx, cy, bw, bh)

def add_crack(img):
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
    return img, (1, cx, cy, bw, bh)

def add_dent(img):
    h, w = img.shape[:2]
    cx = np.random.randint(80, w-80)
    cy = np.random.randint(80, h-80)
    radius = np.random.randint(25, 60)
    
    for r in range(radius, 0, -2):
        alpha = (radius - r) / radius * 0.4
        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), r, (90, 90, 90), -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    bcx = cx / w
    bcy = cy / h
    bw = radius * 2 / w
    bh = radius * 2 / h
    return img, (2, bcx, bcy, bw, bh)

def add_contamination(img):
    h, w = img.shape[:2]
    cx = np.random.randint(50, w-50)
    cy = np.random.randint(50, h-50)
    size = np.random.randint(15, 45)
    
    color = tuple(np.random.randint(0, 80, 3).tolist())
    cv2.circle(img, (cx, cy), size, color, -1)
    
    for _ in range(5):
        ox = np.random.randint(-size//2, size//2)
        oy = np.random.randint(-size//2, size//2)
        s = np.random.randint(5, size//2)
        cv2.circle(img, (cx+ox, cy+oy), s, color, -1)
    
    bcx = cx / w
    bcy = cy / h
    bw = size * 2.5 / w
    bh = size * 2.5 / h
    return img, (3, bcx, bcy, bw, bh)

def main():
    print("="*60)
    print("GENERATING INDUSTRIAL DEFECT DATASET")
    print("="*60)
    
    defect_classes = ['scratch', 'crack', 'dent', 'contamination']
    generators = [add_scratch, add_crack, add_dent, add_contamination]
    
    # Create directories
    for split in ['train', 'val', 'test']:
        (DATA_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (DATA_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
        for cls in defect_classes + ['good']:
            (DATA_DIR / 'classification' / split / cls).mkdir(parents=True, exist_ok=True)
    
    counts = {'train': 200, 'val': 50, 'test': 50}
    
    for split, count in counts.items():
        print(f"\nGenerating {split} set ({count} images)...")
        
        for i in range(count):
            img = generate_base_image()
            labels = []
            
            if np.random.random() < 0.35:
                num_defects = np.random.randint(1, 4)
                selected = np.random.choice(len(generators), min(num_defects, len(generators)), replace=False)
                
                defect_class = None
                for idx in selected:
                    img, label = generators[idx](img)
                    labels.append(label)
                    defect_class = defect_classes[idx]
                
                cls_path = DATA_DIR / 'classification' / split / defect_class / f"{i:04d}.jpg"
                cv2.imwrite(str(cls_path), img)
            else:
                cls_path = DATA_DIR / 'classification' / split / 'good' / f"{i:04d}.jpg"
                cv2.imwrite(str(cls_path), img)
            
            img_path = DATA_DIR / split / 'images' / f"{i:04d}.jpg"
            label_path = DATA_DIR / split / 'labels' / f"{i:04d}.txt"
            
            cv2.imwrite(str(img_path), img)
            
            with open(label_path, 'w') as f:
                for label in labels:
                    f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")
            
            if (i + 1) % 50 == 0:
                print(f"  Generated {i+1}/{count}")
    
    print("\n" + "="*60)
    print("DATASET CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"Train: {counts['train']} images")
    print(f"Val: {counts['val']} images")
    print(f"Test: {counts['test']} images")
    print(f"\nLocation: {DATA_DIR}")

if __name__ == "__main__":
    main()
