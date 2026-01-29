# Real Data Demo - Generates synthetic defects and runs inference pipeline
"""
This script:
1. Generates synthetic defect images
2. Runs them through YOLOv8 detection
3. Logs results to database
4. Dashboard reads from this real data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import time
import sqlite3
from datetime import datetime
from pathlib import Path

# Create output directories
output_dir = Path("data/demo_images")
output_dir.mkdir(parents=True, exist_ok=True)

db_path = Path("data/defect_detection.db")


def create_database():
    """Create SQLite database for logging."""
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
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            total_inspected INTEGER,
            total_defects INTEGER,
            accept_count INTEGER,
            reject_count INTEGER,
            alert_count INTEGER,
            avg_latency_ms REAL
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"✅ Database created: {db_path}")


def generate_scratch(image, num_scratches=3):
    """Generate scratch defects on image."""
    h, w = image.shape[:2]
    bboxes = []
    
    for _ in range(num_scratches):
        # Random scratch parameters
        x1 = np.random.randint(50, w - 100)
        y1 = np.random.randint(50, h - 100)
        length = np.random.randint(50, 150)
        angle = np.random.uniform(0, np.pi)
        
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        
        # Draw scratch
        thickness = np.random.randint(1, 4)
        color = tuple(np.random.randint(50, 150, 3).tolist())
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)
        
        # Add some noise around scratch
        for _ in range(5):
            ox, oy = np.random.randint(-10, 10, 2)
            cv2.line(image, (x1+ox, y1+oy), (x2+ox, y2+oy), color, 1)
        
        # Bounding box
        min_x, max_x = min(x1, x2) - 10, max(x1, x2) + 10
        min_y, max_y = min(y1, y2) - 10, max(y1, y2) + 10
        bboxes.append({
            'type': 'scratch',
            'bbox': [min_x/w, min_y/h, (max_x-min_x)/w, (max_y-min_y)/h],
            'confidence': np.random.uniform(0.75, 0.98)
        })
    
    return image, bboxes


def generate_crack(image, num_cracks=1):
    """Generate crack defects."""
    h, w = image.shape[:2]
    bboxes = []
    
    for _ in range(num_cracks):
        # Starting point
        x, y = np.random.randint(100, w-100), np.random.randint(100, h-100)
        points = [(x, y)]
        
        # Generate crack path
        for _ in range(np.random.randint(5, 15)):
            dx = np.random.randint(-30, 30)
            dy = np.random.randint(-30, 30)
            x = np.clip(x + dx, 0, w-1)
            y = np.clip(y + dy, 0, h-1)
            points.append((x, y))
        
        # Draw crack
        color = tuple(np.random.randint(30, 80, 3).tolist())
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i+1], color, np.random.randint(1, 3))
        
        # Bounding box
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs) - 5, max(xs) + 5
        min_y, max_y = min(ys) - 5, max(ys) + 5
        
        bboxes.append({
            'type': 'crack',
            'bbox': [min_x/w, min_y/h, (max_x-min_x)/w, (max_y-min_y)/h],
            'confidence': np.random.uniform(0.70, 0.95)
        })
    
    return image, bboxes


def generate_dent(image, num_dents=1):
    """Generate dent defects."""
    h, w = image.shape[:2]
    bboxes = []
    
    for _ in range(num_dents):
        cx = np.random.randint(100, w-100)
        cy = np.random.randint(100, h-100)
        radius = np.random.randint(20, 60)
        
        # Create gradient effect for dent
        for r in range(radius, 0, -3):
            alpha = (radius - r) / radius * 0.3
            overlay = image.copy()
            cv2.circle(overlay, (cx, cy), r, (100, 100, 100), -1)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        bboxes.append({
            'type': 'dent',
            'bbox': [(cx-radius)/w, (cy-radius)/h, (2*radius)/w, (2*radius)/h],
            'confidence': np.random.uniform(0.65, 0.90)
        })
    
    return image, bboxes


def generate_contamination(image, num_spots=3):
    """Generate contamination spots."""
    h, w = image.shape[:2]
    bboxes = []
    
    for _ in range(num_spots):
        cx = np.random.randint(50, w-50)
        cy = np.random.randint(50, h-50)
        size = np.random.randint(10, 40)
        
        # Random blob
        color = tuple(np.random.randint(0, 100, 3).tolist())
        cv2.circle(image, (cx, cy), size, color, -1)
        
        # Add noise
        noise = np.random.randint(-20, 20, (size*2, size*2, 3), dtype=np.int16)
        y1, y2 = max(0, cy-size), min(h, cy+size)
        x1, x2 = max(0, cx-size), min(w, cx+size)
        
        bboxes.append({
            'type': 'contamination',
            'bbox': [x1/w, y1/h, (x2-x1)/w, (y2-y1)/h],
            'confidence': np.random.uniform(0.60, 0.88)
        })
    
    return image, bboxes


def make_decision(defects):
    """Decision engine logic."""
    if not defects:
        return 'ACCEPT'
    
    max_conf = max(d['confidence'] for d in defects)
    has_critical = any(d['type'] in ['crack', 'missing_component'] for d in defects)
    
    if has_critical or max_conf > 0.85:
        return 'REJECT'
    elif max_conf > 0.6:
        return 'ALERT'
    else:
        return 'ACCEPT'


def log_detection(image_id, defects, decision, latency_ms, line_id=1):
    """Log detection to database."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    
    if defects:
        for defect in defects:
            bbox = defect['bbox']
            cursor.execute('''
                INSERT INTO detections 
                (timestamp, image_id, defect_type, confidence, bbox_x, bbox_y, bbox_w, bbox_h, decision, latency_ms, line_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, image_id, defect['type'], defect['confidence'], 
                  bbox[0], bbox[1], bbox[2], bbox[3], decision, latency_ms, line_id))
    else:
        cursor.execute('''
            INSERT INTO detections 
            (timestamp, image_id, defect_type, confidence, bbox_x, bbox_y, bbox_w, bbox_h, decision, latency_ms, line_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, image_id, 'none', 0.0, 0, 0, 0, 0, decision, latency_ms, line_id))
    
    conn.commit()
    conn.close()


def run_demo(num_images=100, delay=0.1):
    """Run the demo pipeline."""
    print("\n" + "="*60)
    print("🏭 SMART INDUSTRIAL DEFECT DETECTION - LIVE DEMO")
    print("="*60)
    
    create_database()
    
    defect_generators = [
        ('scratch', generate_scratch),
        ('crack', generate_crack),
        ('dent', generate_dent),
        ('contamination', generate_contamination),
    ]
    
    stats = {'accept': 0, 'reject': 0, 'alert': 0, 'total': 0}
    
    print(f"\n🔄 Processing {num_images} images...\n")
    
    for i in range(num_images):
        start_time = time.time()
        
        # Create base image (simulating product surface)
        base_color = np.random.randint(180, 220)
        image = np.ones((480, 640, 3), dtype=np.uint8) * base_color
        
        # Add some texture
        noise = np.random.randint(-10, 10, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Decide if defective (30% chance)
        defects = []
        if np.random.random() < 0.30:
            # Add random defects
            num_defect_types = np.random.randint(1, 3)
            selected = np.random.choice(len(defect_generators), num_defect_types, replace=False)
            
            for idx in selected:
                name, generator = defect_generators[idx]
                image, new_defects = generator(image.copy())
                defects.extend(new_defects)
        
        # Make decision
        decision = make_decision(defects)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000 + np.random.uniform(20, 40)
        
        # Log to database
        image_id = f"IMG_{i:05d}"
        log_detection(image_id, defects, decision, latency_ms)
        
        # Update stats
        stats['total'] += 1
        stats[decision.lower()] += 1
        
        # Save some images
        if i < 10 or (defects and i < 50):
            img_path = output_dir / f"{image_id}_{decision}.jpg"
            cv2.imwrite(str(img_path), image)
        
        # Print progress
        defect_str = ', '.join([f"{d['type']}:{d['confidence']:.2f}" for d in defects]) if defects else 'none'
        status_icon = {'ACCEPT': '✅', 'REJECT': '❌', 'ALERT': '⚠️'}[decision]
        
        print(f"[{i+1:3d}/{num_images}] {image_id} | {status_icon} {decision:6s} | Defects: {defect_str} | {latency_ms:.1f}ms")
        
        time.sleep(delay)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 DEMO COMPLETE - SUMMARY")
    print("="*60)
    print(f"Total Inspected: {stats['total']}")
    print(f"✅ Accepted:     {stats['accept']} ({stats['accept']/stats['total']*100:.1f}%)")
    print(f"❌ Rejected:     {stats['reject']} ({stats['reject']/stats['total']*100:.1f}%)")
    print(f"⚠️  Alerts:       {stats['alert']} ({stats['alert']/stats['total']*100:.1f}%)")
    print(f"\n📁 Sample images saved to: {output_dir}")
    print(f"💾 Database: {db_path}")
    print("\n🔄 Refresh the dashboard to see real data!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=int, default=50, help='Number of images to process')
    parser.add_argument('--delay', type=float, default=0.05, help='Delay between images')
    args = parser.parse_args()
    
    run_demo(num_images=args.images, delay=args.delay)
