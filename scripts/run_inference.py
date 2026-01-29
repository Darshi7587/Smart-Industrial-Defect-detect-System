# Real-time Inference with Trained Model
"""
Runs inference on real images using the trained YOLOv8 model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO
import sqlite3
from datetime import datetime
import time


def run_inference_on_real_data():
    print("="*60)
    print("REAL-TIME DEFECT DETECTION INFERENCE")
    print("="*60)
    
    # Load trained model
    model_path = Path("runs/defect_detection/weights/best.pt")
    
    if not model_path.exists():
        print(f"\nModel not found at {model_path}")
        print("Using pretrained YOLOv8n instead...")
        model = YOLO('yolov8n.pt')
    else:
        print(f"\nLoading trained model: {model_path}")
        model = YOLO(str(model_path))
    
    # Test images
    test_dir = Path("data/test/images")
    db_path = Path("data/defect_detection.db")
    
    # Setup database
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Get test images
    test_images = list(test_dir.glob("*.jpg"))[:20]
    
    print(f"\nProcessing {len(test_images)} test images...")
    
    results_summary = {'ACCEPT': 0, 'REJECT': 0, 'ALERT': 0}
    
    for img_path in test_images:
        start = time.time()
        
        # Run inference
        results = model(str(img_path), conf=0.3, verbose=False)
        
        # Process results
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                
                defect_type = ['scratch', 'crack', 'dent', 'contamination'][cls]
                detections.append({
                    'type': defect_type,
                    'confidence': conf,
                    'bbox': xyxy
                })
        
        # Decision
        if not detections:
            decision = 'ACCEPT'
        else:
            max_conf = max(d['confidence'] for d in detections)
            if max_conf > 0.8:
                decision = 'REJECT'
            elif max_conf > 0.5:
                decision = 'ALERT'
            else:
                decision = 'ACCEPT'
        
        results_summary[decision] += 1
        
        latency = (time.time() - start) * 1000
        
        # Log to database
        timestamp = datetime.now().isoformat()
        for det in detections:
            cursor.execute('''
                INSERT INTO detections 
                (timestamp, image_id, defect_type, confidence, bbox_x, bbox_y, bbox_w, bbox_h, decision, latency_ms, line_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, img_path.name, det['type'], det['confidence'], 
                  det['bbox'][0], det['bbox'][1], det['bbox'][2], det['bbox'][3], 
                  decision, latency, 1))
        
        if not detections:
            cursor.execute('''
                INSERT INTO detections 
                (timestamp, image_id, defect_type, confidence, bbox_x, bbox_y, bbox_w, bbox_h, decision, latency_ms, line_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, img_path.name, 'none', 0.0, 0, 0, 0, 0, decision, latency, 1))
        
        icon = {'ACCEPT': '✅', 'REJECT': '❌', 'ALERT': '⚠️'}[decision]
        defect_str = ', '.join([f"{d['type']}:{d['confidence']:.2f}" for d in detections]) if detections else 'none'
        print(f"{icon} {img_path.name} | {decision:6s} | {defect_str} | {latency:.1f}ms")
    
    conn.commit()
    conn.close()
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print("="*60)
    print(f"✅ Accepted: {results_summary['ACCEPT']}")
    print(f"❌ Rejected: {results_summary['REJECT']}")
    print(f"⚠️ Alerts:   {results_summary['ALERT']}")
    print(f"\n💾 Results logged to: {db_path}")
    print("\n🔄 Refresh dashboard to see updated results!")


if __name__ == "__main__":
    run_inference_on_real_data()
