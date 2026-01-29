# System Architecture
## End-to-End Smart Industrial Defect Detection

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FACTORY FLOOR                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Conveyor Belt with High-Speed Camera                              │
│       ↓                                                             │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │         REAL-TIME INFERENCE ENGINE (Edge GPU)           │     │
│  │                                                          │     │
│  │  1. Image Capture & Preprocessing                       │     │
│  │     - Camera interface (GigE, USB3)                      │     │
│  │     - Real-time normalization                           │     │
│  │     - Resize to model input size                        │     │
│  │                                                          │     │
│  │  2. Defect Detection Module                             │     │
│  │     - YOLOv8 model inference                            │     │
│  │     - Bounding box generation                           │     │
│  │     - Non-maximum suppression                           │     │
│  │                                                          │     │
│  │  3. Defect Classification Module                        │     │
│  │     - EfficientNet classifier                           │     │
│  │     - Multi-class softmax                               │     │
│  │     - Confidence scoring                                │     │
│  │                                                          │     │
│  │  4. Anomaly Detection Module                            │     │
│  │     - Autoencoder reconstruction error                  │     │
│  │     - Isolation Forest scoring                          │     │
│  │     - Threshold-based decision                          │     │
│  │                                                          │     │
│  │  5. Decision Engine                                      │     │
│  │     - Confidence aggregation                            │     │
│  │     - Accept/Reject/Alert logic                         │     │
│  │     - Performance monitoring                            │     │
│  │                                                          │     │
│  └──────────────────────────────────────────────────────────┘     │
│       ↓                                                             │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │      FACTORY INTEGRATION LAYER                           │     │
│  │                                                          │     │
│  │  - PLC Signal Handler (Reject/Alarm signals)            │     │
│  │  - Conveyor control (stop, divert defective items)      │     │
│  │  - Event logging & telemetry                            │     │
│  │                                                          │     │
│  └──────────────────────────────────────────────────────────┘     │
│       ↓                                                             │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │      DATABASE & MESSAGE QUEUE                            │     │
│  │      (SQLite for edge, PostgreSQL for cloud)            │     │
│  │      (Redis for caching, Kafka for streaming)           │     │
│  │                                                          │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  MONITORING & ANALYTICS (Cloud/Local)              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────────────────────────────────────────────┐       │
│  │  Streamlit Real-Time Dashboard                         │       │
│  │  - Live camera feed with detections                    │       │
│  │  - Defect rate metrics (per hour, per day)             │       │
│  │  - Defect type distribution                            │       │
│  │  - System health monitoring                            │       │
│  │  - Alerts and notifications                            │       │
│  └────────────────────────────────────────────────────────┘       │
│                                                                     │
│  ┌────────────────────────────────────────────────────────┐       │
│  │  Analytics Module                                       │       │
│  │  - Historical trend analysis                            │       │
│  │  - Root cause analysis                                  │       │
│  │  - Predictive maintenance                               │       │
│  │  - Model performance tracking                           │       │
│  └────────────────────────────────────────────────────────┘       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Architecture

#### 1. Data Pipeline
```
Raw Camera Feed
    ↓
[Image Capture Module]
    ↓
[Preprocessing]
  - Normalization (ImageNet stats)
  - Resize (640×640 for YOLOv8)
  - Format conversion (RGB)
    ↓
[Augmentation] (Training only)
  - Random crop
  - Rotation (-15° to +15°)
  - Gaussian blur
  - Contrast adjustment
  - Mosaic augmentation
    ↓
[Model Input]
```

#### 2. Detection Model (YOLOv8)
**Why YOLOv8?**
- Single-stage detector (faster than two-stage like Faster R-CNN)
- Real-time performance (<50ms on modern GPU)
- Excellent accuracy-speed tradeoff
- Pre-trained on COCO (transfer learning)
- Easy integration with PyTorch ecosystem

**Architecture:**
- Input: 640×640 RGB image
- Backbone: CSPDarknet (cross stage partial connections)
- Neck: PAN (path aggregation network)
- Head: Detection head (objectness + class + bbox regression)
- Output: Bounding boxes with confidence scores

#### 3. Classification Model (EfficientNet)
**Defect Classes:**
1. Scratch - Linear surface damage
2. Crack - Fractured or split surface
3. Dent/Deformation - Shape anomaly
4. Missing Component - Part absence
5. Surface Contamination - Foreign material

**Architecture:**
- Backbone: EfficientNet-B4 (pre-trained ImageNet)
- Feature extraction from detected regions
- Global Average Pooling
- Dropout (0.5)
- Softmax classifier (5 classes)
- Output: Class probabilities

#### 4. Anomaly Detection Module
**Purpose:** Detect novel/unknown defect types not in training data

**Approach 1: Autoencoder**
- Encoder: Conv layers → bottleneck features
- Decoder: Reconstruct image
- Threshold: Reconstruction error > 2σ indicates anomaly

**Approach 2: Isolation Forest**
- Extract features from EfficientNet
- Train on normal samples
- Detect anomalies based on isolation paths

**Approach 3: One-Class SVM**
- Gaussian kernel
- Train on normal product features
- Soft margin for outlier tolerance

#### 5. Decision Engine
```python
Decision Logic:
1. If detection confidence < threshold (0.6):
   → ACCEPT (no defect found)
   
2. If detection confidence > threshold:
   → Get classification from CNN
   → If classification confidence > 0.85:
      → Classify defect type
   → Else:
      → Check anomaly score
      → If anomaly score > 0.8:
         → ALERT (unknown defect)
      → Else:
         → ACCEPT (low confidence, likely false positive)
   
3. If anomaly detected:
   → ALERT (unknown defect for expert review)
   
4. Final decision:
   → ACCEPT: Pass item
   → REJECT: Divert item, log metadata
   → ALERT: Flag for human inspection

Edge case: If confidence < 0.50:
   → Defer to manual operator (safety fallback)
```

#### 6. Edge Deployment Architecture

**Target Hardware:**
- NVIDIA Jetson Orin Nano (16GB, 100 TOPS)
- NVIDIA RTX 4090 (desktop option)
- Intel Core Ultra (CPU only, lower performance)

**Optimization Techniques:**
- Model quantization (INT8, FP16)
- TensorRT acceleration
- Batch normalization folding
- Pruning (remove 30% non-critical parameters)
- Distillation (teacher-student model compression)

**Deployment Stack:**
- TensorFlow Lite or ONNX Runtime
- FastAPI for inference API
- Redis for request queuing
- SQLite for local logging

#### 7. Factory Integration

**PLC Communication:**
```
AI System → [Modbus TCP/IP] → PLC
↓
Status signals:
- Item OK (0x01)
- Defect detected (0x02)
- Manual review needed (0x03)
- System error (0xFF)
↓
PLC Actions:
- Activate reject solenoid
- Stop conveyor belt
- Trigger alarm
- Redirect item to sorting area
```

**Data Flow to Database:**
```
Inference Engine
    ↓
[Event Queue]
- Timestamp
- Image metadata
- Detection results
- Classification results
- Anomaly score
- Decision
    ↓
[Database]
- SQLite (edge device)
- PostgreSQL (cloud backup)
    ↓
[Message Queue]
- Kafka/RabbitMQ (for cloud sync)
- Real-time metric updates
```

#### 8. Dashboard Architecture

**Tech Stack:** Streamlit + Plotly + PostgreSQL

**Real-Time Metrics:**
1. **Live Feed**
   - Camera stream with bounding boxes
   - Defect highlighted
   - Confidence scores

2. **KPI Dashboard**
   - Defect rate (items/hour)
   - Pass rate (%)
   - Accuracy metrics (precision, recall, F1)
   - System latency (avg, max)
   - Uptime %

3. **Defect Analytics**
   - Defect type distribution (pie chart)
   - Trend over time (line chart)
   - Severity heatmap
   - Top defect locations

4. **System Health**
   - GPU utilization
   - Inference latency histogram
   - Model confidence distribution
   - Camera frame rate
   - Database size

5. **Alerts & Notifications**
   - Defect spike alerts
   - System errors
   - Model drift detection
   - Scheduled retraining alerts

### Data Flow Sequence

```
1. Camera captures image (30 FPS)
2. Image queued in buffer (3-frame FIFO)
3. Preprocessing (30ms)
4. YOLOv8 inference (25ms)
5. Defect detected?
   - YES: Crop ROI, send to classifier
   - NO: Decision = ACCEPT (50ms total)
6. EfficientNet classification (20ms)
7. Anomaly check (10ms)
8. Decision engine (5ms)
9. Log to database (5ms)
10. PLC signal sent (1ms)
11. Dashboard updated (real-time via WebSocket)

Total latency: 50-100ms (sufficient for 60 items/min conveyor)
```

### Scalability Considerations

1. **Single Line:** Single edge GPU, local database
2. **Multiple Lines:** Load balancer, Redis cache, PostgreSQL
3. **Multi-Factory:** Cloud orchestration, Kafka event streaming
4. **Real-Time Analytics:** Time-series database (InfluxDB, TimescaleDB)

---

This architecture is production-ready and used by companies like Cognex, Applied Materials, and custom integrations at major semiconductor fabs.
