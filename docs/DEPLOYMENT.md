# API Deployment Guide

## 🚀 Deployment Options

### 1. Local Development

```bash
# Start the API server
python -m src.inference.inference_server --port 8000

# Test with curl
curl http://localhost:8000/health
```

### 2. Docker Deployment

```bash
cd docker
docker-compose up -d
```

### 3. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: defect-detection-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: defect-detection
  template:
    metadata:
      labels:
        app: defect-detection
    spec:
      containers:
      - name: api
        image: defect-detection:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: defect-detection-service
spec:
  selector:
    app: defect-detection
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 📡 API Endpoints

### Health Check
```
GET /health
```

### Single Image Prediction
```
POST /predict
Content-Type: multipart/form-data
Body: image file
```

### Batch Prediction
```
POST /predict/batch
Content-Type: multipart/form-data
Body: multiple image files
```

### Statistics
```
GET /stats
```

### Update Thresholds
```
POST /config/thresholds
Content-Type: application/json
Body: {"detection": 0.5, "classification": 0.85, "anomaly": 2.0}
```

## 🔧 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEVICE` | Inference device | `cuda:0` |
| `PORT` | API port | `8000` |
| `WORKERS` | Number of workers | `1` |
| `LOG_LEVEL` | Logging level | `INFO` |

## 📊 Monitoring

### Health Checks
- Endpoint: `/health`
- Interval: 30s
- Timeout: 10s

### Metrics (Prometheus)
```python
# Add to inference_server.py for production
from prometheus_client import Counter, Histogram

PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')
```

### Logging
- All predictions logged to database
- Alerts sent for critical issues
- Dashboard provides real-time visualization
