# Smart Industrial Defect Detection - FastAPI Inference Server
"""
REST API server for defect detection inference.
"""

import os
import io
import time
import base64
import numpy as np
import cv2
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# Pydantic models for API
class DefectInfo(BaseModel):
    bbox: List[float]
    detection_confidence: float
    detection_class: str
    classification: Optional[str]
    classification_confidence: float


class InferenceResponse(BaseModel):
    image_id: str
    timestamp: float
    decision: str
    defects: List[DefectInfo]
    anomaly_score: float
    confidence: float
    latency_ms: float
    annotated_image_base64: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
    latency_stats: dict
    uptime_seconds: float


class BatchInferenceRequest(BaseModel):
    images_base64: List[str]
    image_ids: Optional[List[str]] = None
    return_annotated: bool = False


# Global variables
inference_pipeline = None
start_time = None


def create_app(detection_model_path: str = None,
               classification_model_path: str = None,
               anomaly_model_path: str = None,
               device: str = 'cuda:0') -> FastAPI:
    """
    Create FastAPI application for inference.
    
    Args:
        detection_model_path: Path to detection model
        classification_model_path: Path to classification model
        anomaly_model_path: Path to anomaly model
        device: Device to run on
    
    Returns:
        FastAPI application
    """
    global inference_pipeline, start_time
    
    app = FastAPI(
        title="Smart Industrial Defect Detection API",
        description="Real-time AI-powered defect detection for manufacturing",
        version="1.0.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize models on startup."""
        global inference_pipeline, start_time
        
        from src.inference.pipeline import InferencePipeline
        
        logger.info("Initializing inference pipeline...")
        
        inference_pipeline = InferencePipeline(
            detection_model_path=detection_model_path,
            classification_model_path=classification_model_path,
            anomaly_model_path=anomaly_model_path,
            device=device
        )
        
        # Warmup
        inference_pipeline.warmup(num_iterations=5)
        
        start_time = time.time()
        logger.info("Inference server ready")
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            models_loaded={
                "detector": inference_pipeline.detector_available if inference_pipeline else False,
                "classifier": inference_pipeline.classifier_available if inference_pipeline else False,
                "anomaly": inference_pipeline.anomaly_available if inference_pipeline else False
            },
            latency_stats=inference_pipeline.get_latency_stats() if inference_pipeline else {},
            uptime_seconds=time.time() - start_time if start_time else 0
        )
    
    @app.post("/predict", response_model=InferenceResponse)
    async def predict(file: UploadFile = File(...),
                      return_annotated: bool = True):
        """
        Run inference on uploaded image.
        
        Args:
            file: Image file
            return_annotated: Include annotated image in response
        
        Returns:
            Inference results
        """
        if inference_pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            # Read image
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image file")
            
            # Run inference
            result = inference_pipeline.predict(
                image,
                image_id=file.filename,
                return_annotated=return_annotated
            )
            
            # Encode annotated image to base64
            annotated_base64 = None
            if return_annotated and result.annotated_image is not None:
                _, buffer = cv2.imencode('.jpg', result.annotated_image)
                annotated_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return InferenceResponse(
                image_id=result.image_id,
                timestamp=result.timestamp,
                decision=result.decision,
                defects=[DefectInfo(**d) for d in result.defects],
                anomaly_score=result.anomaly_score,
                confidence=result.confidence,
                latency_ms=result.latency_ms,
                annotated_image_base64=annotated_base64
            )
        
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict/batch")
    async def predict_batch(request: BatchInferenceRequest):
        """
        Run batch inference on multiple images.
        
        Args:
            request: Batch inference request
        
        Returns:
            List of inference results
        """
        if inference_pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            results = []
            
            for i, img_base64 in enumerate(request.images_base64):
                # Decode base64 image
                img_bytes = base64.b64decode(img_base64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    continue
                
                # Get image ID
                image_id = request.image_ids[i] if request.image_ids else f"batch_img_{i}"
                
                # Run inference
                result = inference_pipeline.predict(
                    image,
                    image_id=image_id,
                    return_annotated=request.return_annotated
                )
                
                # Prepare response
                annotated_base64 = None
                if request.return_annotated and result.annotated_image is not None:
                    _, buffer = cv2.imencode('.jpg', result.annotated_image)
                    annotated_base64 = base64.b64encode(buffer).decode('utf-8')
                
                results.append({
                    "image_id": result.image_id,
                    "timestamp": result.timestamp,
                    "decision": result.decision,
                    "defects": result.defects,
                    "anomaly_score": result.anomaly_score,
                    "confidence": result.confidence,
                    "latency_ms": result.latency_ms,
                    "annotated_image_base64": annotated_base64
                })
            
            return {"results": results, "total": len(results)}
        
        except Exception as e:
            logger.error(f"Batch inference error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/stats")
    async def get_stats():
        """Get inference statistics."""
        if inference_pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        return {
            "latency_stats": inference_pipeline.get_latency_stats(),
            "models": {
                "detector": inference_pipeline.detector_available,
                "classifier": inference_pipeline.classifier_available,
                "anomaly": inference_pipeline.anomaly_available
            },
            "thresholds": {
                "detection": inference_pipeline.detection_threshold,
                "classification": inference_pipeline.classification_threshold,
                "anomaly": inference_pipeline.anomaly_threshold
            }
        }
    
    @app.post("/config/thresholds")
    async def update_thresholds(detection: float = None,
                                 classification: float = None,
                                 anomaly: float = None):
        """Update detection thresholds."""
        if inference_pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if detection is not None:
            inference_pipeline.detection_threshold = detection
        if classification is not None:
            inference_pipeline.classification_threshold = classification
        if anomaly is not None:
            inference_pipeline.anomaly_threshold = anomaly
        
        return {
            "detection_threshold": inference_pipeline.detection_threshold,
            "classification_threshold": inference_pipeline.classification_threshold,
            "anomaly_threshold": inference_pipeline.anomaly_threshold
        }
    
    return app


def run_server(host: str = "0.0.0.0",
               port: int = 8000,
               detection_model_path: str = None,
               classification_model_path: str = None,
               anomaly_model_path: str = None,
               device: str = 'cuda:0'):
    """
    Run inference server.
    
    Args:
        host: Server host
        port: Server port
        detection_model_path: Path to detection model
        classification_model_path: Path to classification model
        anomaly_model_path: Path to anomaly model
        device: Device to run on
    """
    import uvicorn
    
    app = create_app(
        detection_model_path=detection_model_path,
        classification_model_path=classification_model_path,
        anomaly_model_path=anomaly_model_path,
        device=device
    )
    
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Inference Server')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--detection_model', type=str, default=None)
    parser.add_argument('--classification_model', type=str, default=None)
    parser.add_argument('--anomaly_model', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    run_server(
        host=args.host,
        port=args.port,
        detection_model_path=args.detection_model,
        classification_model_path=args.classification_model,
        anomaly_model_path=args.anomaly_model,
        device=args.device
    )
