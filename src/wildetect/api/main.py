"""
FastAPI backend for WildDetect.

This module provides REST API endpoints for wildlife detection,
dataset management, and model training.
"""

import logging
import os
import shutil

# Add parent directory to path
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from wildetect.core.detector import WildlifeDetector
from wildetect.core.fiftyone_manager import FiftyOneManager
from wildetect.core.trainer import ModelTrainer
from wildetect.utils.config import create_directories, get_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="WildDetect API",
    description="Wildlife detection from aerial images",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
detector = None
fo_manager = None
trainer = None


class DetectionRequest(BaseModel):
    image_path: str
    confidence: Optional[float] = 0.5


class DetectionResponse(BaseModel):
    image_path: str
    detections: List[Dict[str, Any]]
    total_count: int
    species_counts: Dict[str, int]
    processing_time: float


class TrainingRequest(BaseModel):
    dataset_name: str
    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640


class TrainingResponse(BaseModel):
    task_id: str
    status: str
    message: str


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global detector, fo_manager, trainer

    try:
        # Create directories
        create_directories()

        # Initialize components
        detector = WildlifeDetector()
        fo_manager = FiftyOneManager()
        trainer = ModelTrainer()

        logger.info("API components initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing API components: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "WildDetect API", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "detector_ready": detector is not None,
        "fiftyone_ready": fo_manager is not None,
        "trainer_ready": trainer is not None,
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect_wildlife(request: DetectionRequest):
    """Detect wildlife in an image."""
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector not initialized")

    try:
        import time

        start_time = time.time()

        result = detector.detect(request.image_path, request.confidence)

        processing_time = time.time() - start_time

        return DetectionResponse(
            image_path=result["image_path"],
            detections=result.get("detections", []),
            total_count=result.get("total_count", 0),
            species_counts=result.get("species_counts", {}),
            processing_time=processing_time,
        )

    except Exception as e:
        logger.error(f"Error during detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/batch")
async def detect_batch(image_paths: List[str], confidence: float = 0.5):
    """Detect wildlife in multiple images."""
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector not initialized")

    try:
        results = detector.detect_batch(image_paths, confidence)
        return {"results": results}

    except Exception as e:
        logger.error(f"Error during batch detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image for processing."""
    try:
        # Create upload directory
        config = get_config()
        upload_dir = Path(config["paths"]["images_dir"])
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "filename": file.filename,
            "file_path": str(file_path),
            "size": file.size,
        }

    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dataset/info")
async def get_dataset_info():
    """Get information about the FiftyOne dataset."""
    if fo_manager is None:
        raise HTTPException(status_code=500, detail="FiftyOne manager not initialized")

    try:
        info = fo_manager.get_dataset_info()
        return info

    except Exception as e:
        logger.error(f"Error getting dataset info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dataset/stats")
async def get_dataset_stats():
    """Get dataset statistics."""
    if fo_manager is None:
        raise HTTPException(status_code=500, detail="FiftyOne manager not initialized")

    try:
        stats = fo_manager.get_annotation_stats()
        return stats

    except Exception as e:
        logger.error(f"Error getting dataset stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dataset/export")
async def export_dataset(format: str = "coco", output_path: Optional[str] = None):
    """Export dataset annotations."""
    if fo_manager is None:
        raise HTTPException(status_code=500, detail="FiftyOne manager not initialized")

    try:
        if output_path is None:
            config = get_config()
            output_path = f"{config['paths']['annotations_dir']}/export_{format}"

        fo_manager.export_annotations(output_path, format)

        return {
            "message": "Dataset exported successfully",
            "format": format,
            "output_path": output_path,
        }

    except Exception as e:
        logger.error(f"Error exporting dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", response_model=TrainingResponse)
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start model training in background."""
    if trainer is None:
        raise HTTPException(status_code=500, detail="Trainer not initialized")

    try:
        # Generate task ID
        import uuid

        task_id = str(uuid.uuid4())

        # Add training task to background
        background_tasks.add_task(
            run_training_task,
            task_id,
            request.dataset_name,
            request.epochs,
            request.batch_size,
            request.imgsz,
        )

        return TrainingResponse(
            task_id=task_id, status="started", message="Training started in background"
        )

    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_training_task(
    task_id: str, dataset_name: str, epochs: int, batch_size: int, imgsz: int
):
    """Run training task in background."""
    try:
        logger.info(f"Starting training task {task_id}")

        # Prepare training data
        config = get_config()
        data_dir = f"{config['paths']['data_dir']}/training_data_{task_id}"

        training_info = trainer.prepare_training_data(dataset_name, data_dir)
        logger.info(f"Prepared training data: {training_info}")

        # Start training
        data_path = f"{data_dir}/dataset.yaml"
        results = trainer.train_model(
            data_path=data_path, epochs=epochs, batch_size=batch_size, imgsz=imgsz
        )

        logger.info(f"Training completed for task {task_id}: {results}")

    except Exception as e:
        logger.error(f"Error in training task {task_id}: {e}")


@app.get("/model/info")
async def get_model_info():
    """Get information about the current model."""
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector not initialized")

    try:
        info = detector.get_model_info()
        return info

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_configuration():
    """Get current configuration."""
    try:
        config = get_config()
        return config

    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    config = get_config()
    uvicorn.run(
        "wildetect.api.main:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=config["api"]["reload"],
    )
