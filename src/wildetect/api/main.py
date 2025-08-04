"""
FastAPI backend for WildDetect.

This module provides REST API endpoints for wildlife detection,
dataset management, and model training.
"""

import json
import logging
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from uuid import uuid4

import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ..core.campaign_manager import CampaignConfig, CampaignManager
from ..core.config import ROOT, FlightSpecs, LoaderConfig, PredictionConfig
from ..core.data.census import CensusDataManager
from ..core.data.drone_image import DroneImage
from ..core.data.utils import get_images_paths
from ..core.detection_pipeline import DetectionPipeline, MultiThreadedDetectionPipeline
from ..core.visualization.fiftyone_manager import FiftyOneManager
from ..core.visualization.geographic import (
    GeographicVisualizer,
    VisualizationConfig,
)

# Create FastAPI app
app = FastAPI(
    title="WildDetect API",
    description="REST API for WildDetect - Wildlife Detection System",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Pydantic models for request/response
class DetectionRequest(BaseModel):
    model_path: Optional[str] = None
    dataset_name: Optional[str] = None
    model_type: str = "yolo"
    confidence: float = 0.2
    device: str = "auto"
    batch_size: int = 32
    tile_size: int = 800
    output: Optional[str] = "results"
    roi_weights: Optional[str] = None
    feature_extractor_path: str = "facebook/dinov2-with-registers-small"
    cls_label_map: List[str] = ["0:groundtruth", "1:other"]
    keep_classes: List[str] = ["groundtruth"]
    cls_imgsz: int = 128
    inference_service_url: Optional[str] = None
    nms_iou: float = 0.5
    overlap_ratio: float = 0.2
    sensor_height: float = 24.0
    focal_length: float = 35.0
    flight_height: float = 180.0
    pipeline_type: str = "single"
    queue_size: int = 3


class CensusRequest(BaseModel):
    campaign_id: str
    model_path: Optional[str] = None
    roi_weights: Optional[str] = None
    feature_extractor_path: str = "facebook/dinov2-with-registers-small"
    cls_label_map: List[str] = ["0:groundtruth", "1:other"]
    keep_classes: List[str] = ["groundtruth"]
    confidence: float = 0.2
    device: str = "auto"
    inference_service_url: Optional[str] = None
    nms_iou: float = 0.5
    overlap_ratio: float = 0.2
    batch_size: int = 32
    tile_size: int = 800
    cls_imgsz: int = 96
    sensor_height: float = 24.0
    focal_length: float = 35.0
    flight_height: float = 180.0
    output: Optional[str] = None
    pilot_name: Optional[str] = None
    target_species: Optional[List[str]] = None
    export_to_fiftyone: bool = True
    create_map: bool = True
    pipeline_type: str = "single"
    queue_size: int = 3


class VisualizationRequest(BaseModel):
    show_confidence: bool = False


class AnalysisRequest(BaseModel):
    create_map: bool = True


class SystemInfo(BaseModel):
    pytorch_version: str
    cuda_available: bool
    cuda_device: Optional[str] = None
    dependencies: Dict[str, bool]


class DetectionResponse(BaseModel):
    job_id: str
    status: str
    message: str
    results_path: Optional[str] = None
    total_images: Optional[int] = None
    total_detections: Optional[int] = None
    species_counts: Optional[Dict[str, int]] = None


class CensusResponse(BaseModel):
    job_id: str
    status: str
    message: str
    campaign_id: str
    results_path: Optional[str] = None
    statistics: Optional[Dict] = None


# Global storage for job status
job_status = {}


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stdout)]

    if isinstance(log_file, str):
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def parse_cls_label_map(cls_label_map: List[str]) -> Dict[int, str]:
    """Parse a list of key:value strings into a dictionary with int keys and str values."""
    result = {}
    for item in cls_label_map:
        key, value = item.split(":", 1)
        result[int(key)] = value
    return result


def get_detection_statistics(drone_images: List[DroneImage]) -> dict:
    """Calculate detection statistics from drone images."""
    total_detections = 0
    species_counts = {}

    for drone_image in drone_images:
        detections = drone_image.get_non_empty_predictions()
        total_detections += len(detections)

        for detection in detections:
            species = detection.class_name
            species_counts[species] = species_counts.get(species, 0) + 1

    return {
        "total_detections": total_detections,
        "species_counts": species_counts,
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "WildDetect API",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "detect": "/detect",
            "census": "/census",
            "visualize": "/visualize",
            "analyze": "/analyze",
            "info": "/info",
            "upload": "/upload",
            "jobs": "/jobs/{job_id}",
        },
    }


@app.get("/info")
async def get_system_info():
    """Get system information."""
    info = SystemInfo(
        pytorch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_device=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        dependencies={},
    )

    # Check key dependencies
    dependencies = [
        "fiftyone",
        "folium",
        "geopy",
        "huggingface",
        "pillow",
        "pyproj",
        "shapely",
        "ultralytics",
        "torch",
        "rich",
        "typer",
    ]

    for dep in dependencies:
        try:
            __import__(dep)
            info.dependencies[dep] = True
        except ImportError:
            info.dependencies[dep] = False

    return info


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload image files for processing."""
    upload_dir = Path(ROOT) / "uploads" / str(uuid4())
    upload_dir.mkdir(parents=True, exist_ok=True)

    uploaded_files = []
    for file in files:
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail=f"File {file.filename} is not an image"
            )

        if file.filename:
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            uploaded_files.append(str(file_path))

    return {
        "message": f"Uploaded {len(uploaded_files)} files",
        "upload_dir": str(upload_dir),
        "files": uploaded_files,
    }


@app.post("/detect")
async def detect_wildlife(request: DetectionRequest, background_tasks: BackgroundTasks):
    """Run wildlife detection on images."""
    job_id = str(uuid4())
    job_status[job_id] = {
        "status": "running",
        "message": "Starting detection...",
        "progress": 0,
    }

    def run_detection():
        try:
            setup_logging(
                log_file=str(
                    ROOT
                    / f"logs/detect_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                )
            )
            logger = logging.getLogger(__name__)

            job_status[job_id]["message"] = "Setting up detection pipeline..."

            # Create flight specs
            flight_specs = FlightSpecs(
                sensor_height=request.sensor_height,
                focal_length=request.focal_length,
                flight_height=request.flight_height,
            )

            # Parse cls_label_map
            label_map_dict = parse_cls_label_map(request.cls_label_map)

            for class_name in request.keep_classes:
                if class_name not in label_map_dict.values():
                    raise ValueError(f"Class {class_name} not found in cls_label_map")

            # Create configurations
            pred_config = PredictionConfig(
                model_path=request.model_path,
                model_type=request.model_type,
                confidence_threshold=request.confidence,
                device=request.device,
                batch_size=request.batch_size,
                tilesize=request.tile_size,
                flight_specs=flight_specs,
                roi_weights=request.roi_weights,
                cls_imgsz=request.cls_imgsz,
                keep_classes=request.keep_classes,
                feature_extractor_path=request.feature_extractor_path,
                cls_label_map=label_map_dict,
                inference_service_url=request.inference_service_url,
                verbose=True,
                nms_iou=request.nms_iou,
                overlap_ratio=request.overlap_ratio,
                pipeline_type=request.pipeline_type,
                queue_size=request.queue_size,
            )

            loader_config = LoaderConfig(
                tile_size=request.tile_size,
                batch_size=request.batch_size,
                num_workers=0,
                flight_specs=flight_specs,
            )

            job_status[job_id]["message"] = "Creating detection pipeline..."

            # Create detection pipeline
            if pred_config.pipeline_type == "multi":
                pipeline = MultiThreadedDetectionPipeline(
                    config=pred_config,
                    loader_config=loader_config,
                    queue_size=pred_config.queue_size,
                )
            else:
                pipeline = DetectionPipeline(
                    config=pred_config,
                    loader_config=loader_config,
                )

            save_path = None
            if request.output:
                output_path = Path(request.output)
                output_path.mkdir(parents=True, exist_ok=True)
                save_path = str(output_path / "results.json")

            job_status[job_id]["message"] = "Running detection..."
            job_status[job_id]["progress"] = 50

            # Run detection
            drone_images = pipeline.run_detection(
                image_paths=[],  # Will be set by upload endpoint
                image_dir=None,
                save_path=save_path,
            )

            # Calculate statistics
            stats = get_detection_statistics(drone_images)

            job_status[job_id]["status"] = "completed"
            job_status[job_id]["message"] = "Detection completed successfully"
            job_status[job_id]["progress"] = 100
            job_status[job_id]["results"] = {
                "results_path": save_path,
                "total_images": len(drone_images),
                "total_detections": stats["total_detections"],
                "species_counts": stats["species_counts"],
            }

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["message"] = f"Detection failed: {str(e)}"
            job_status[job_id]["progress"] = 0

    background_tasks.add_task(run_detection)

    return DetectionResponse(
        job_id=job_id,
        status="running",
        message="Detection job started",
    )


@app.post("/census")
async def run_census_campaign(
    request: CensusRequest, background_tasks: BackgroundTasks
):
    """Run wildlife census campaign."""
    job_id = str(uuid4())
    job_status[job_id] = {
        "status": "running",
        "message": "Starting census campaign...",
        "progress": 0,
    }

    def run_census():
        try:
            setup_logging(
                log_file=str(
                    ROOT
                    / f"logs/census_api_{request.campaign_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                )
            )
            logger = logging.getLogger(__name__)

            job_status[job_id]["message"] = "Setting up census campaign..."

            flight_specs = FlightSpecs(
                sensor_height=request.sensor_height,
                focal_length=request.focal_length,
                flight_height=request.flight_height,
            )

            # Parse cls_label_map
            label_map_dict = parse_cls_label_map(request.cls_label_map)
            for class_name in request.keep_classes:
                if class_name not in label_map_dict.values():
                    raise ValueError(f"Class {class_name} not found in cls_label_map")

            # Create campaign metadata
            campaign_metadata = {
                "pilot_info": {
                    "name": request.pilot_name or "Unknown",
                    "experience": "Unknown",
                },
                "weather_conditions": {},
                "mission_objectives": ["wildlife_survey"],
                "target_species": request.target_species,
                "flight_parameters": vars(flight_specs),
                "equipment_info": {},
            }

            # Create configurations
            pred_config = PredictionConfig(
                model_path=request.model_path,
                confidence_threshold=request.confidence,
                device=request.device,
                batch_size=request.batch_size,
                tilesize=request.tile_size,
                flight_specs=flight_specs,
                roi_weights=request.roi_weights,
                cls_imgsz=request.cls_imgsz,
                keep_classes=request.keep_classes,
                feature_extractor_path=request.feature_extractor_path,
                cls_label_map=label_map_dict,
                inference_service_url=request.inference_service_url,
                verbose=True,
                nms_iou=request.nms_iou,
                overlap_ratio=request.overlap_ratio,
                pipeline_type=request.pipeline_type,
                queue_size=request.queue_size,
            )

            loader_config = LoaderConfig(
                tile_size=request.tile_size,
                batch_size=request.batch_size,
                num_workers=0,
                flight_specs=flight_specs,
            )

            # Create campaign configuration
            campaign_config = CampaignConfig(
                campaign_id=request.campaign_id,
                loader_config=loader_config,
                prediction_config=pred_config,
                metadata=campaign_metadata,
                fiftyone_dataset_name=f"campaign_{request.campaign_id}",
            )

            job_status[job_id]["message"] = "Initializing campaign manager..."
            job_status[job_id]["progress"] = 25

            # Initialize campaign manager
            campaign_manager = CampaignManager(campaign_config)

            job_status[job_id]["message"] = "Running census campaign..."
            job_status[job_id]["progress"] = 50

            # Run complete campaign
            results = campaign_manager.run_complete_campaign(
                image_paths=[],  # Will be set by upload endpoint
                output_dir=request.output,
                export_to_fiftyone=request.export_to_fiftyone,
            )

            job_status[job_id]["status"] = "completed"
            job_status[job_id]["message"] = "Census campaign completed successfully"
            job_status[job_id]["progress"] = 100
            job_status[job_id]["results"] = {
                "campaign_id": request.campaign_id,
                "statistics": results["statistics"],
                "output_dir": request.output,
            }

        except Exception as e:
            logger.error(f"Census campaign failed: {e}")
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["message"] = f"Census campaign failed: {str(e)}"
            job_status[job_id]["progress"] = 0

    background_tasks.add_task(run_census)

    return CensusResponse(
        job_id=job_id,
        status="running",
        message="Census campaign started",
        campaign_id=request.campaign_id,
    )


@app.post("/visualize")
async def visualize_results(
    results_path: str = Form(...), request: VisualizationRequest = Form(...)
):
    """Visualize detection results."""
    try:
        results_path_obj = Path(results_path)
        if not results_path_obj.exists():
            raise HTTPException(
                status_code=404, detail=f"Results file not found: {results_path}"
            )

        # Load results
        with open(results_path_obj, "r") as f:
            results = json.load(f)

        # Create output directory
        output_dir = Path(
            "visualizations"
        ) / f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create geographic visualization if data available
        if isinstance(results, dict) and "drone_images" in results:
            drone_images = results["drone_images"]
            config = VisualizationConfig()
            visualizer = GeographicVisualizer(config)
            map_file = str(output_dir / "geographic_visualization.html")
            visualizer.create_map(drone_images, map_file)

            return {
                "message": "Visualization created successfully",
                "map_file": map_file,
                "output_dir": str(output_dir),
            }
        else:
            raise HTTPException(
                status_code=400, detail="No geographic data found in results"
            )

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


@app.post("/analyze")
async def analyze_results(
    results_path: str = Form(...), request: AnalysisRequest = Form(...)
):
    """Analyze detection results."""
    try:
        results_path_obj = Path(results_path)
        if not results_path_obj.exists():
            raise HTTPException(
                status_code=404, detail=f"Results file not found: {results_path}"
            )

        # Load results
        with open(results_path_obj, "r") as f:
            results = json.load(f)

        # Create output directory
        output_dir = Path(
            "analysis"
        ) / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Analyze results
        analysis = {
            "total_images": 0,
            "total_detections": 0,
            "species_breakdown": {},
            "geographic_coverage": {},
        }

        if isinstance(results, list):
            analysis["total_images"] = len(results)
            for result in results:
                if "total_detections" in result:
                    analysis["total_detections"] += result["total_detections"]
                if "class_counts" in result:
                    for species, count in result["class_counts"].items():
                        analysis["species_breakdown"][species] = (
                            analysis["species_breakdown"].get(species, 0) + count
                        )

        # Save analysis report
        report_file = output_dir / "analysis_report.json"
        with open(report_file, "w") as f:
            json.dump(analysis, f, indent=2)

        return {
            "message": "Analysis completed successfully",
            "analysis": analysis,
            "report_file": str(report_file),
            "output_dir": str(output_dir),
        }

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a background job."""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")

    return job_status[job_id]


@app.get("/fiftyone/launch")
async def launch_fiftyone():
    """Launch FiftyOne app."""
    try:
        FiftyOneManager.launch_app()
        return {"message": "FiftyOne app launched successfully"}
    except Exception as e:
        logger.error(f"Failed to launch FiftyOne: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to launch FiftyOne: {str(e)}"
        )


@app.get("/fiftyone/datasets/{dataset_name}")
async def get_dataset_info(dataset_name: str):
    """Get FiftyOne dataset information."""
    try:
        fo_manager = FiftyOneManager(dataset_name)
        dataset_info = fo_manager.get_dataset_info()
        annotation_stats = fo_manager.get_annotation_stats()

        return {"dataset_info": dataset_info, "annotation_stats": annotation_stats}
    except Exception as e:
        logger.error(f"Failed to get dataset info: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get dataset info: {str(e)}"
        )


@app.post("/fiftyone/export/{dataset_name}")
async def export_dataset(
    dataset_name: str,
    export_format: str = Form("coco"),
    export_path: Optional[str] = Form(None),
):
    """Export FiftyOne dataset."""
    try:
        if not export_path:
            export_path = f"exports/{dataset_name}_{export_format}"

        fo_manager = FiftyOneManager(dataset_name)
        Path(export_path).mkdir(parents=True, exist_ok=True)

        if fo_manager.dataset:
            fo_manager.dataset.export(
                export_dir=export_path, dataset_type=export_format, overwrite=True
            )
            return {
                "message": "Dataset exported successfully",
                "export_path": export_path,
            }
        else:
            raise HTTPException(status_code=500, detail="Dataset not initialized")

    except Exception as e:
        logger.error(f"Failed to export dataset: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to export dataset: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
