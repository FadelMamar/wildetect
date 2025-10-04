"""
Example usage of RasterDetectionPipeline for detecting objects in large raster files.

This example demonstrates how to use the RasterDetectionPipeline to process
large GeoTIFF or other raster files that are too large to fit in memory.

The pipeline:
1. Divides the raster into overlapping windows
2. Runs detection on each window
3. Returns detections with their spatial coordinates in the raster coordinate system
"""


from wildetect.core.config import LoaderConfig, PredictionConfig
from wildetect.core.detectors import RasterDetectionPipeline


def main():
    """Run raster detection example."""

    # Configure the loader
    loader_config = LoaderConfig(
        tile_size=640,  # Size of each raster patch
        overlap=0.2,    # 20% overlap between patches to handle edge cases
        batch_size=4,   # Process 4 patches at a time
        num_workers=0,  # Number of data loading workers
    )

    # Configure the prediction
    prediction_config = PredictionConfig(
        mlflow_model_name="detector",
        mlflow_model_alias="demo",
        device="cpu",  # Use GPU if available
        verbose=True,
    )

    # Create the pipeline
    pipeline = RasterDetectionPipeline(
        config=prediction_config,
        loader_config=loader_config,
    )

    # Path to your raster file (GeoTIFF, etc.)
    raster_path = r"C:\Users\FADELCO\Downloads\day5_ebenhazer_burnt_geier_merged_archive\transparent_day_5_ebenhazer_burnt_geier_merged_mosaic_group1.tif"

    # Path to save results
    save_path = "results/raster_detections.json"

    # Run detection
    print(f"Processing raster: {raster_path}")
    results = pipeline.run_detection(
        raster_path=raster_path,
        save_path=save_path,
        override_loading_config=True,  # Use model's recommended tile size if available
    )

    
    


if __name__ == "__main__":
    main()

