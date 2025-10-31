import os
import sys
from pathlib import Path
from wildata.adapters.utils import ExifGPSManager
from wildata.partitioning.utils import GPSUtils
from wildata.config import ROOT
import numpy as np
import pandas as pd


def example_batch_update():
    # Example usage: update EXIF GPS for all images in a folder using a CSV
    image_folder = r"D:\workspace\data\savmap_dataset_v2\images_splits"

    max_images = 20

    image_paths = [p for p in Path(image_folder).glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]][:]
    random_lat = np.random.uniform(40.68, 40.69, size=len(image_paths)).round(6)
    random_lon = np.random.uniform(-74.04, -74.05, size=len(image_paths)).round(6)
    random_alt = np.random.uniform(300.0, 301.0, size=len(image_paths)).round(6)

    mock_csv = dict()
    mock_csv["filename"] = [p.name for p in image_paths]
    mock_csv["latitude"] = random_lat
    mock_csv["longitude"] = random_lon
    mock_csv["altitude"] = random_alt

    df_mock_csv = pd.DataFrame.from_dict(mock_csv,orient="columns")

    df_mock_csv.to_csv(ROOT / "examples" / "mock_csv.csv", index=False)

    csv_path = "None"  # Change to your CSV file
    output_dir = ROOT / "data" / "savmap_dataset_v2_splits_with_gps"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the manager
    manager = ExifGPSManager()

    # Update all images in the folder using the CSV
    # The CSV must have columns: filename, latitude, longitude, [altitude]
    # Set inplace=True to modify images in place, or False to overwrite
    manager.update_folder_from_csv(
        image_folder=image_folder,
        csv_path=csv_path,
        output_dir=str(output_dir),
        csv_data=df_mock_csv,
        skip_rows=0,
        filename_col="filename",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude"  # Optional, only if your CSV has altitude
    )
    print(f"EXIF GPS update complete for images in {image_folder} using {csv_path}")

def example_single_image_update():
    image_path = r"D:\workspace\data\savmap_dataset_v2\images_splits\00a033fefe644429a1e0fcffe88f8b39_1.JPG"
    output_path = "image_with_gps.jpg"
    manager = ExifGPSManager()
    manager.add_gps_to_image(
        input_path=image_path,
        output_path=output_path,
        latitude=40.689247,
        longitude=-74.044502,
        altitude=300.0
    )
    print("GPS image without gps: ", GPSUtils.get_gps_coord(image_path))
    print("GPS coordinates from image with gps: ", GPSUtils.get_gps_coord(output_path))


if __name__ == "__main__":
    # main() 
    example_batch_update()
    #example_single_image_update()