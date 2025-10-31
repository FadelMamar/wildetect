import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Optional

import pandas as pd
import piexif
from PIL import Image, ImageOps
from tqdm import tqdm

from ..logging_config import get_logger

logger = get_logger(__name__)


def read_image(image_path: str) -> Image.Image:
    """Load an image from a file path."""
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    return image


def decimal_to_dms(decimal_degree):
    """
    Convert decimal degrees to degrees, minutes, seconds tuple for EXIF.
    Returns: (degrees, minutes, seconds)
    """
    abs_value = abs(decimal_degree)
    degrees = int(abs_value)
    minutes_float = (abs_value - degrees) * 60
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60
    return degrees, minutes, seconds


class ExifGPSManager:
    """
    Class to manage EXIF GPS data for images, including batch updates from CSV.
    """

    def __init__(self):
        pass

    def _get_gps_data(self, latitude, longitude, altitude):
        lat_dms_tuple = decimal_to_dms(abs(latitude))
        lon_dms_tuple = decimal_to_dms(abs(longitude))
        lat_dms = [
            (int(lat_dms_tuple[0]), 1),
            (int(lat_dms_tuple[1]), 1),
            (
                Fraction(lat_dms_tuple[2]).limit_denominator().numerator,
                Fraction(lat_dms_tuple[2]).limit_denominator().denominator,
            ),
        ]
        lon_dms = [
            (int(lon_dms_tuple[0]), 1),
            (int(lon_dms_tuple[1]), 1),
            (
                Fraction(lon_dms_tuple[2]).limit_denominator().numerator,
                Fraction(lon_dms_tuple[2]).limit_denominator().denominator,
            ),
        ]
        lat_ref = "N" if latitude >= 0 else "S"
        lon_ref = "E" if longitude >= 0 else "W"
        gps_data = {
            piexif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
            piexif.GPSIFD.GPSLatitude: lat_dms,
            piexif.GPSIFD.GPSLatitudeRef: lat_ref,
            piexif.GPSIFD.GPSLongitude: lon_dms,
            piexif.GPSIFD.GPSLongitudeRef: lon_ref,
        }
        alt_fraction = Fraction(abs(altitude)).limit_denominator()
        gps_data[piexif.GPSIFD.GPSAltitude] = (
            alt_fraction.numerator,
            alt_fraction.denominator,
        )
        gps_data[piexif.GPSIFD.GPSAltitudeRef] = 0 if altitude >= 0 else 1
        return gps_data

    def _get_updated_exif_dict(self, exif_dict, latitude, longitude, altitude):
        exif_dict = (
            exif_dict.copy()
            if exif_dict
            else {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        )
        exif_dict["GPS"] = self._get_gps_data(latitude, longitude, altitude)
        return exif_dict

    def add_gps_to_image(self, input_path, output_path, latitude, longitude, altitude):
        """
        Add GPS coordinates to image EXIF data
        """

        try:
            img = read_image(input_path)
            exif_dict = piexif.load(img.info.get("exif", b""))
            exif_dict = self._get_updated_exif_dict(
                exif_dict, latitude, longitude, altitude
            )
            exif_bytes = piexif.dump(exif_dict)
            img.save(output_path, exif=exif_bytes, quality=95)
            logger.debug(f"GPS data added successfully to {output_path}")
        except:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    def update_folder_from_csv(
        self,
        image_folder: str,
        csv_path: str,
        output_dir: str,
        skip_rows: int = 0,
        filename_col="filename",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        csv_data: Optional[pd.DataFrame] = None,
        max_workers: int = 3,
    ):
        """
        Update EXIF GPS data for all images in a folder using a CSV file.
        CSV must have columns: filename, latitude, longitude, [altitude]
        If inplace is False, images are written to output_dir (preserving filenames).
        """
        if csv_data is None:
            try:
                df = pd.read_csv(csv_path, skiprows=skip_rows, sep=";")
                df[filename_col]
            except Exception as e:
                df = pd.read_csv(csv_path, skiprows=skip_rows, sep=",")
                df[filename_col]
        else:
            df = csv_data

        def image_info():
            for _, row in df.iterrows():
                filename = row[filename_col]
                lat = float(row[lat_col])
                lon = float(row[lon_col])
                alt = float(row[alt_col])

                image_path = os.path.join(image_folder, filename)
                out_path = os.path.join(output_dir, filename)
                yield image_path, out_path, lat, lon, alt

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        num_images = len(df)
        with tqdm(total=num_images, unit="images") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self.add_gps_to_image,
                        input_path=image_path,
                        output_path=out_path,
                        latitude=lat,
                        longitude=lon,
                        altitude=alt,
                    )
                    for image_path, out_path, lat, lon, alt in image_info()
                ]
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)
