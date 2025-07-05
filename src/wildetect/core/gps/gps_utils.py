"""
GPS utilities for wildlife detection with optional dependencies.
"""

import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from ..config import FlightSpecs

logger = logging.getLogger(__name__)

# Optional imports with graceful fallbacks
try:
    import utm
    UTM_AVAILABLE = True
except ImportError:
    UTM_AVAILABLE = False
    logger.warning("UTM library not available. GPS coordinate conversion will be limited.")

try:
    import geopy
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    logger.warning("Geopy library not available. GPS parsing will be limited.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL library not available. Image EXIF extraction will be limited.")


def get_pixel_gps_coordinates(
        x,
        y,
        lat_center,
        lon_center,
        W,
        H,
        gsd: float = 2.6,
        return_as_utm: bool = False,
    ) -> tuple[float, float]:
    """computes (x,y) pixel gps coordinates

    Args:
        x (int): x center
        y (int): y center
        lat_center (float): latitude of center of roi or image
        lon_center (float): longitude of center of roi or image
        W (int): roi or image width
        H (int): roi or image height
        gsd (float, optional): in cm/px. Defaults to 2.6.

    Returns:
        tuple: latitude, longitude
    """
    # Convert center to UTM
    easting_center, northing_center, zone_num, zone_let = utm.from_latlon(
        lat_center, lon_center
    )

    gsd *= 1e-2  # convert to m/px

    # Calculate offsets
    delta_x = (x - W / 2) * gsd
    delta_y = (H / 2 - y) * gsd  # Invert y-axis

    # Compute UTM
    easting = easting_center + delta_x
    northing = northing_center + delta_y

    if return_as_utm:
        return easting, northing
        # Convert back to lat/lon
    else:
        try:
            lat, lon = utm.to_latlon(easting, northing, zone_num, zone_let)
        except Exception as e:
            raise ValueError(
                f"Invalid UTM coordinates: {easting}, {northing}, {zone_num}, {zone_let}"
                + f"or Invalid input values {x}, {y}, {lat_center}, {lon_center}, {W}, {H}, {gsd}."
            )

        return lat, lon


class GPSUtils:

    # Sensor height mapping for common camera models
    SENSOR_HEIGHTS = {
        "FC7303": 24.0,  # DJI Phantom 4 Pro
        "FC6310": 24.0,  # DJI Phantom 4
        "FC220": 24.0,   # DJI Spark
        "FC330": 24.0,   # DJI Mavic Pro
        "default": 24.0  # Default sensor height,
        "ZenmuseP1":24.0
    }
    
    @staticmethod
    def get_exif(file_name: str, image: Image = None) -> dict | None:
        from PIL.ExifTags import TAGS

        if image is None:
            with Image.open(file_name) as img:
                exif_data = img._getexif()
        else:
            assert isinstance(image, Image.Image), "Provide PIL Image"
            exif_data = image._getexif()

        if exif_data is None:
            return None

        extracted_exif = dict()
        for k, v in exif_data.items():
            extracted_exif[TAGS.get(k)] = v

        return extracted_exif

    @staticmethod
    def get_gps_info(labeled_exif: dict) -> dict | None:
        # https://exiftool.org/TagNames/GPS.html
        from PIL.ExifTags import GPSTAGS

        gps_info = labeled_exif.get("GPSInfo", None)

        if gps_info is None:
            return None

        info = {GPSTAGS.get(key, key): value for key, value in gps_info.items()}

        info["GPSAltitude"] = info["GPSAltitude"].__repr__()

        # convert bytes types
        for k, v in info.items():
            if isinstance(v, bytes):
                info[k] = list(v)

        return info

    @staticmethod
    def to_decimal(gps_coord: str) -> tuple:
        if gps_coord is None:
            return (None, None, None)

        lat, long, alt = geopy.Point.from_string(gps_coord)
        coords = lat, long, alt * 1e3
        return coords

    @staticmethod
    def get_gps_coord(
        file_name: str,
        image: Image = None,
        altitude: str = None,
        return_as_decimal: bool = False,
    ) -> tuple | None:
        extracted_exif = GPSUtils.get_exif(file_name=file_name, image=image)

        if extracted_exif is None:
            return None

        gps_info = GPSUtils.get_gps_info(extracted_exif)

        if gps_info is None:
            return None

        if gps_info.get("GPSAltitudeRef", None):
            altitude_map = {
                0: "Above Sea Level",
                1: "Below Sea Level",
                2: "Positive Sea Level (sea-level ref)",
                3: "Negative Sea Level (sea-level ref)",
            }

            # map GPSAltitudeRef
            try:
                gps_info["GPSAltitudeRef"] = altitude_map[gps_info["GPSAltitudeRef"]]
            except:
                gps_info["GPSAltitudeRef"] = altitude_map[gps_info["GPSAltitudeRef"][0]]

        # rewite latitude
        gps_coords = dict()
        for coord in ["GPSLatitude", "GPSLongitude"]:
            degrees, minutes, seconds = gps_info[coord]
            ref = gps_info[coord + "Ref"]
            gps_coords[coord] = f"{degrees} {minutes}m {seconds}s {ref}"

        coords = gps_coords["GPSLatitude"] + " " + gps_coords["GPSLongitude"]

        if altitude is None:
            alt = f"{gps_info.get('GPSAltitude', None)}m"
        else:
            alt = altitude

        if alt:
            coords = (
                gps_coords["GPSLatitude"] + " " + gps_coords["GPSLongitude"] + " " + alt
            )
        else:
            coords = gps_coords["GPSLatitude"] + " " + gps_coords["GPSLongitude"]
        if return_as_decimal:
            lat, long, alt = geopy.Point.from_string(coords)
            coords = lat, long, alt * 1e3

        return coords, gps_info


def get_gsd(
        image_path: str,
        flight_specs: FlightSpecs,
        image: Image.Image | None = None
    ):
    ##-- Extract exif
    exif = GPSUtils.get_exif(file_name=image_path, image=image)

    if image:
        _, image_height = image.size
    else:
        try:
            image_height = exif["ExifImageHeight"]
        except:
            image_height = Image.open(image_path).size[1]

    if flight_specs.focal_length is None:
        focal_length = exif["FocalLength"]

    if flight_specs.sensor_height is None:
        sensor_height = GPSUtils.SENSOR_HEIGHTS.get(exif["Model"])
        if sensor_height is None:
            raise ValueError("Sensor height not found. Please provide it.")
    else:
        assert isinstance(flight_specs.sensor_height, float) or isinstance(flight_specs.sensor_height, int), (
            f"Received {flight_specs.sensor_height}"
        )

    ##-- Compute gsd
    flight_height = flight_specs.flight_height * 1e2 # in cm
    focal_length *= 0.1  # in cm
    sensor_height *= 0.1  # in cm

    # in cm/px
    gsd = (flight_height * sensor_height) / (focal_length * image_height)

    return round(gsd, 3)