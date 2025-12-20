from typing import Optional, Tuple

import geopy
from PIL import Image

from ..adapters.utils import read_image
from ..logging_config import get_logger

logger = get_logger(__name__)


def get_timestamp(image: Image.Image) -> str:
    timestamp = image._getexif()[36867]
    return timestamp


class GPSUtils:
    # Sensor height mapping for common camera models
    SENSOR_HEIGHTS = {
        "FC7303": 24.0,  # DJI Phantom 4 Pro
        "FC6310": 24.0,  # DJI Phantom 4
        "FC220": 24.0,  # DJI Spark
        "FC330": 24.0,  # DJI Mavic Pro
        "default": 24.0,  # Default sensor height,
        "ZenmuseP1": 24.0,
    }

    @staticmethod
    def get_exif(file_name: str, image: Optional[Image.Image] = None) -> Optional[dict]:
        from PIL.ExifTags import TAGS

        if image is None:
            img = read_image(file_name)
            exif_data = img._getexif()  # type: ignore
        else:
            assert isinstance(image, Image.Image), "Provide PIL Image"
            try:
                exif_data = image._getexif()  # type: ignore
            except AttributeError:
                return None
            except Exception as e:
                logger.warning(f"Error getting exif: {e}")
                raise Exception(f"Error getting exif: {e}")

        if exif_data is None:
            return None

        extracted_exif = dict()
        for k, v in exif_data.items():
            extracted_exif[TAGS.get(k)] = v

        return extracted_exif

    @staticmethod
    def get_gps_info(labeled_exif: dict) -> Optional[dict]:
        # https://exiftool.org/TagNames/GPS.html
        from PIL.ExifTags import GPSTAGS

        assert isinstance(labeled_exif, dict), "Provide dict"

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
    def get_gps_coord(
        file_name: str,
        image: Optional[Image.Image] = None,
    ) -> Optional[Tuple[float, float]]:
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

        coord = geopy.Point.from_string(
            gps_coords["GPSLatitude"] + " " + gps_coords["GPSLongitude"]
        )
        return coord.latitude, coord.longitude
