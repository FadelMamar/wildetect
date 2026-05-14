
"""# match saved tiles with their GPS coordinates
#### Meanings of arguments
- ```-ratioheight``` : proportion of tile  w.r.t height of image. Example 0.5 means dividing the image in two bands w.r.t height.
- ```-ratiowidth``` : proportion of tile w.r.t to width of image. Example 1.0 means the width of the tile is the same as the image.
- ```-overlapfactor``` : percentage of overlap. It should be less than 1.
- ```-rmheight``` : percentage of height to remove or crop at bottom and top
- ```-rmwidth``` : percentage of width to remove or crop on each side of the image
- ```-pattern``` : "**/*.JPG" will get all .JPG images in directory and subdirectories. On windows it will get both .JPG and .jpg. On unix it will only get .JPG images
"""
from typing import Sequence, Dict, Optional, Any
from itertools import chain
from datetime import datetime, timezone

# torchvision and torch removed from top-level to avoid issues with spawn/fork
# from torchvision.utils import save_image
# import torchvision.transforms
from tqdm import tqdm
import math
from itertools import product
import numpy as np
import pandas as pd
import json, os
from pathlib import Path
from PIL import Image
import logging
import fire
import yaml
from functools import lru_cache
import numpy as np

from pydantic import BaseModel, Field, model_validator

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from wildetect.core.gps import GPSUtils, get_gsd, get_pixel_gps_coordinates
from wildetect.core.config import FlightSpecs

logger = logging.getLogger(__name__)


def _is_ignored_image_path(path: Path) -> bool:
    """Ignore OS-generated image artifacts such as macOS AppleDouble files."""
    return path.name.startswith("._")

@lru_cache(maxsize=10)
def load_images_paths(image_dir:str,patterns:tuple[str])->list[str]:
    images_paths = chain.from_iterable([Path(image_dir).glob(p) for p in patterns])
    images_paths = sorted([p for p in set(images_paths) if not _is_ignored_image_path(p)])
    return list(map(str,images_paths))

class Args(BaseModel):

    # configuration of full resolution images
    root: Optional[str] = None
    rmheight: Optional[float] = None
    rmwidth: Optional[float] = None
    flight_height: float = 180
    sensor_height: float = 24.0
    overlapfactor: float = Field(default=0.0,le=1.0,ge=0.0)
    ratiowidth: float = Field(default=0.5,le=1.,gt=0.0)
    ratioheight: float = Field(default=0.5,le=1.,gt=0.0)

    # configuration in csv
    config_file_csv: Optional[str] = None

    n_workers: int = 3

    mae_threshold:float = 255.0
    failure_threshold: int = 10

    out_json_coords_files: Optional[str] = None
    load_existing_json_file: bool = False
    
    save_tiles: bool = False
    out_folder: Optional[str] = None

    patterns: tuple[str, ...] = Field(
        default_factory=lambda: tuple(
            f"**/{ext}"
            for base in ("*.jpg", "*.jpeg", "*.png")
            for ext in (base, base.upper())
        )
    )

    # tile
    tiled_images_folder: Optional[str] = None

    # debug mode
    debug: bool = False

    # CSV
    altitude: Optional[float] = None
    filename_col:str = "filename"  # CSV column name for filenames
    lat_col:str = "latitude" # CSV column name for latitude
    lon_col:str = "longitude"  # CSV column name for longitude
    alt_col:str = "altitude"  # CSV column name for altitude
    out_csv_path: Optional[str] = None
    out_report_json: Optional[str] = None
    log_file: Optional[str] = "tile_gps_matching.log"

    @property
    def expected_tiles_per_image(self) -> int:
        """Lower bound on expected number of tiles. Does not account for overlap factor"""
        return round((1.0 / self.ratiowidth) * (1.0 / self.ratioheight))

    @model_validator(mode="after")
    def validate_args(self) -> "Args":
        if self.config_file_csv is None:
            if self.root is None or self.rmheight is None or self.rmwidth is None:
                raise ValueError("root, rmheight, and rmwidth are required if config_file_csv is not provided")

            if self.out_json_coords_files is None:
                root_name = Path(self.root).stem
                self.out_json_coords_files = str(Path(self.root).with_name(root_name+"-coordinates.json"))

            if self.out_csv_path is None:
                root_name = Path(self.root).stem
                self.out_csv_path = str(Path(self.root).with_name(root_name + "-coordinates.csv"))
            else:
                out_csv = Path(self.out_csv_path)
                if out_csv.suffix.lower() != ".csv":
                    raise ValueError("out_csv_path must be a .csv file path")
                if out_csv.parent != Path(".") and not out_csv.parent.exists():
                    raise FileNotFoundError(
                        f"Output directory does not exist for out_csv_path: {out_csv.parent}"
                    )

        if self.out_folder is not None:
            Path(self.out_folder).mkdir(parents=False, exist_ok=True)

        return self


def setup_logging(log_file: Optional[str]) -> None:
    if not log_file:
        return

    log_path = Path(log_file).expanduser()
    if log_path.parent != Path("."):
        log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Avoid duplicate file handlers if run is called multiple times.
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == str(log_path.resolve()):
            return

    root_logger.addHandler(file_handler)


def get_coordinates(image_width,tile_w,image_height,tile_h,overlaping_factor):

        # x limits
        lim = math.ceil((image_width-tile_w)/((1-overlaping_factor)*tile_w))
        x_right = [math.floor(tile_w + i*(1-overlaping_factor)*tile_w) for i in range(lim)]
        x_coords = [(x-tile_w,x) for x in x_right]
        if len(x_coords)>0:
            left,right = x_coords[-1]
            x_coords[-1] = (left,image_width) # extending to remaining pixels

        # y limits
        lim = math.ceil((image_height-tile_h)/((1-overlaping_factor)*tile_h))
        y_bottom = [math.floor(tile_h + i*(1-overlaping_factor)*tile_h) for i in range(lim)]
        y_coords = [(y-tile_h,y) for y in y_bottom]
        if len(y_coords)>0:
            top,bottom = y_coords[-1]
            y_coords[-1] = (top,image_height) # extending to remaining pixels

        # tiles coordinates
        if len(y_coords) == 0:
            y_coords = [(0,image_height),]

        if len(x_coords) == 0:
            x_coords = [(0,image_width),]

        coordinates = product(x_coords,y_coords)
        return list(coordinates)

def get_patches(image_tensor, coords: list):
    patches = list()
    # Detect if it's CHW (Torch) or HWC (NumPy)
    # NumPy arrays from Image.open are typically HWC
    is_chw = hasattr(image_tensor, "shape") and len(image_tensor.shape) == 3 and image_tensor.shape[0] in [1, 3]
    
    for (x_left, x_right), (y_top, y_bottom) in coords:
        if is_chw:
            patches.append(image_tensor[:, y_top:y_bottom, x_left:x_right])
        else:
            patches.append(image_tensor[y_top:y_bottom, x_left:x_right, :])
    return patches

def save_list_images(
    image_tensor,
    tiles_bounds: list,
    basename: str,
    dest_folder: str
) -> None:
    """Save mini-batch tensors/arrays into image files."""
    # get patches
    is_chw = hasattr(image_tensor, "shape") and len(image_tensor.shape) == 3 and image_tensor.shape[0] in [1, 3]
    patches = list()
    for (x_left, x_right), (y_top, y_bottom) in tiles_bounds:
        if is_chw:
            patches.append(image_tensor[:, y_top:y_bottom, x_left:x_right])
        else:
            patches.append(image_tensor[y_top:y_bottom, x_left:x_right, :])

    base_wo_extension, extension = basename.split('.')[0], basename.split('.')[1]
    for i, b in enumerate(range(len(patches))):
        full_path = '_'.join([base_wo_extension, str(i) + '.']) + extension
        save_path = os.path.join(dest_folder, full_path)
        
        # Convert tensor/array to PIL and save without torchvision
        patch = patches[b]
        if hasattr(patch, 'numpy'):
            patch = patch.numpy()
        
        # If it's CHW (from ToTensor), convert to HWC
        if patch.ndim == 3 and patch.shape[0] in [1, 3]:
            patch = patch.transpose(1, 2, 0)
            
        if patch.max() <= 1.0:
            patch = (patch * 255).astype(np.uint8)
            
        Image.fromarray(patch).save(save_path)

def get_tile_metadata(args:Args,img_path:str) -> dict:

    with Image.open(img_path) as pil_img:
        img_name = os.path.basename(img_path)
        image_width, image_height = pil_img.size

        # Cropping out image-level overlap
        height_overlap = math.ceil(args.rmheight * image_height)
        width_overlap = math.ceil(args.rmwidth * image_width)

        if height_overlap*width_overlap > 0 :
            image_width -= 2 * width_overlap
            image_height -= 2 * height_overlap
            logger.debug(f"Removing {2*width_overlap} pixels to the width; and {2*height_overlap} pixels to the height.")
        elif (height_overlap == 0) and (width_overlap != 0):
            image_width -= 2 * width_overlap
            logger.debug(f"Removing {2*width_overlap} pixels to the width; and {2*height_overlap} pixels to the height.")
        elif width_overlap == 0 and (height_overlap != 0):
            image_height -= 2 * height_overlap
            logger.debug(f"Removing {2*width_overlap} pixels to the width; and {2*height_overlap} pixels to the height.")
        
        # Computes tile width and height using the given ratios
        if args.ratiowidth > 0.0:
            width = math.ceil(image_width*args.ratiowidth)
        else:
            raise ValueError("ratiowidth should be greater than 0.0")

        if args.ratioheight > 0.0:
            height = math.ceil(image_height*args.ratioheight)
        else:
            raise ValueError("ratioheight should be greater than 0.0")
        
        # get tile coordinates
        coords =  get_coordinates(image_width,tile_w=width,image_height=image_height,tile_h=height,overlaping_factor=args.overlapfactor)       

        if len(coords) != args.expected_tiles_per_image:
            logger.warning(f"Expected at least {args.expected_tiles_per_image} tile bounds for {img_name}, but generated {len(coords)}.")

        # get tiles gps coordinates
        image_gps = GPSUtils.get_gps_coord(file_name=None,
                                            return_as_decimal=True,
                                            image=pil_img)
        if image_gps is not None:
            (lat,long,alt),_ = image_gps
            alt = alt/1000 # conver to meters
            tile_gps_coords = []
            gsd = get_gsd(
                    image=pil_img,
                    image_path=None,
                    flight_specs=FlightSpecs(
                        sensor_height=args.sensor_height,
                        flight_height=args.flight_height,
                    ),
                )
            for (x_left,x_right),(y_top,y_bottom) in coords:
                x = (x_left+x_right)/2
                y = (y_top+y_bottom)/2
                tile_lat, tile_lon = get_pixel_gps_coordinates(x=x,y=y,
                                                W=image_width,
                                                H=image_height,
                                                lat_center=lat,lon_center=long,
                                                gsd=gsd)
                tile_gps_coords.append((float(tile_lat),float(tile_lon)))
        else:
            tile_gps_coords = [None for _ in range(len(coords))]
        
        tile_metadata = dict()
        tile_metadata[Path(img_path).stem] = dict(tiles_bounds=coords,
                                            tiles_gps_coords=tile_gps_coords,                    
                                            )
        if args.save_tiles:
            # If saving tiles, we do need the array
            img_array = np.array(pil_img)
            # Apply crop to array if needed
            if height_overlap > 0 or width_overlap > 0:
                h_off = height_overlap if height_overlap > 0 else 0
                w_off = width_overlap if width_overlap > 0 else 0
                img_array = img_array[h_off:pil_img.height-h_off, w_off:pil_img.width-w_off]
            
            save_list_images(image_tensor=img_array, # Not actually a tensor anymore
                            tiles_bounds=coords,
                            basename=img_name,
                            dest_folder=args.out_folder)
        return tile_metadata

def get_tiles_gps_and_dimensions(args:Args) -> dict:

    images_paths = load_images_paths(image_dir=args.root,patterns=args.patterns)

    if args.debug:
        images_paths = images_paths[:5]

    tile_metadata = dict()
    with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
        futures = [executor.submit(get_tile_metadata, args, img_path) for img_path in images_paths]
        disable = args.debug # Disable tqdm if in debug/sweep mode
        for future in tqdm(as_completed(futures), total=len(images_paths), desc='Exporting patches', disable=disable):
            tile_metadata.update(future.result())

    # saving metdata
    json_path = args.out_json_coords_files
    with open(json_path, "w") as f:
        json.dump(tile_metadata, f, indent=1)

    return tile_metadata

def load_coordinates(coord_path:str):
    """
    Load coordinates from a JSON file
    
    Args:
        coord_path (str): path to the JSON file
    
    Returns:
        dict: tile data
    """
    with open(coord_path,'r') as file:
        tile_data = json.load(file)
    return tile_data

def verify_tile(parent_array: np.ndarray, tile_path: Path, coords: Sequence[Sequence[int]], thresh: float = 255.0):
    """
    Verify that a tile matches its parent image at the given coordinates.
    """
    try:
        tile_path = Path(tile_path)
        (x_min, x_max), (y_min, y_max) = coords
        parent_patch = parent_array[y_min:y_max, x_min:x_max]
        
        tile_img = Image.open(tile_path)
        tile_array = np.array(tile_img)
        
        if parent_patch.shape != tile_array.shape:
            parent_h, parent_w = parent_patch.shape[:2]
            tile_h, tile_w = tile_array.shape[:2]

            # Ratios help identify systematic resizing/cropping issues quickly.
            h_ratio = (tile_h / parent_h) if parent_h else float("nan")
            w_ratio = (tile_w / parent_w) if parent_w else float("nan")
            parent_area = parent_h * parent_w
            tile_area = tile_h * tile_w
            area_ratio = (tile_area / parent_area) if parent_area else float("nan")

            logger.warning(
                f"Dimension mismatch for {tile_path.name}: "
                f"Parent patch {parent_patch.shape} vs Tile {tile_array.shape} | "
                f"tile/parent ratios: h={h_ratio:.4f}, w={w_ratio:.4f}, area={area_ratio:.4f}"
            )
            return False
            
        mae = np.mean(np.abs(parent_patch.astype(np.float32) - tile_array.astype(np.float32)))
        
        if mae > thresh:
            logger.warning(f"Verification failed for {tile_path.name}: MAE = {mae:.2f} > {thresh}")
            
        return True
    except Exception as e:
        logger.warning(f"Error during verification of {tile_path.name}: {str(e)}")
        return False

def _find_parent_image(images_paths: list[str], image_name: str) -> Optional[Path]:
    """Search for parent image file with common extensions."""
    for p in images_paths:
        if Path(p).stem == image_name:
            return p
    return None

def _process_parent_group(
    image_name: str,
    tiles: list,
    tile_data: dict,
    parent_images_paths: str,
    mae_threshold: float = 255.,
    expected_tiles_per_image: int = 0
) -> tuple:
    """
    Process all tiles belonging to a single parent image.
    
    Loads the parent image once, then verifies and collects metadata
    for each tile in the group.
    
    Args:
        image_name: parent image stem name
        tiles: list of (tile_path, index) tuples for this parent
        tile_data: full tile metadata dict
        parent_root: directory containing parent images
    
    Returns:
        (group_metadata, group_report, error_msg)
        - group_metadata: dict mapping tile_path -> {tile_bounds, tile_gps_coords}
        - group_report: per-parent structured counters/details
        - error_msg: None if successful, error string if parent failed
    """
    metadata = tile_data.get(image_name)
    parent_report: dict[str, Any] = {
        "parent": image_name,
        "success_count": 0,
        "verification_failures": 0,
        "index_failures": 0,
        "other_failures": 0,
        "failure_details": [],
    }
    if metadata is None:
        return {}, parent_report, f"Image {image_name} not found in metadata"

    parent_path = _find_parent_image(parent_images_paths, image_name)
    if parent_path is None:
        return {}, parent_report, f"Parent image not found for {image_name}"
    parent_report["parent_path"] = str(parent_path)

    try:
        with Image.open(parent_path) as img:
            parent_array = np.array(img)
    except Exception as e:
        return {}, parent_report, f"Failed to load parent image {image_name}: {e}"

    group_metadata = {}
    tiles_bounds = metadata.get("tiles_bounds") if isinstance(metadata, dict) else None
    tiles_gps_coords = metadata.get("tiles_gps_coords") if isinstance(metadata, dict) else None

    if not isinstance(tiles_bounds, list) or not isinstance(tiles_gps_coords, list):
        logger.warning(
            "Malformed metadata for parent=%s parent_path=%s: "
            "tiles_bounds_type=%s tiles_gps_coords_type=%s",
            image_name,
            parent_path,
            type(tiles_bounds).__name__,
            type(tiles_gps_coords).__name__,
        )
        parent_report["other_failures"] += 1
        parent_report["failure_details"].append(
            {
                "type": "malformed_metadata",
                "parent": image_name,
                "parent_path": str(parent_path),
                "tiles_bounds_type": type(tiles_bounds).__name__,
                "tiles_gps_coords_type": type(tiles_gps_coords).__name__,
            }
        )
        return {}, parent_report, f"Malformed metadata for {image_name}"

    for tile_path, index in tiles:
        try:
            tile_coords = tiles_bounds[index]
            tile_gps = tiles_gps_coords[index]

            if not verify_tile(parent_array, tile_path, tile_coords, mae_threshold):
                parent_report["verification_failures"] += 1
                parent_report["failure_details"].append(
                    {
                        "type": "verification_failed",
                        "parent": image_name,
                        "parent_path": str(parent_path),
                        "index": index,
                        "tile": str(tile_path),
                        "tile_bounds": tile_coords,
                    }
                )
                continue

            group_metadata[str(tile_path)] = dict(
                tile_bounds=tile_coords,
                tile_gps_coords=tile_gps,
            )
            parent_report["success_count"] += 1
        except IndexError as e:
            logger.warning(
                "Tile index out of range while matching: "
                "parent=%s parent_path=%s index=%s bounds_len=%s gps_len=%s tile=%s error=%s",
                image_name,
                parent_path,
                index,
                len(tiles_bounds),
                len(tiles_gps_coords),
                tile_path,
                e,
            )
            parent_report["index_failures"] += 1
            parent_report["failure_details"].append(
                {
                    "type": "index_out_of_range",
                    "parent": image_name,
                    "parent_path": str(parent_path),
                    "index": index,
                    "bounds_len": len(tiles_bounds),
                    "gps_len": len(tiles_gps_coords),
                    "tile": str(tile_path),
                    "error": str(e),
                }
            )
        except Exception as e:
            logger.warning(
                "Failed to match tile: parent=%s parent_path=%s index=%s tile=%s error=%s",
                image_name,
                parent_path,
                index,
                tile_path,
                e,
            )
            parent_report["other_failures"] += 1
            parent_report["failure_details"].append(
                {
                    "type": "match_exception",
                    "parent": image_name,
                    "parent_path": str(parent_path),
                    "index": index,
                    "tile": str(tile_path),
                    "error": str(e),
                }
            )

    if len(group_metadata) != expected_tiles_per_image:
        raise ValueError(f"Expected at least {expected_tiles_per_image} tiles for {image_name}, but found {len(group_metadata)}. Tiling config is wrong.")

    return group_metadata, parent_report, None

def match_tiles_gps(
    tile_data: dict,
    images_dir: str,
    parent_root: str,
    image_ext_patterns:list,
    max_workers: int = 3,
    failure_threshold: int = 10,
    mae_threshold: float = 255.,
    expected_tiles_per_image: int = 1
) -> tuple[dict, dict[str, Any]]:
    """
    Match tiles gps coordinates to tile paths and verify them using
    multi-threading.
    
    Tiles are grouped by parent image so each thread loads its parent
    image once, then verifies all associated tiles. This maximizes I/O
    parallelism while avoiding shared-cache contention.
    
    Args:
        tile_data (dict): tile data
        images_dir (str): directory containing tile images
        parent_root (str): directory containing parent images
        max_workers (int): number of threads to use
    
    Returns:
        dict: tile metadata
    """
    # Discover and group tiles by parent image name
    tile_paths = load_images_paths(image_dir=images_dir,patterns=image_ext_patterns)

    # Get Parent images
    parent_images_paths = load_images_paths(image_dir=parent_root,patterns=image_ext_patterns)

    parent_groups: Dict[str, list] = {}
    skipped = 0
    for tile_path in tile_paths:
        try:
            parts = Path(tile_path).stem.split('_')
            index = int(parts[-1])
            image_name = "_".join(parts[:-1])
        except (ValueError, IndexError):
            skipped += 1
            continue
        parent_groups.setdefault(image_name, []).append((tile_path, index))

    if skipped:
        logger.warning(f"Skipped {skipped} tiles with unparseable names")

    # Filter groups to only those present in tile_data
    parent_groups = {k: v for k, v in parent_groups.items() if k in tile_data}

    if not parent_groups:
        logger.warning("No tiles matched any parent images in the metadata.")
        return {}, report

    # Process groups in parallel
    tiles_metadata = {}
    failed = set()
    report: dict[str, Any] = {
        "summary": {
            "tiles_discovered": len(tile_paths),
            "tiles_with_unparseable_name": skipped,
            "parent_groups": len(parent_groups),
            "tiles_matched_successfully": 0,
            "verification_failures": 0,
            "index_failures": 0,
            "other_failures": 0,
            "failed_parent_count": 0,
        },
        "failed_parents": [],
        "failure_details": [],
    }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {
            executor.submit(
                _process_parent_group,
                image_name,
                tiles,
                tile_data,
                parent_images_paths,
                mae_threshold,
                expected_tiles_per_image
            ): image_name
            for image_name, tiles in parent_groups.items()
        }

        disable = (tile_data is not None and len(tile_data) < 10) or (hasattr(args, 'debug') and args.debug)
        for future in tqdm(
            as_completed(future_to_name),
            total=len(future_to_name),
            desc='Matching tiles gps coordinates',
            disable=disable
        ):
            image_name = future_to_name[future]
            try:
                group_meta, parent_report, error_msg = future.result()
            except Exception as e:
                #logger.warning(f"Unexpected error processing {image_name}: {e}")
                failed.add(image_name)
                report["summary"]["other_failures"] += 1
                report["failure_details"].append(
                    {
                        "type": "unexpected_parent_exception",
                        "parent": image_name,
                        "error": str(e),
                    }
                )
            else:
                if error_msg:
                    logger.warning(error_msg)
                    failed.add(image_name)
                    report["failure_details"].append(
                        {
                            "type": "parent_error",
                            "parent": image_name,
                            "error": error_msg,
                        }
                    )
                else:
                    tiles_metadata.update(group_meta)
                    report["summary"]["tiles_matched_successfully"] += parent_report.get("success_count", 0)
                    report["summary"]["verification_failures"] += parent_report.get("verification_failures", 0)
                    report["summary"]["index_failures"] += parent_report.get("index_failures", 0)
                    report["summary"]["other_failures"] += parent_report.get("other_failures", 0)
                    report["failure_details"].extend(parent_report.get("failure_details", []))

            if len(failed) >= failure_threshold:
                logger.warning("Too many failures, cancelling remaining tasks")
                for f in future_to_name:
                    f.cancel()
                raise RuntimeError(f"Too many failures, cancelling remaining tasks. We reached threshold: {failure_threshold}")

    report["failed_parents"] = sorted(list(failed))
    report["summary"]["failed_parent_count"] = len(failed)

    if len(failed):
        logger.warning(f"Failed to match {len(failed)} images")
    if report["summary"]["verification_failures"] > 0:
        logger.warning(f"Verification failed for {report['summary']['verification_failures']} tiles")

    return tiles_metadata, report


def _default_report_path(out_csv_path: str) -> str:
    out_csv = Path(out_csv_path)
    return str(out_csv.with_name(f"{out_csv.stem}-matching-report.json"))


def write_match_report_json(
    args: Args,
    match_report: dict[str, Any],
    out_csv_path: str,
    csv_rows_written: int,
) -> str:
    report_path = args.out_report_json or _default_report_path(out_csv_path)
    report_file = Path(report_path)
    if not report_file.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist for report: {report_file.parent}")

    payload: dict[str, Any] = {
        "run_info": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "root": args.root,
            "tiled_images_folder": args.tiled_images_folder,
            "out_csv_path": out_csv_path,
            "out_json_coords_files": args.out_json_coords_files,
            "patterns": list(args.patterns),
            "n_workers": args.n_workers,
            "mae_threshold": args.mae_threshold,
            "expected_tiles_per_image": args.expected_tiles_per_image,
        },
        "summary": dict(match_report.get("summary", {})),
        "failed_parents": list(match_report.get("failed_parents", [])),
        "failure_details": list(match_report.get("failure_details", [])),
    }
    payload["summary"]["csv_rows_written"] = csv_rows_written

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Wrote run report JSON: %s", report_file)
    return str(report_file)

def convert_metadata_to_csv(
    tiles_metadata: dict,
    out_csv_path: str,
    altitude: Optional[float] = None,
    filename_col:str = "filename",  # CSV column name for filenames
    lat_col:str = "latitude",  # CSV column name for latitude
    lon_col:str = "longitude",  # CSV column name for longitude
    alt_col:str = "altitude",  # CSV column name for altitude
) -> tuple[str, int]:

    out_path = Path(out_csv_path)
    if not out_path.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {out_path.parent}")

    rows = []
    for tile_path, metadata in tqdm(tiles_metadata.items(),desc="Creating CSV"):
        lat = None
        lon = None
        tile_gps = None

        if isinstance(metadata, dict):
            tile_gps = metadata.get("tile_gps_coords")
        else:
            logger.warning(f"Malformed metadata for tile {tile_path}: expected dict")

        if tile_gps is not None:
            if isinstance(tile_gps, (list, tuple)) and len(tile_gps) == 2:
                lat, lon = tile_gps
            else:
                logger.warning(
                    f"Malformed tile_gps_coords for tile {tile_path}: {tile_gps}"
                )

        rows.append(
            {
                filename_col: Path(tile_path).name,
                lat_col: lat,
                lon_col: lon,
                alt_col: altitude,
            }
        )
    
    if len(rows) == 0:
        raise ValueError("No rows to save")

    dataframe = pd.DataFrame(rows, columns=[filename_col, lat_col, lon_col, alt_col])
    dataframe.to_csv(out_path, index=False, encoding="utf-8")
    logger.info(f"Wrote {len(rows)} rows to CSV: {out_path}")
    return str(out_path), len(rows)

def process_single_run(args: Args):
    logger.info(f"config:{args}")
    if args.load_existing_json_file:
        tile_data = load_coordinates(args.out_json_coords_files)
    else:
        tile_data = get_tiles_gps_and_dimensions(args)
    
    tiles_metadata, match_report = match_tiles_gps(
        tile_data,
        args.tiled_images_folder,
        parent_root=args.root,
        max_workers=args.n_workers,
        mae_threshold=args.mae_threshold,
        image_ext_patterns=args.patterns,
        failure_threshold=args.failure_threshold,
        expected_tiles_per_image=args.expected_tiles_per_image
    )
    
    if args.debug:
        logger.info(f"Skipping CSV writing because debug mode is enabled. Outputs not saved.")
        return

    out_csv_path, rows_written = convert_metadata_to_csv(
        tiles_metadata,
        out_csv_path=args.out_csv_path,
        altitude=args.altitude if args.altitude is not None else args.flight_height,
        filename_col=args.filename_col,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        alt_col=args.alt_col,
    )
    if rows_written == 0:
        summary = match_report.get("summary", {})
        logger.warning(
            "CSV output is empty (0 rows). verification_failures=%s index_failures=%s "
            "other_failures=%s failed_parent_count=%s out_csv=%s",
            summary.get("verification_failures", 0),
            summary.get("index_failures", 0),
            summary.get("other_failures", 0),
            summary.get("failed_parent_count", 0),
            out_csv_path,
        )
    write_match_report_json(
        args=args,
        match_report=match_report,
        out_csv_path=out_csv_path,
        csv_rows_written=rows_written,
    )

def main(args: Args):

    setup_logging(args.log_file)

    if args.config_file_csv:
        
        df = pd.read_csv(args.config_file_csv, sep=';', decimal=',')
        batch_stats: dict[str, Any] = {
            "success_count": 0,
            "failure_count": 0,
            "skipped_count": 0,
            "rows": {},
        }
        for row_idx, row in tqdm(df.iterrows(),desc="Processing CSV rows",total=len(df)):
                        
            dump = args.model_dump(exclude={'config_file_csv', 'out_json_coords_files', 'out_csv_path'})
            
            roots = []
            if 'raw_image_path' in row and pd.notna(row['raw_image_path']):
                roots.append(str(row['raw_image_path']).strip())
            
            for col in df.columns:
                if str(col).startswith('Unnamed') and pd.notna(row[col]):
                    val = str(row[col]).strip()
                    if val:
                        roots.append(val)
            
            if not roots:
                logger.warning(f"No raw_image_path found for row, skipping: {row.to_dict()}")
                batch_stats["skipped_count"] += 1
                batch_stats["rows"][int(row_idx)] = {
                    "status": "skipped",
                    "error": "No raw_image_path found",
                }
                continue
                
            dump['root'] = os.path.commonpath(roots) if len(roots) > 1 else roots[0]
            
            if pd.notna(row.get('tiled_image_path')):
                dump['tiled_images_folder'] = str(row['tiled_image_path']).strip()
            if pd.notna(row.get('rm_height')):
                dump['rmheight'] = float(row['rm_height'])
            if pd.notna(row.get('rm_width')):
                dump['rmwidth'] = float(row['rm_width'])
            if pd.notna(row.get('ratio_height')):
                dump['ratioheight'] = float(row['ratio_height'])
            if pd.notna(row.get('ratio_width')):
                dump['ratiowidth'] = float(row['ratio_width'])
            if pd.notna(row.get('overlap_factor')):
                dump['overlapfactor'] = float(row['overlap_factor'])
                
            first_root = roots[0]
            root_name = Path(first_root).stem
            dump['out_json_coords_files'] = str(Path(first_root).with_name(root_name+"-coordinates.json"))
            dump['out_csv_path'] = str(Path(first_root).with_name(root_name + "-coordinates.csv"))
            
            try:
                row_args = Args(**dump)
                logger.info(f"Processing row with root: {row_args.root}")
                process_single_run(row_args)
                batch_stats["success_count"] += 1
                batch_stats["rows"][int(row_idx)] = {
                    "status": "success",
                    "root": row_args.root,
                    "out_csv_path": dump.get("out_csv_path"),
                }
            except Exception as e:
                logger.error(f"Failed to process row: {row.to_dict()}\nError: {e}")
                batch_stats["failure_count"] += 1
                batch_stats["rows"][int(row_idx)] = {
                    "status": "failure",
                    "root": dump.get("root"),
                    "error": str(e),
                }
        logger.info("CSV batch row summary: %s", batch_stats)
    else:
        process_single_run(args)

def _args_from_config_and_overrides(
    config: Optional[str], overrides: dict[str, object]
) -> Args:
    merged: dict = {}
    if config is not None:
        path = Path(config).expanduser().resolve()
        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config must be a YAML mapping at top level, got {type(loaded)}")
        merged = dict(loaded)

    field_names = set(Args.model_fields.keys())
    for key, value in overrides.items():
        if key not in field_names:
            continue
        if value is not None:
            merged[key] = value

    return Args(**merged)

def _sweep_worker(sweep_args: tuple) -> dict:
    """Top-level worker so ProcessPoolExecutor can pickle it."""
    from typing import Any
    base_dump, rmh, rmw, ovlp, rw, rh = sweep_args
    overrides = dict(
        rmheight=rmh, rmwidth=rmw, overlapfactor=ovlp,
        ratiowidth=rw, ratioheight=rh, debug=True,
    )
    row: dict = dict(
        rmheight=rmh, rmwidth=rmw, overlapfactor=ovlp,
        ratiowidth=rw, ratioheight=rh, status="ok", error=None,
    )
    try:
        run_args = Args(**{**base_dump, **overrides})
        process_single_run(run_args)
        logger.info(
            "Config ok:   rmh=%.3f rmw=%.3f ovlp=%.2f rw=%.4f rh=%.4f",
            rmh, rmw, ovlp, rw, rh,
        )
    except Exception as exc:
        row["status"] = "fail"
        row["error"] = str(exc)
        logger.debug(
            "Config fail: rmh=%.3f rmw=%.3f ovlp=%.2f rw=%.4f rh=%.4f -> %s",
            rmh, rmw, ovlp, rw, rh, exc,
        )
    return row


def find_missing_configs(args: Args) -> pd.DataFrame:
    """Sweep tiling parameter combinations and identify which ones fail.

    Runs ``process_single_run`` in debug mode for every combination of
    ``rmheight``, ``rmwidth``, ``overlapfactor``, ``ratiowidth``, and
    ``ratioheight``.  Each run is wrapped in a try/except so that failing
    configs are recorded instead of aborting the whole sweep.

    Args:
        args: Base ``Args`` instance.  Must have ``debug=True``.

    Returns:
        A :class:`~pandas.DataFrame` with one row per parameter combo and
        columns ``rmheight``, ``rmwidth``, ``overlapfactor``, ``ratiowidth``,
        ``ratioheight``, ``status`` (``"ok"`` / ``"fail"``), and ``error``
        (the exception message, or ``None`` on success).  The frame is also
        saved as a CSV next to the log file.
    """
    assert args.debug, "Please enable debug mode."

    # --- parameter grid ---------------------------------------------------
    rmheight_vals = np.arange(0.2, 0.3, 0.01)
    rmwidth_vals = np.arange(0.2, 0.3, 0.01)
    overlapfactor_vals = np.arange(0.1, 0.3, 0.1)
    ratiowidth_vals = [1, 0.5, 1 / 3]
    ratioheight_vals = [1, 0.5, 1 / 3]

    combos = list(
        product(rmheight_vals, rmwidth_vals, overlapfactor_vals, ratiowidth_vals, ratioheight_vals)
    )
    # Shuffle so workers sample the space evenly from the start
    np.random.shuffle(combos)
    logger.info("Sweeping %d parameter combinations", len(combos))

    # --- sequential sweep -------------------------------------------------
    base_dump = args.model_dump()
    results = []
    
    for rmh, rmw, ovlp, rw, rh in tqdm(combos, desc="Config sweep"):
        sweep_args = (base_dump, float(rmh), float(rmw), float(ovlp), float(rw), float(rh))
        results.append(_sweep_worker(sweep_args))

    # --- collate & save ---------------------------------------------------
    df = pd.DataFrame(
        results,
        columns=["rmheight", "rmwidth", "overlapfactor", "ratiowidth", "ratioheight", "status", "error"],
    )

    ok_count = (df["status"] == "ok").sum()
    logger.info(
        "Sweep complete: %d ok / %d fail out of %d combos",
        ok_count, len(df) - ok_count, len(df),
    )

    out_csv = (
        Path(args.log_file).with_name("config_sweep_results.csv")
        if args.log_file
        else Path("config_sweep_results.csv")
    )
    df.to_csv(out_csv, index=False)
    logger.info("Sweep results saved to %s", out_csv)

    return df


class TileGpsMatchingCli:
    """CLI for tile GPS matching (Fire). Use the ``run`` command."""

    def run(
        self,
        config: Optional[str] = None,
        trace: bool = False,
        help: bool = False,  # noqa: A002 — Fire passes this for ``run --help``
        **overrides: object,
    ) -> None:
        """Load YAML from ``config`` then apply any ``Args`` fields passed as flags."""
        if help:
            logger.info(
                "For a full flag list including ``Args`` fields, run: "
                "%s run -- --help",
                Path(__file__).name,
            )
            return
        del trace  # Fire-only flag
        args = _args_from_config_and_overrides(config, dict(overrides))
        main(args)

    def sweep(
        self,
        config: Optional[str] = None,
        **overrides: object,
    ) -> None:
        """Sweep tiling parameter combos and report which ones fail.

        Loads args the same way as ``run``, forces ``debug=True``, then
        calls :func:`find_missing_configs`.  Results are saved to
        ``config_sweep_results.csv``.
        """
        overrides["debug"] = True
        args = _args_from_config_and_overrides(config, dict(overrides))
        setup_logging(args.log_file)
        find_missing_configs(args)


if __name__ == "__main__":
    fire.Fire(TileGpsMatchingCli)
