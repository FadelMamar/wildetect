
"""# match saved tiles with their GPS coordinates
#### Meanings of arguments
- ```-ratioheight``` : proportion of tile  w.r.t height of image. Example 0.5 means dividing the image in two bands w.r.t height.
- ```-ratiowidth``` : proportion of tile w.r.t to width of image. Example 1.0 means the width of the tile is the same as the image.
- ```-overlapfactor``` : percentage of overlap. It should be less than 1.
- ```-rmheight``` : percentage of height to remove or crop at bottom and top
- ```-rmwidth``` : percentage of width to remove or crop on each side of the image
- ```-pattern``` : "**/*.JPG" will get all .JPG images in directory and subdirectories. On windows it will get both .JPG and .jpg. On unix it will only get .JPG images
"""
from dataclasses import dataclass
from typing import Sequence, Dict, Optional
from itertools import chain

from torchvision.utils import save_image
import torchvision.transforms
from tqdm import tqdm
import math
from itertools import product
import numpy as np
import json, os
from pathlib import Path
from PIL import Image
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from wildetect.core.gps import GPSUtils, get_gsd, get_pixel_gps_coordinates
from wildetect.core.config import FlightSpecs

#from wildata.adapters.utils import read_image

logger = logging.getLogger(__name__)

@dataclass
class Args:

    root:str

    rmheight:float
    rmwidth:float

    flight_height:float=180
    sensor_height:float=7.4,

    overlapfactor:float=0.1

    ratiowidth:float=0.5
    ratioheight:float=0.5  

    n_workers:int=1 

    out_file:str="coordinates.json"
    
    save_tiles:bool=False
    out_folder:Optional[str]=None
        
    patterns:tuple = tuple(
        f"**/{ext}"
        for base in ("*.jpg", "*.jpeg", "*.png")
        for ext in (base, base.upper())
    )


    def __post_init__(self,):
        if self.out_folder is not None:
            Path(self.out_folder).mkdir(parents=False,exist_ok=True)
        
        assert Path(self.out_file).parent.exists(), "Output directory for JSON file does not exist"
        assert self.overlapfactor<1, 'It should be less than 1.'
        assert (self.ratiowidth<=1.0) and (self.ratioheight<=1.0), "The ratios should be at most 1.0"


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

def get_patches(image_tensor,coords:list):
    
    patches = list()
    
    # store patches
    for (x_left,x_right),(y_top,y_bottom) in coords:
        patches.append(image_tensor[:,y_top:y_bottom,x_left:x_right])
        
    return patches

def save_list_images(
    image_tensor:list,
    tiles_bounds:list,
    basename: str,
    dest_folder: str
    ) -> None:
    ''' Save mini-batch tensors into image files

    Use torchvision save_image function,
    see https://pytorch.org/vision/stable/utils.html#torchvision.utils.save_image

    Args:
        batch (list): mini-batch tensor
        basename (str) : parent image name, with extension
        dest_folder (str): destination folder path
    '''
    

    # get patches
    patches = list()
    for (x_left,x_right),(y_top,y_bottom) in tiles_bounds:
        patches.append(image_tensor[:,y_top:y_bottom,x_left:x_right])

    base_wo_extension, extension = basename.split('.')[0], basename.split('.')[1]
    for i, b in enumerate(range(len(patches))):
        full_path = '_'.join([base_wo_extension, str(i) + '.']) + extension
        save_path = os.path.join(dest_folder, full_path)
        save_image(patches[b], fp=save_path)

def get_tile_metadata(args:Args,img_path:str) -> dict:

    try:
        pil_img = Image.open(img_path)
    except Exception:
        logger.warning(f"failed for: {img_path}")
        raise FileNotFoundError(f"Failed to open image: {img_path}")

    img_tensor = torchvision.transforms.ToTensor()(pil_img)
    img_name = os.path.basename(img_path)

    # Cropping out image-level overlap
    height_overlap = math.ceil(args.rmheight * img_tensor.shape[1])
    width_overlap = math.ceil(args.rmwidth * img_tensor.shape[2])

    if height_overlap*width_overlap > 0 :
        img_tensor = img_tensor[:,height_overlap:-height_overlap, width_overlap:-width_overlap]
        logger.debug(f"Removing {2*width_overlap} pixels to the width; and {2*height_overlap} pixels to the height.")
    elif (height_overlap == 0) and (width_overlap != 0):
        img_tensor = img_tensor[:,:, width_overlap:-width_overlap]
        logger.debug(f"Removing {2*width_overlap} pixels to the width; and {2*height_overlap} pixels to the height.")
    elif width_overlap == 0 and (height_overlap != 0):
        img_tensor = img_tensor[:,height_overlap:-height_overlap,:]
        logger.debug(f"Removing {2*width_overlap} pixels to the width; and {2*height_overlap} pixels to the height.")
    
    # Computes tile width and height using the given ratios
    if args.ratiowidth > 0.0:
        width = math.ceil(img_tensor.shape[2]*args.ratiowidth)
    if args.ratioheight > 0.0:
        height = math.ceil(img_tensor.shape[1]*args.ratioheight)
    
    image_width=img_tensor.shape[2]
    image_height=img_tensor.shape[1]

    # get tile coordinates
    coords =  get_coordinates(image_width,tile_w=width,image_height=image_height,tile_h=height,overlaping_factor=args.overlapfactor)       

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
        save_list_images(image_tensor=img_tensor,
                        tiles_bounds=coords,
                        basename=img_name,
                        dest_folder=args.out_folder)
    return tile_metadata 

def get_tiles_gps_and_dimensions(args:Args) -> dict:

    images_paths = chain.from_iterable([Path(args.root).glob(p) for p in args.patterns])
    images_paths = list(set(images_paths))

    tile_metadata = dict()
    with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
        futures = [executor.submit(get_tile_metadata, args, img_path) for img_path in images_paths]
        for future in tqdm(as_completed(futures), total=len(images_paths), desc='Exporting patches'):
            tile_metadata.update(future.result())

    # saving metdata
    json_path = args.out_file
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


def verify_tile(parent_array: np.ndarray, tile_path: Path, coords: Sequence[Sequence[int]], thresh: float = 20.0):
    """
    Verify that a tile matches its parent image at the given coordinates.
    """
    try:
        (x_min, x_max), (y_min, y_max) = coords
        parent_patch = parent_array[y_min:y_max, x_min:x_max]
        
        tile_img = Image.open(tile_path)
        tile_array = np.array(tile_img)
        
        if parent_patch.shape != tile_array.shape:
            logger.warning(f"Dimension mismatch for {tile_path.name}: "
                           f"Parent patch {parent_patch.shape} vs Tile {tile_array.shape}")
            return False
            
        mae = np.mean(np.abs(parent_patch.astype(np.float32) - tile_array.astype(np.float32)))
        
        if mae > thresh:
            logger.warning(f"Verification failed for {tile_path.name}: MAE = {mae:.2f} > {thresh}")
            return False
            
        return True
    except Exception as e:
        logger.warning(f"Error during verification of {tile_path.name}: {str(e)}")
        return False


def _find_parent_image(parent_root: str, image_name: str) -> Optional[Path]:
    """Search for parent image file with common extensions."""
    for ext in ['.JPG', '.jpg', '.png', '.jpeg', '.PNG']:
        path = Path(parent_root) / f"{image_name}{ext}"
        if path.exists():
            return path
    return None


def _process_parent_group(
    image_name: str,
    tiles: list,
    tile_data: dict,
    parent_root: str,
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
        (group_metadata, verification_failures, error_msg)
        - group_metadata: dict mapping tile_path -> {tile_bounds, tile_gps_coords}
        - verification_failures: number of tiles that failed verification
        - error_msg: None if successful, error string if parent failed
    """
    metadata = tile_data.get(image_name)
    if metadata is None:
        return {}, 0, f"Image {image_name} not found in metadata"

    parent_path = _find_parent_image(parent_root, image_name)
    if parent_path is None:
        return {}, 0, f"Parent image not found for {image_name}"

    try:
        parent_array = np.array(Image.open(parent_path))
    except Exception as e:
        return {}, 0, f"Failed to load parent image {image_name}: {e}"

    group_metadata = {}
    v_failures = 0

    for tile_path, index in tiles:
        try:
            tile_coords = metadata['tiles_bounds'][index]
            tile_gps = metadata['tiles_gps_coords'][index]

            if not verify_tile(parent_array, tile_path, tile_coords):
                v_failures += 1
                continue

            group_metadata[str(tile_path)] = dict(
                tile_bounds=tile_coords,
                tile_gps_coords=tile_gps,
            )
        except Exception as e:
            logger.warning(f"Failed to match tile {index} - {tile_path}: {e}")

    return group_metadata, v_failures, None


def match_tiles_gps(
    tile_data: dict,
    images_dir: str,
    parent_root: str,
    max_workers: int = 3,
):
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
    _exts = ['*.jpg', '*.jpeg', '*.png']
    tile_paths = chain.from_iterable(
        [Path(images_dir).glob(p) for p in _exts + [e.upper() for e in _exts]]
    )
    tile_paths = list(set(tile_paths))

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
        logger.debug(f"Skipped {skipped} tiles with unparseable names")

    # Process groups in parallel
    tiles_metadata = {}
    failed = set()
    verification_failures = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {
            executor.submit(
                _process_parent_group,
                image_name,
                tiles,
                tile_data,
                parent_root,
            ): image_name
            for image_name, tiles in parent_groups.items()
        }

        for future in tqdm(
            as_completed(future_to_name),
            total=len(future_to_name),
            desc='Matching tiles gps coordinates',
        ):
            image_name = future_to_name[future]
            try:
                group_meta, v_failures, error_msg = future.result()
            except Exception as e:
                logger.warning(f"Unexpected error processing {image_name}: {e}")
                failed.add(image_name)
                continue

            if error_msg:
                logger.warning(error_msg)
                failed.add(image_name)
            else:
                tiles_metadata.update(group_meta)
                verification_failures += v_failures

            if len(failed) > 10:
                logger.warning("Too many failures, cancelling remaining tasks")
                for f in future_to_name:
                    f.cancel()
                break

    logger.info(f"Failed to match {len(failed)} images")
    if verification_failures > 0:
        logger.info(f"Verification failed for {verification_failures} tiles")

    return tiles_metadata


if __name__ == "__main__":

    args = Args(root=r"D:\workspace\data\savmap_dataset_v2\raw\images",
            overlapfactor=0.1,
            ratiowidth=0.33,
            ratioheight=0.5,
            rmheight=0.0,
            n_workers=3,
            rmwidth=0.0,
            flight_height=180,
            sensor_height=7.4,
            out_file=r"D:\workspace\data\savmap_dataset_v2\raw\tiles_coordinates.json",
            out_folder=r"D:\workspace\data\savmap_dataset_v2\raw\tiles",
            save_tiles=False
            )
    
    

    tile_data = get_tiles_gps_and_dimensions(args)

    #tile_data = load_coordinates(args.out_file)

    match_tiles_gps(tile_data, r"D:\workspace\data\savmap_dataset_v2\raw\tiles", parent_root=args.root)
