
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
from collections import OrderedDict
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
        
    patterns:str=("*.JPG","*.jpg","*.png","*.PNG","*.jpeg","*.JPEG")


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

def process_one_image(args:Args,img_path:str):

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
            lat, lon = get_pixel_gps_coordinates(x=x,y=y,
                                            W=image_width,
                                            H=image_height,
                                            lat_center=lat,lon_center=long,
                                            gsd=gsd)
            tile_gps_coords.append((float(lat),float(lon)))
    else:
        tile_gps_coords = [None for _ in range(len(coords))]
    
    tile_metadata = dict()
    tile_metadata[Path(img_path).stem] = dict(xy_coords=coords,
                                        gps_coords=tile_gps_coords,                    
                                        )
    return tile_metadata
        

def get_tiles_gps_and_dimensions(args:Args):

    assert Path(args.out_file).parent.exists(), "Output directory does not exist"
    assert args.overlapfactor<1, 'It should be less than 1.'
    assert (args.ratiowidth<=1.0) and (args.ratioheight<=1.0), "The ratios should be at most 1.0"

    images_paths = chain.from_iterable([Path(args.root).glob(p) for p in args.patterns])
    images_paths = list(images_paths)

    tile_metadata = dict()
    with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
        futures = [executor.submit(process_one_image, args, img_path) for img_path in images_paths]
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


def match_tiles_gps(tile_data: dict, images_dir: str, parent_root: str, cache_size: int = 5):
    """
    Match tiles gps coordinates to tile paths and verify them.
    
    Args:
        tile_data (dict): tile data
        images_dir (str): directory containing tile images
        parent_root (str): directory containing parent images
        cache_size (int): number of parent images to keep in FIFO cache
    
    Returns:
        dict: tile metadata
    """
    tile_paths = chain.from_iterable([Path(images_dir).glob(p) for p in ['*.JPG', '*.jpeg', '*.png', '*.jpg', '*.PNG', '*.JPEG']])
    tile_paths = list(tile_paths)

    tiles_metadata = dict()
    parent_cache = OrderedDict()
    failed = set()
    verification_failures = 0
    
    for tile_path in tqdm(tile_paths, desc='Matching tiles gps coordinates'):
        try:
            parts = Path(tile_path).stem.split('_')
            index = int(parts[-1])
            image_name = "_".join(parts[:-1])
        except (ValueError, IndexError):
            continue
        
        metadata = tile_data.get(image_name)
        if metadata is None:
            if image_name not in failed:
                logger.warning(f"Image {image_name} not found in metadata")
            failed.add(image_name)
        else:
            try:
                tile_coords = metadata['xy_coords'][index]
                tile_gps = metadata['gps_coords'][index]
                
                # Check cache or load parent image
                if image_name not in parent_cache:
                    if len(parent_cache) >= cache_size:
                        parent_cache.popitem(last=False)
                    
                    # Search for parent image with common extensions
                    parent_path = None
                    for ext in ['.JPG', '.jpg', '.png', '.jpeg', '.PNG']:
                        path = Path(parent_root) / f"{image_name}{ext}"
                        if path.exists():
                            parent_path = path
                            break
                    
                    if parent_path:
                        parent_cache[image_name] = np.array(Image.open(parent_path))
                    else:
                        logger.warning(f"Parent image not found for {image_name}")
                        parent_cache[image_name] = None
                
                parent_array = parent_cache[image_name]
                if parent_array is not None:
                    if not verify_tile(parent_array, tile_path, tile_coords):
                        verification_failures += 1
                
                tiles_metadata[str(tile_path)] = dict(tile_coordinates=tile_coords,
                                                  tile_gps_coord=tile_gps)
            except Exception as e:
                if image_name not in failed:
                    logger.warning(f"Failed to match tile {index} - {tile_path}: {e}")
                failed.add(image_name)
        
        if len(failed) > 10:
            break
                
    logger.info(f"Failed to match {len(failed)} images")
    if verification_failures > 0:
        logger.info(f"Verification failed for {verification_failures} tiles")


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
            )

    tile_data = get_tiles_gps_and_dimensions(args)

    #tile_data = load_coordinates(args.out_file)

    match_tiles_gps(tile_data, r"D:\workspace\data\savmap_dataset_v2\images_splits", parent_root=args.root)
