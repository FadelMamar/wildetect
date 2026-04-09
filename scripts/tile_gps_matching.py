
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
from typing import Sequence
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
import geopy
from wildetect.core.gps import GPSUtils, get_gsd, get_pixel_gps_coordinates

@dataclass
class Args:

    root:str
    dest:str    

    rmheight:float
    rmwidth:float

    flight_height:float=180
    sensor_height:float=7.4,

    overlapfactor:float=0.1

    ratiowidth:float=0.5
    ratioheight:float=0.5   
        
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

# Helper funcs
def get_patches(image,coords:list):
    
    patches = list()
    
    # store patches
    for (x_left,x_right),(y_top,y_bottom) in coords:
        patches.append(image[:,y_top:y_bottom,x_left:x_right])
        
    return patches

def main(args:Args):

    dest = Path(args.dest)
    if not dest.exists():
        dest.mkdir(parents=True, exist_ok=True)

    images_paths = chain.from_iterable([Path(args.root).glob(p) for p in args.patterns])
    images_paths = list(images_paths)

    tile_metadata = dict()
    for img_path in tqdm(images_paths, desc='Exporting patches'):
        try:
            pil_img = Image.open(img_path)
        except :
            print("failed for: ",img_path,flush=True)
            continue
        img_tensor = torchvision.transforms.ToTensor()(pil_img)
        img_name = os.path.basename(img_path)


        # Cropping out image-level overlap
        height_overlap = math.ceil(args.rmheight * img_tensor.shape[1])
        width_overlap = math.ceil(args.rmwidth * img_tensor.shape[2])

        if height_overlap*width_overlap > 0 :
            img_tensor = img_tensor[:,height_overlap:-height_overlap, width_overlap:-width_overlap]
            print(f"Removing {2*width_overlap} pixels to the width; and {2*height_overlap} pixels to the height.")
        elif (height_overlap == 0) and (width_overlap != 0):
            img_tensor = img_tensor[:,:, width_overlap:-width_overlap]
        elif width_overlap == 0 and (height_overlap != 0):
            img_tensor = img_tensor[:,height_overlap:-height_overlap,:]
        
        # Computes tile width and height using the given ratios
        assert (args.ratiowidth<=1.0) and (args.ratioheight<=1.0), "The ratios should be at most 1.0"
        if args.ratiowidth > 0.0:
            width = math.ceil(img_tensor.shape[2]*args.ratiowidth)
        if args.ratioheight > 0.0:
            height = math.ceil(img_tensor.shape[1]*args.ratioheight)
        
        # checking overlapfactor provided
        assert args.overlapfactor<1, 'It should be less than 1.'

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
                    sensor_height=args.sensor_height,
                    flight_height=args.flight_height,
                )
            for (x_left,x_right),(y_top,y_bottom) in coords:
                x = (x_left+x_right)/2
                y = (y_top+y_bottom)/2
                gps = get_pixel_gps_coordinates(x=x,y=y,
                                                                W=image_width,
                                                                H=image_height,
                                                                lat_center=lat,lon_center=long,
                                                                gsd=gsd)
                gps = geopy.Point(gps[0],gps[1],alt)
                tile_gps_coords.append(str(gps))
        else:
            tile_gps_coords = [None for _ in range(len(coords))]

        tile_metadata[str(img_path)] = dict(tile_coordinates=coords,
                                            tile_gps_coord=tile_gps_coords,                    
                                            )

    # saving metdata
    json_path = Path(args.dest) / f"coordinates.json"
    with open(json_path, "w") as f:
        json.dump(tile_metadata, f, indent=1)

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

def match_tiles_gps(tile_data:dict,images_dir:str):
    """
    Match tiles gps coordinates to tile paths
    
    Args:
        tile_data (dict): tile data
        images_dir (str): directory containing tile images
    
    Returns:
        dict: tile metadata
    """
    tile_paths = get_images_paths(images_dir=images_dir)

    tiles_metadata = dict()
    for tile_path in tile_paths:
        image_name,index = Path(tile_path).stem.split('_')
        image_name = "".join(image_name.split('_')[0])
        index = int(index)
        
        metadata = tile_data.get(image_name)
        if metadata is None:
            print(f"Image {image_name} not found in metadata")
        else:
            try:
                tile_coords = metadata['tile_coordinates'][index]
                tile_gps = metadata['tile_gps_coord'][index]
                tiles_metadata[str(tile_path)] = dict(tile_coordinates=tile_coords,
                                                tile_gps_coord=tile_gps)
            except Exception as e:
                print(e,index,tile_path,len(metadata['tile_coordinates']))
                break

if __name__ == "__main__":
    args = Args(root=r"D:\savmap_dataset_v2\raw\images",
            overlapfactor=0.1,
            ratiowidth=0.5,
            ratioheight=0.33,
            rmheight=0.1,   
            rmwidth=0.1,
            flight_height=180,
            sensor_height=7.4,
            dest=r"D:\savmap_dataset_v2\slipts_tmp",
            save_coords_only=True,        
            )

    main(args)
