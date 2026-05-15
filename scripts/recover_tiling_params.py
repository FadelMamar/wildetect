import argparse
import math
import sys
import subprocess
import yaml
import fire
from pathlib import Path
from itertools import product
from typing import List, Tuple
from itertools import chain

import cv2
import numpy as np

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

def find_tile_coordinates(parent_img_path: str, tile_paths: List[str]) -> List[Tuple[int, int]]:
    """Use template matching to find the top-left (X, Y) coordinate for each tile in the parent image."""
    parent_img = cv2.imread(str(parent_img_path))
    if parent_img is None:
        raise ValueError(f"Could not read parent image at {parent_img_path}")
    
    parent_gray = cv2.cvtColor(parent_img, cv2.COLOR_BGR2GRAY)
    
    tile_coords = []
    for tp in tile_paths:
        tile_img = cv2.imread(str(tp))
        if tile_img is None:
            raise ValueError(f"Could not read tile image at {tp}")
        
        tile_gray = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)
        
        result = cv2.matchTemplate(parent_gray, tile_gray, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        
        tile_coords.append(max_loc) # (X, Y)
        
    return tile_coords

def simulate_bounds(
    image_width: int, image_height: int, 
    rmheight: float, rmwidth: float, 
    ratiowidth: float, ratioheight: float, 
    overlapfactor: float
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Simulate the forward bounds generation, returning original-image relative bounds."""
    
    height_overlap = math.ceil(rmheight * image_height)
    width_overlap = math.ceil(rmwidth * image_width)
    
    w_crop = image_width
    h_crop = image_height
    
    if height_overlap*width_overlap > 0 :
        w_crop -= 2 * width_overlap
        h_crop -= 2 * height_overlap
    elif (height_overlap == 0) and (width_overlap != 0):
        w_crop -= 2 * width_overlap
    elif width_overlap == 0 and (height_overlap != 0):
        h_crop -= 2 * height_overlap
        
    width = math.ceil(w_crop * ratiowidth)
    height = math.ceil(h_crop * ratioheight)
    
    coords = get_coordinates(
        image_width=w_crop, tile_w=width, 
        image_height=h_crop, tile_h=height, 
        overlaping_factor=overlapfactor
    )
    
    # shift bounds back to the parent image coordinates
    shifted_coords = []
    for (x_left, x_right), (y_top, y_bottom) in coords:
        shifted_coords.append((
            (x_left + width_overlap, x_right + width_overlap),
            (y_top + height_overlap, y_bottom + height_overlap)
        ))
        
    return shifted_coords

def recover_parameters(parent_img_path: str, tiles_dir: str, config_path: str):
    parent_img = cv2.imread(str(parent_img_path))
    if parent_img is None:
        raise ValueError(f"Could not read parent image at {parent_img_path}")
    H, W = parent_img.shape[:2]
    
    tile_paths = sorted(list(Path(tiles_dir).glob("*.JPG")))
    parent_stem = Path(parent_img_path).stem
    parent_tiles = [tp for tp in tile_paths if tp.stem.startswith(parent_stem)]
    
    if not parent_tiles:
        print(f"No tiles found for {parent_stem} in {tiles_dir}")
        return
        
    print(f"Found {len(parent_tiles)} tiles for {parent_stem}")
    
    # Measure tile dimensions
    sample_tile = cv2.imread(str(parent_tiles[0]))
    tile_h, tile_w = sample_tile.shape[:2]
    
    print("Running template matching...")
    matched_coords = find_tile_coordinates(parent_img_path, parent_tiles)
    
    X_coords = [c[0] for c in matched_coords]
    Y_coords = [c[1] for c in matched_coords]
    
    min_X = min(X_coords)
    min_Y = min(Y_coords)
    
    width_overlap = min_X
    height_overlap = min_Y
    
    est_rmwidth = width_overlap / W
    est_rmheight = height_overlap / H
    
    w_crop = W - 2 * width_overlap
    h_crop = H - 2 * height_overlap
    
    est_ratiowidth = tile_w / w_crop
    est_ratioheight = tile_h / h_crop
    
    # find step in X
    unique_X = sorted(list(set(X_coords)))
    if len(unique_X) > 1:
        step_x = unique_X[1] - unique_X[0]
        # (1 - overlapfactor) * tile_w = step_x => overlapfactor = 1 - (step_x / tile_w)
        est_overlapfactor = 1 - (step_x / tile_w)
    else:
        unique_Y = sorted(list(set(Y_coords)))
        if len(unique_Y) > 1:
            step_y = unique_Y[1] - unique_Y[0]
            est_overlapfactor = 1 - (step_y / tile_h)
        else:
            est_overlapfactor = 0.2
            
    print("\n--- Estimated Parameters ---")
    print(f"rmwidth:       {est_rmwidth:.4f}  (width_overlap={width_overlap})")
    print(f"rmheight:      {est_rmheight:.4f}  (height_overlap={height_overlap})")
    print(f"ratiowidth:    {est_ratiowidth:.4f}")
    print(f"ratioheight:   {est_ratioheight:.4f}")
    print(f"overlapfactor: {est_overlapfactor:.4f}")
    
    # Grid search around estimates
    def get_search_range(est, step, count=10, min_val=0.0):
        # returns np array around est
        vals = np.arange(est - count*step, est + count*step + step/2, step)
        return vals[(vals >= min_val) & (vals <= 1)]
        
    # the original script used steps of 0.01 for rmh/rmw and 0.1 for overlapfactor
    # but we will do a finer search around the exact estimate
    rmw_vals = get_search_range(est_rmwidth, 0.01, count=5)
    rmh_vals = get_search_range(est_rmheight, 0.01, count=5)
    rw_vals  = [1.0, 0.5, 1/3, 0.3336, est_ratiowidth] # test specific and estimated
    rh_vals  = [1.0, 0.5, 1/3, 0.3336, est_ratioheight]
    of_vals  = get_search_range(est_overlapfactor, 0.01, count=5)
    
    # Also add standard values just in case
    rw_vals = sorted(list(set(rw_vals)))
    rh_vals = sorted(list(set(rh_vals)))
    
    print("\nRunning targeted parameter sweep...")
    combos = list(product(rmw_vals, rmh_vals, rw_vals, rh_vals, of_vals))
    print(f"Total combinations to check: {len(combos)}")
    
    target_top_lefts = sorted([(x, y) for x, y in matched_coords])
    
    best_combo = None
    best_error = float('inf')
    
    for rmw, rmh, rw, rh, of in combos:
        try:
            sim_bounds = simulate_bounds(W, H, rmh, rmw, rw, rh, of)
            sim_top_lefts = sorted([(x_left, y_top) for (x_left, x_right), (y_top, y_bottom) in sim_bounds])
            
            if len(sim_top_lefts) != len(target_top_lefts):
                continue
                
            # Error is sum of absolute differences in coordinates
            error = sum(abs(tx - sx) + abs(ty - sy) for (tx, ty), (sx, sy) in zip(target_top_lefts, sim_top_lefts))
            
            if error < best_error:
                best_error = error
                best_combo = (rmw, rmh, rw, rh, of)
                
            if error == 0:
                print("\n*** Exact match found! ***")
                best_error = 0
                best_combo = (rmw, rmh, rw, rh, of)
                break
                
        except Exception:
            continue
            
    if best_combo is not None:
        rmw, rmh, rw, rh, of = best_combo
        print("\n--- Recovered Parameters ---")
        print(f"rmwidth:       {rmw:.4f}")
        print(f"rmheight:      {rmh:.4f}")
        print(f"ratiowidth:    {rw:.4f}")
        print(f"ratioheight:   {rh:.4f}")
        print(f"overlapfactor: {of:.4f}")
        print(f"Match Error:   {best_error}")
        
        sim_bounds = simulate_bounds(W, H, rmh, rmw, rw, rh, of)
        print(f"Generated Tiles: {len(sim_bounds)}")
        
        print("\n--- Validating with tile_gps_matching.py ---")
        try:
            cmd = [
                "uv", "run", "scripts/tile_gps_matching.py", "run",
                f"--config={config_path}",
                f"--rmwidth={rmw}",
                f"--rmheight={rmh}",
                f"--ratiowidth={rw}",
                f"--ratioheight={rh}",
                f"--overlapfactor={of}",
                "--debug=True"
            ]
            print(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                print("Validation successful! The pipeline ran without errors.")
            else:
                print(f"Validation failed (exit code {result.returncode}).")
                print(result.stderr)
        except Exception as e:
            print(f"Could not run validation: {e}")
            
    else:
        print("\nFailed to find matching parameters.")


def _is_ignored_image_path(path: Path) -> bool:
    """Ignore OS-generated image artifacts such as macOS AppleDouble files."""
    return path.name.startswith("._")

def load_images_paths(image_dir:str)->list[str]:
    patterns = tuple(
            f"**/{ext}"
            for base in ("*.jpg", "*.jpeg", "*.png")
            for ext in (base, base.upper())
        )
    images_paths = chain.from_iterable([Path(image_dir).glob(p) for p in patterns])
    images_paths = sorted([p for p in set(images_paths) if not _is_ignored_image_path(p)])
    return list(map(str,images_paths))

def main(config: str = "config/tile-gps-matching.yaml", parent_image: str = "", tiles_dir: str = ""):
    """
    Recover tiling parameters using template matching.
    
    Args:
        config: Path to the YAML config file to read defaults from and validate against.
        parent_image: Path to the parent image (default auto-detected from config).
        tiles_dir: Path to the directory containing tiled images (default auto-detected from config).
    """
    default_parent = ""
    default_tiles = ""
    
    config_path = Path(config)
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                cfg_data = yaml.safe_load(f)
            
            root_dir = Path(cfg_data.get("root", ""))
            default_tiles = cfg_data.get("tiled_images_folder", "")
            
            if root_dir.exists():
                images = load_images_paths(root_dir)
                if images:
                    default_parent = str(images[0])
        except Exception as e:
            print(f"Warning: Could not read defaults from {config_path}: {e}")

    final_parent = parent_image if parent_image else default_parent
    final_tiles = tiles_dir if tiles_dir else default_tiles

    if not final_parent or not final_tiles:
        print("Error: Could not determine default paths from config, and arguments were not provided.")
        sys.exit(1)
    
    recover_parameters(final_parent, final_tiles, str(config_path))


if __name__ == '__main__':
    fire.Fire(main)
