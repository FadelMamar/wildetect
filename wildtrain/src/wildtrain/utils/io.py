import yaml
import os
import traceback
from pathlib import Path
from typing import Optional
from PIL import Image, ImageOps

from wildata.pipeline.path_manager import PathManager

from .logging import get_logger

logger = get_logger(__name__)

def read_image(image_path: str) -> Image.Image:
    """Load an image from a file path."""
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    return image

def load_yaml(path: str):
    with open(path, "r",encoding="utf-8") as file:
        return yaml.safe_load(file)

def save_yaml(cfg: dict, save_path: str, mode="w"):
    try:
        with open(save_path, mode, encoding="utf-8") as file:
            yaml.dump(cfg, file)
        logger.info(f"Successfully saved yaml to {save_path}")
    except Exception:
        logger.error(f"Error saving yaml to {save_path}: {traceback.format_exc()}")
        
def remove_label_cache(data_config_yaml: str):
    """
    Remove the labels.cache files from the dataset directories specified in the YOLO data config YAML.

    Args:
        data_config_yaml (str): Path to the YOLO data config YAML file.
    """
    # Remove labels.cache
    yolo_config = load_yaml(data_config_yaml)
    root = yolo_config["path"]
    for split in ["train", "val", "test"]:
        # try:
        if split in yolo_config.keys():
            for p in yolo_config[split]:
                path = os.path.join(root, p, "../labels.cache")
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Removing: {os.path.join(root, p, '../labels.cache')}")
        else:
            logger.info(f"split={split} does not exist.")

def merge_data_cfg(root_data_directory:Optional[str]=None,data_configs:Optional[list[str]]=None,output_path:Optional[str]=None,force_merge:bool=True):
    
    assert root_data_directory is not None or data_configs is not None, "Either root_data_directory or data_configs must be provided"


    if root_data_directory is not None:
        path_manager = PathManager(Path(root_data_directory))
        data_configs = [path/"data.yaml" for path in path_manager.yolo_formats_dir.iterdir() if path.is_dir()]
    
    assert len(data_configs) > 0, "No data.yaml files found"

    
    roots = []
    train_paths = []
    val_paths = []
    test_paths = []
    names = []
    nc = []
    
    # load data.yaml for each dataset
    for data_cfg in data_configs:
        data_cfg = load_yaml(data_cfg)
        roots.append(data_cfg["path"])
        train_paths.append(data_cfg["train"])
        val_paths.append(data_cfg["val"])
        test_paths.append(data_cfg["test"])
        names.append(data_cfg["names"])
        if "nc" in data_cfg:
            nc.append(data_cfg["nc"])
        else:
            nc.append(len(data_cfg["names"]))
    
    # unify the cfgs
    unified_cfg = {
        "path": roots[0],
        "train": [],
        "val": [],
        "test": [],
        "names": names[0],
        "nc": nc[0]
    }

    # get common prefix for all paths
    common_path = os.path.commonpath(roots)
    unified_cfg["path"] = common_path

    # check that all nc are the same
    if not force_merge:
        for n in nc:
            if n != nc[0]:
                raise ValueError(f"nc are not the same: {n} != {nc[0]}")
        unified_cfg["nc"] = nc[0]
    else:
        unified_cfg["nc"] = max(nc)
    
    # check that all names are the same
    if not force_merge:
        for name in names:
            if name != names[0]:
                raise ValueError(f"Names are not the same: {name} != {names[0]}")
        unified_cfg["names"] = names[0]
    else:
        unified_cfg["names"] = {i: f"class_{i}" for i in range(max(nc))}
    
    for root, train_path, val_path, test_path in zip(roots, train_paths, val_paths, test_paths):
        
        train_path = os.path.join(root, train_path)
        val_path = os.path.join(root, val_path)
        test_path = os.path.join(root, test_path)

        if os.path.exists(train_path):
            unified_cfg["train"].append(os.path.relpath(train_path, common_path))
        if os.path.exists(val_path):
            unified_cfg["val"].append(os.path.relpath(val_path, common_path))
        if os.path.exists(test_path):
            unified_cfg["test"].append(os.path.relpath(test_path, common_path))
    
    if len(unified_cfg["test"]) == 0:
        unified_cfg.pop("test")
    
    if len(unified_cfg["val"]) == 0:
        unified_cfg.pop("val")
    
    if len(unified_cfg["train"]) == 0:
        raise ValueError("No train paths found")

    # save the unified cfg
    if output_path is not None:
        save_yaml(unified_cfg, output_path, mode="w")

    return unified_cfg