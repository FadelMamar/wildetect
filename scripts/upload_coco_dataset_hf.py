import os
import re
import supervision as sv
from datasets import Dataset, Features, Value, Image as ImageFeature, List, get_dataset_split_names
from pathlib import Path
import fire
from PIL import Image
from tqdm import tqdm
import logging
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("upload_coco_hf.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / ".cache/huggingface/datasets"

def upload(images_dir:str, json_file:str, repository:str, skip_existing:bool=True)->Dataset:
    
    ds = sv.DetectionDataset.from_coco(
        images_directory_path=images_dir,
        annotations_path=json_file,
    )

    def gen():
        for name, image, annot in tqdm(ds, total=len(ds), desc="Processing images"):
            xyxy = annot.xyxy.tolist()
            class_id = annot.class_id.tolist()
            yield {"file_name": name, "image": Image.fromarray(image), "annotations": xyxy, "class_id": class_id}
    
    split_name = re.sub(r"[^\w.]", "", Path(json_file).stem)
    features = Features({
        'file_name': Value('string'),
        'image': ImageFeature(),
        'class_id': List(Value('int32')),
        'annotations': List(feature=List(feature=Value('float32')))
    })

    splits = get_dataset_split_names(repository)
    if skip_existing and (split_name in splits):
        logger.info(f"Split {split_name} already exists in repository {repository}, skipping")
        return

    logger.info(f"Generating dataset for {json_file}")

    hf_dataset = Dataset.from_generator(
        gen,
        features=features,
        split=split_name,
        cache_dir=str(CACHE_DIR),
        writer_batch_size=50,
    )
    
    logger.info(f"STARTING UPLOAD")

    hf_dataset.push_to_hub(repository, max_shard_size="1GB",num_proc=3,data_dir=split_name)

    logger.info(f"SUCCESS: Completed upload for {json_file}")

def upload_all(images_dir:str, json_dir:str, repository:str, remove_cache:bool=True):
    json_files = list(Path(json_dir).glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files to upload from {json_dir}")
    for json_file in json_files:
        try:
            upload(images_dir, str(json_file), repository)           
        except Exception as e:
            logger.error(f"FAILED to upload {json_file}: {e}", exc_info=True)
            continue
        finally:
            if os.path.exists(CACHE_DIR) and remove_cache:
                shutil.rmtree(CACHE_DIR)

if __name__ == "__main__":
    fire.Fire()
