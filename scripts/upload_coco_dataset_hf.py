import os
import supervision as sv
from datasets import Dataset, Features, Value, Image as ImageFeature, List
from pathlib import Path
import fire
from PIL import Image
from tqdm import tqdm

def upload(images_dir:str, json_file:str, repository:str)->Dataset:
    
    ds = sv.DetectionDataset.from_coco(
        images_directory_path=images_dir,
        annotations_path=json_file,
    )

    def gen():
        for name, image, annot in tqdm(ds, total=len(ds), desc="Processing images"):
            xyxy = annot.xyxy.tolist()
            class_id = annot.class_id.tolist()
            yield {"file_name": name, "image": Image.fromarray(image), "annotations": xyxy, "class_id": class_id}
    
    split_name = Path(json_file).stem.replace(" ","").replace(",","").replace("-","")
    features = Features({
        'file_name': Value('string'),
        'image': ImageFeature(),
        'class_id': List(Value('int32')),
        'annotations': List(feature=List(feature=Value('float32')))
    })

    hf_dataset = Dataset.from_generator(
        gen,
        features=features,
        split=split_name,
        cache_dir='.cache/huggingface/datasets',
        
    )
    
    hf_dataset.push_to_hub(repository, max_shard_size="1GB",num_proc=3,data_dir=split_name)

if __name__ == "__main__":
    fire.Fire()
