import json
import os
from pathlib import Path
import fire

def update_file_names(json_path, output_path=None):
    """
    Loads a COCO format JSON file and updates the 'file_name' field for all images
    following a specified logic.

    Args:
        json_path (str): Path to the input JSON file.
        output_path (str, optional): Path to save the updated JSON. Defaults to overwriting the input.
        update_logic (callable, optional): A function that takes the original file_name 
                                           and returns the updated file_name. 
                                           Defaults to extracting the basename (filename only) 
                                           from a Windows or Unix path.
    """

    update_logic = lambda x: f"{Path(x).parent.parent.name}/{Path(x).parent.name}/{Path(x).name}"

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Apply the logic to each image's file_name
    count = 0
    for image in data.get('images', []):
        if 'file_name' in image:
            original_path = image['file_name'].replace("D:\\","/").replace("\\","/")
            new_path = update_logic(original_path)
            if original_path != new_path:
                image['file_name'] = new_path
                count += 1

    if output_path is None:
        output_path = json_path

    # Save the updated JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"Updated {count} file paths. Saved to {output_path}")

def update_directory(directory:str):

    for p in Path(directory).glob("**/*.json"):
        try:
            update_file_names(p)
        except Expection as e:
            print(e)

if __name__ == "__main__":
    fire.Fire()
