from wildtrain.converters.labelstudio_converter import LabelstudioConverter
from pathlib import Path
from wildtrain.config import ROOT
# Path to your Label Studio JSON annotation file
ls_json_path = r"D:\workspace\data\project-4-at-2025-07-14-10-55-95d5eea7.json" 

dotenv_path = Path(__file__).parent.parent / ".env" 

# Instantiate the converter
converter = LabelstudioConverter(dotenv_path=dotenv_path)

# Name for your dataset
dataset_name = "my_labelstudio_dataset"

# Convert the Label Studio JSON to COCO format (returns dataset_info and split_data)
dataset_info, split_data = converter.convert(
    input_file=ls_json_path,
    dataset_name=dataset_name,
    parse_ls_config=False,
    ls_xml_config=str(ROOT / "configs" / "label_studio_config.xml"),
    #
)

# Print the results
print("Dataset Info:")
print(dataset_info)
print(split_data)
