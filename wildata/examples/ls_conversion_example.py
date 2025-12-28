import fire

from wildata.converters import LabelstudioConverter
from pathlib import Path
from wildata.config import ROOT

# Path to your Label Studio JSON annotation file
ls_json_path = r"D:\workspace\repos\wildetect\Dry season - Kapiri Camp - 9-11, Rep 2.json"

def example():
    
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

def example2():
    from wildata.converters import LabelStudioParser

    parser = LabelStudioParser.from_file(ls_json_path)
    print(parser.get_summary())

    for ann in parser.iter_annotations():
        print(f"{ann.image_path}: {ann.label} @ {ann.bbox_pixel}")
        break

    df_annotations = parser.to_dataframe()
    print(df_annotations['task_id'].nunique(), parser.task_count)
    assert df_annotations['task_id'].nunique() == parser.task_count, f"Task count mismatch: {df_annotations['task_id'].nunique()} != {parser.task_count}"
    
    print(df_annotations.head(2))

if __name__ == "__main__":
    fire.Fire()
