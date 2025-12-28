from wildtrain.utils.io import load_all_detection_datasets

def main(root_data_directory:str,split:str):
    datasets = load_all_detection_datasets(root_data_directory=root_data_directory, split=split)
    print(f"Loaded {len(datasets)} datasets for split {split}")

if __name__ == "__main__":
    import fire
    fire.Fire(main)
