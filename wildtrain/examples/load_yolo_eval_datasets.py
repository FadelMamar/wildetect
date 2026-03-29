from wildtrain.utils.io import load_all_detection_datasets


def main(root_data_directory: str = r"D:\PhD\workspace\data", split: str = "val"):
    datasets = load_all_detection_datasets(
        root_data_directory=root_data_directory, split=split
    )
    print(f"Loaded {len(datasets)} datasets for split {split}")


if __name__ == "__main__":
    main()
