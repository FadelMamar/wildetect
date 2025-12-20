#!/usr/bin/env python3
"""
Example script demonstrating how to compute mean and standard deviation
of images in a dataset for data preprocessing.
"""

from wildtrain.data.classification_datamodule import (
    ClassificationDataModule,
    compute_dataset_stats,
)


def main():
    """
    Example usage of the compute_dataset_stats function.
    """
    # Example configuration
    data_dir = "D:\PhD\workspace\data"  # Update this to your data directory
    batch_size = 32

    # Create the data module
    datamodule = ClassificationDataModule(
        root_data_directory=data_dir, batch_size=batch_size, transforms=None,load_as_single_class=True
    )

    # Setup the data module to load datasets
    datamodule.setup(stage="fit")

    print("\nComputing statistics using standalone function...")
    mean, std = compute_dataset_stats(
        datamodule.train_dataset,
        batch_size=batch_size,
        num_workers=0,
    )

    print(f"\nStandalone function results:")
    print(f"Mean: {mean.tolist()}")
    print(f"Std: {std.tolist()}")


if __name__ == "__main__":
    main()
