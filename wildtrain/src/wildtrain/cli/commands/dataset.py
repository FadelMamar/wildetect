"""Dataset-related CLI commands."""

import typer
import json
from pathlib import Path
from typing import Optional

from rich.progress import Progress, SpinnerColumn, TextColumn

from ...data.classification_datamodule import ClassificationDataModule, compute_dataset_stats
from .utils import console

dataset_app = typer.Typer(name="dataset", help="Dataset commands")


@dataset_app.command()
def stats(
    data_dir: Path = typer.Argument(..., help="Path to dataset directory"),
    split: str = typer.Option("train", help="Split to compute statistics for"),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for statistics"
    ),
) -> dict:
    """Get dataset statistics and information (mean, std)."""
    console.print(f"[bold green]Analyzing dataset at:[/bold green] {data_dir}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Computing dataset statistics...", total=None)

        # Create the data module and load data
        datamodule = ClassificationDataModule(
            root_data_directory=str(data_dir), batch_size=32, transforms=None, load_as_single_class=True
        )
        if split == "train":
            datamodule.setup(stage="fit")
            data = datamodule.train_dataset
        elif split == "val":
            datamodule.setup(stage="validate")
            data = datamodule.val_dataset
        elif split == "test":
            datamodule.setup(stage="test")
            data = datamodule.test_dataset
        else:
            raise ValueError(f"Invalid split: {split}")

        mean, std = compute_dataset_stats(
            data,
            batch_size=32,
            num_workers=0,
        )

        stats = {
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

        progress.update(task, description="Dataset analysis completed!")

        console.print("\n[bold blue]Dataset Statistics:[/bold blue]")
        console.print(f"  📊 Mean: {stats['mean']}")
        console.print(f"  📏 Std: {stats['std']}")

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            
            console.print(f"  💾 Statistics saved to: {output_file}")
        
        return stats
