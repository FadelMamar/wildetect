"""
Command-line interface for the WildTrain data pipeline using Typer.
"""

import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from ..adapters.utils import ExifGPSManager
from ..config import (
    ROOT,
    BulkCreateROIDatasetConfig,
    BulkImportDatasetConfig,
    ExifGPSUpdateConfig,
    ImportDatasetConfig,
    ROIDatasetConfig,
)
from ..converters import LabelstudioConverter
from ..logging_config import setup_logging
from ..pipeline import DataPipeline
from ..visualization import FiftyOneManager
from .import_logic import _import_dataset_core, import_one_worker
from .roi_logic import create_roi_dataset_core, create_roi_one_worker
from .utils import create_dataset_name

__version__ = "0.1.0"

app = typer.Typer(
    name="wildata",
    help="Data Pipeline - Manage datasets in master format and create framework-specific formats",
    add_completion=False,
    rich_markup_mode="rich",
)

# create logs directory if it doesn't exist
(ROOT / "logs").mkdir(parents=True, exist_ok=True)


@app.command()
def version():
    """Show version information."""
    typer.echo(f"wildata version {__version__}")


@app.command()
def import_dataset(
    config_file: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to YAML config file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Import a dataset from various formats into the WildData pipeline."""
    log_file = (
        ROOT
        / "logs"
        / "import_datasets"
        / f"import_dataset_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
    setup_logging(log_file=log_file.as_posix())

    try:
        config = ImportDatasetConfig.from_yaml(config_file)
    except Exception:
        typer.echo(f"[ERROR] Failed to load config file: {traceback.format_exc()}")
        raise typer.Exit(1)

    _import_dataset_core(config, verbose)


@app.command()
def bulk_import_datasets(
    config_file: str = typer.Option(
        ..., "--config", "-c", help="Path to YAML config file (YAML only)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    num_workers: int = typer.Option(
        2, "--num-workers", "-n", help="Number of workers to use for bulk import"
    ),
):
    """Bulk import multiple datasets from all files in a directory.

    The config YAML should contain:
      source_path: path/to/directory  # directory containing dataset files
      source_format: yolo  # or coco, ls
      ... (other config fields)

    Each file in the directory will be imported as a dataset, with the dataset name derived from the filename (without extension).
    """

    log_file = (
        ROOT
        / "logs"
        / "import_datasets"
        / f"bulk_import_datasets_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
    setup_logging(log_file=log_file.as_posix())

    if not (config_file.endswith(".yaml") or config_file.endswith(".yml")):
        typer.echo(
            "[ERROR] Only YAML config files are supported for bulk import. Please provide a .yaml or .yml file."
        )
        raise typer.Exit(1)
    try:
        config = BulkImportDatasetConfig.from_yaml(config_file)
    except Exception:
        typer.echo(f"[ERROR] Failed to load YAML config file: {traceback.format_exc()}")
        raise typer.Exit(1)

    # Validate directory
    files = []
    for source_path in config.source_paths:
        if not os.path.isdir(source_path):
            typer.echo(f"[ERROR] source_path must be a directory: {source_path}")
            raise typer.Exit(1)
        files.extend(
            [
                f.resolve().as_posix()
                for f in Path(source_path).iterdir()
                if f.is_file() and not f.name.startswith(".")
            ]
        )

    if not files:
        typer.echo(f"[ERROR] No files found in directories: {config.source_paths}")
        raise typer.Exit(1)
    dataset_names = [create_dataset_name(f) for f in files]
    formats = [config.source_format] * len(files)

    # Convert config to dict for pickling
    config_dict = config.model_dump()
    args_list = [
        (i, src, name, fmt, config_dict, verbose)
        for i, (src, name, fmt) in enumerate(zip(files, dataset_names, formats))
    ]

    results = [None] * len(files)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(import_one_worker, args): i
            for i, args in enumerate(args_list)
        }
        for future in as_completed(future_to_idx):
            i, name, success, msg = future.result()
            results[i] = success
            if msg:
                typer.echo(msg)
            elif success:
                typer.echo(
                    f"[SUCCESS] Import finished for '{name}' [{i + 1}/{len(files)}]"
                )
            else:
                typer.echo(f"[ERROR] Import failed for '{name}' [{i + 1}/{len(files)}]")
    typer.echo(f"\nBulk import complete. {sum(results)}/{len(results)} succeeded.")


@app.command()
def create_roi_dataset(
    config_file: str = typer.Option(
        ..., "--config", "-c", help="Path to YAML config file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Create an ROI dataset from a source dataset using a YAML config file."""
    log_file = (
        ROOT
        / "logs"
        / "roi_creation"
        / f"create_roi_dataset_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
    setup_logging(log_file=log_file.as_posix())

    # Only config file is allowed
    try:
        config = ROIDatasetConfig.from_yaml(config_file)
    except Exception:
        typer.echo(f"[ERROR] Failed to load config file: {traceback.format_exc()}")
        raise typer.Exit(1)

    create_roi_dataset_core(config, verbose)


@app.command()
def bulk_create_roi_datasets(
    config_file: str = typer.Option(
        ..., "--config", "-c", help="Path to YAML config file (YAML only)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    num_workers: int = typer.Option(
        2, "--num-workers", "-n", help="Number of workers to use for bulk ROI creation"
    ),
):
    """Bulk create ROI datasets from all files in a directory (multiprocessing).
    Each file in the directory will be used to create an ROI dataset, with the dataset name derived from the filename (without extension).
    """
    log_file = (
        ROOT
        / "logs"
        / "roi_creation"
        / f"bulk_create_roi_datasets_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
    setup_logging(log_file=log_file.as_posix())

    if not (config_file.endswith(".yaml") or config_file.endswith(".yml")):
        typer.echo(
            "[ERROR] Only YAML config files are supported for bulk ROI creation. Please provide a .yaml or .yml file."
        )
        raise typer.Exit(1)
    try:
        config = BulkCreateROIDatasetConfig.from_yaml(config_file)
    except Exception:
        typer.echo(f"[ERROR] Failed to load YAML config file: {traceback.format_exc()}")
        raise typer.Exit(1)

    # Validate directory
    files = []
    for source_path in config.source_paths:
        if not os.path.isdir(source_path):
            typer.echo(f"[ERROR] source_path must be a directory: {source_path}")
            raise typer.Exit(1)
        files.extend(
            [
                f.resolve().as_posix()
                for f in Path(source_path).iterdir()
                if f.is_file() and not f.name.startswith(".")
            ]
        )
    if not files:
        typer.echo(f"[ERROR] No files found in directories: {config.source_paths}")
        raise typer.Exit(1)
    dataset_names = [create_dataset_name(f) for f in files]

    config_dict = config.model_dump()
    args_list = [
        (i, src, name, config_dict, verbose)
        for i, (src, name) in enumerate(zip(files, dataset_names))
    ]

    results = [None] * len(files)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(create_roi_one_worker, args): i
            for i, args in enumerate(args_list)
        }
        for future in as_completed(future_to_idx):
            i, name, success, msg = future.result()
            results[i] = success
            if msg:
                typer.echo(msg)
            elif success:
                typer.echo(
                    f"[SUCCESS] ROI creation finished for '{name}' [{i + 1}/{len(files)}]"
                )
            else:
                typer.echo(
                    f"[ERROR] ROI creation failed for '{name}' [{i + 1}/{len(files)}]"
                )
    typer.echo(
        f"\nBulk ROI creation complete. {sum(results)}/{len(results)} succeeded."
    )


@app.command()
def list_datasets(
    root: str = typer.Option(
        "data", "--root", "-r", help="Root directory for data storage"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """List all available datasets."""
    try:
        pipeline = DataPipeline(root=root, split_name="train")
        datasets = pipeline.list_datasets()

        if not datasets:
            typer.echo("📭 No datasets found")
            return

        typer.echo(f"📚 Found {len(datasets)} dataset(s):")
        for dataset in datasets:
            typer.echo(f"   • {dataset['dataset_name']}")
            if verbose:
                typer.echo(
                    f"     Number of images: {dataset.get('total_images', 'Unknown')}"
                )
                typer.echo(
                    f"     Number of annotations: {dataset.get('total_annotations', 'Unknown')}"
                )
                typer.echo(f"     Splits: {dataset.get('splits', 'Unknown')}")

    except Exception:
        typer.echo(f"[ERROR] Failed to list datasets: {str(traceback.format_exc())}")
        raise typer.Exit(1)


@app.command()
def visualize_classification(
    dataset_name: str = typer.Argument(..., help="Name for the FiftyOne dataset"),
    root_data_directory: Optional[str] = typer.Option(
        None, "--root", help="Root data directory for classification dataset"
    ),
    load_as_single_class: Optional[bool] = typer.Option(
        False, "--single-class", help="Load as single class (True/False)"
    ),
    background_class_name: Optional[str] = typer.Option(
        "background", "--background-class", help="Background class name"
    ),
    single_class_name: Optional[str] = typer.Option(
        "wildlife", "--single-class-name", help="Single class name"
    ),
    keep_classes: Optional[str] = typer.Option(
        None, "--keep-classes", help="Comma-separated list of classes to keep"
    ),
    discard_classes: Optional[str] = typer.Option(
        None, "--discard-classes", help="Comma-separated list of classes to discard"
    ),
    split: str = typer.Option(
        "train", "--split", help="Dataset split (train/val/test)"
    ),
):
    """Visualize a classification dataset in FiftyOne (wraps import_classification_data)."""
    # Parse keep/discard classes if provided
    keep_classes_list = keep_classes.split(",") if keep_classes else None
    discard_classes_list = discard_classes.split(",") if discard_classes else None

    mgr = FiftyOneManager()
    mgr.import_classification_data(
        root_data_directory=root_data_directory or "",
        dataset_name=dataset_name,
        load_as_single_class=load_as_single_class or False,
        background_class_name=background_class_name or "background",
        single_class_name=single_class_name or "wildlife",
        keep_classes=keep_classes_list,
        discard_classes=discard_classes_list,
        split=split,
    )
    typer.echo(
        f"[SUCCESS] Visualization launched in FiftyOne for dataset '{dataset_name}' (split: {split})"
    )


@app.command()
def visualize_detection(
    dataset_name: str = typer.Argument(..., help="Name for the FiftyOne dataset"),
    root_data_directory: str = typer.Option(
        ..., "--root", help="Root data directory for detection dataset"
    ),
    split: str = typer.Option(
        "train", "--split", help="Dataset split (train/val/test)"
    ),
):
    """Visualize a detection dataset in FiftyOne (wraps import_detection_data)."""
    mgr = FiftyOneManager()
    mgr.import_detection_data(
        root_data_directory=root_data_directory,
        dataset_name=dataset_name,
        split=split,
    )
    typer.echo(
        f"[SUCCESS] Visualization launched in FiftyOne for detection dataset '{dataset_name}' (split: {split})"
    )


@app.command()
def update_gps_from_csv(
    config_file: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to YAML config file"
    ),    
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Update EXIF GPS data for images using coordinates from a CSV file."""
    log_file = (
        ROOT
        / "logs"
        / "gps_update"
        / f"update_gps_from_csv_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
    setup_logging(log_file=log_file.as_posix())

    try:
        config = ExifGPSUpdateConfig.from_yaml(config_file)
    except Exception:
        typer.echo(f"[ERROR] Failed to load config file: {traceback.format_exc()}")
        raise typer.Exit(1)
    
    try:
        if verbose:
            typer.echo(f"[FOLDER] Image folder: {config.image_folder}")
            typer.echo(f"[FILE] CSV file: {config.csv_path}")
            typer.echo(f"[OUTPUT] Output directory: {config.output_dir}")
            typer.echo(f"[SKIP] Skip rows: {config.skip_rows}")
            typer.echo(f"[COLUMN] Filename column: {config.filename_col}")
            typer.echo(f"[LAT] Latitude column: {config.lat_col}")
            typer.echo(f"[LON] Longitude column: {config.lon_col}")
            typer.echo(f"[ALT] Altitude column: {config.alt_col}")

        # Initialize ExifGPSManager and update GPS data
        gps_manager = ExifGPSManager()
        gps_manager.update_folder_from_csv(
            image_folder=config.image_folder,
            csv_path=config.csv_path,
            output_dir=config.output_dir,
            skip_rows=config.skip_rows,
            filename_col=config.filename_col,
            lat_col=config.lat_col,
            lon_col=config.lon_col,
            alt_col=config.alt_col,
        )

        typer.echo(
            f"[SUCCESS] Successfully updated GPS data for images in '{config.image_folder}'"
        )
        typer.echo(f"[OUTPUT] Updated images saved to: {config.output_dir}")

    except Exception as e:
        typer.echo(f"[ERROR] Failed to update GPS data: {str(e)}")
        if verbose:
            typer.echo(f"   Traceback: {traceback.format_exc()}")
        raise typer.Exit(1)

@app.command()
def convert_ls_to_coco(
    input_file: str = typer.Argument(
        ..., help="Path to Label Studio JSON annotation file"
    ),
    out_file: Optional[str] = typer.Option(
        None, "--out-file", "-o", help="Path to write COCO JSON output"
    ),
    ls_xml_config: Optional[str] = typer.Option(
        None,
        "--ls-xml-config",
        help="Optional Label Studio XML config to preserve category IDs",
    ),
    dotenv_path: Optional[str] = typer.Option(
        None,
        "--dotenv-path",
        help="Path to .env with LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY",
    ),
    parse_ls_config: bool = typer.Option(
        False,
        "--parse-ls-config",
        help="Use Label Studio project config from API to derive categories",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Convert a Label Studio JSON export to COCO and save it."""
    try:
        input_path = Path(input_file)
        if not input_path.is_file():
            typer.echo(f"[ERROR] Input file not found: {input_file}")
            raise typer.Exit(1)
        if input_path.suffix.lower() != ".json":
            typer.echo(f"[ERROR] Input file must be a JSON file: {input_file}")
            raise typer.Exit(1)

        output_path = Path(out_file) if out_file else input_path.with_name(
            f"{input_path.stem}_coco.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        converter = LabelstudioConverter(dotenv_path=dotenv_path)
        category_mapping = None

        if ls_xml_config is not None:
            xml_path = Path(ls_xml_config)
            if not xml_path.is_file():
                typer.echo(f"[ERROR] Label Studio XML config not found: {ls_xml_config}")
                raise typer.Exit(1)
            category_mapping = converter.get_category_mapping(ls_xml_config=ls_xml_config)
            if verbose:
                typer.echo(
                    f"[INFO] Loaded category mapping from XML with {len(category_mapping)} class(es)"
                )
        elif parse_ls_config:
            parsed_config = converter.get_ls_parsed_config(ls_json_path=str(input_path))
            if parsed_config:
                category_mapping = converter.get_category_mapping(
                    parsed_config=parsed_config
                )
                if verbose:
                    typer.echo(
                        "[INFO] Loaded category mapping from Label Studio parsed config "
                        f"with {len(category_mapping)} class(es)"
                    )
            elif verbose:
                typer.echo(
                    "[INFO] Label Studio parsed config unavailable; using default category order"
                )
        else:
            raise ValueError()

        coco_annotations = converter.convert_ls_json_to_coco(
            input_file=str(input_path),
            category_mapping=category_mapping,
            out_file_name=str(output_path),
        )

        typer.echo(f"[SUCCESS] COCO annotations saved to: {output_path.as_posix()}")
        if verbose:
            typer.echo(f"   Images: {len(coco_annotations.get('images', []))}")
            typer.echo(f"   Annotations: {len(coco_annotations.get('annotations', []))}")
            typer.echo(f"   Categories: {len(coco_annotations.get('categories', []))}")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"[ERROR] Failed to convert Label Studio JSON to COCO: {str(e)}")
        if verbose:
            typer.echo(f"   Traceback: {traceback.format_exc()}")
        raise typer.Exit(1)


# Add API subcommand
from ..api.cli import app as api_app

app.add_typer(api_app, name="api", help="API server commands")
