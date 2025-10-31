"""
Core import logic for CLI commands.
"""

import traceback
from typing import Dict, Optional

import typer
from pydantic import ValidationError

from ..config import AugmentationConfig, ROIConfig, TilingConfig
from ..pipeline import DataPipeline
from ..transformations import (
    AugmentationTransformer,
    BoundingBoxClippingTransformer,
    TilingTransformer,
    TransformationPipeline,
)
from .models import ImportDatasetConfig


def _import_dataset_core(config: ImportDatasetConfig, verbose: bool = False) -> bool:
    """Core logic for importing a dataset, shared by CLI and bulk import."""
    # Convert ROI config if provided
    roi_config = None
    if not config.disable_roi and config.roi_config:
        roi_config = ROIConfig(
            random_roi_count=config.roi_config.random_roi_count,
            roi_box_size=config.roi_config.roi_box_size,
            min_roi_size=config.roi_config.min_roi_size,
            dark_threshold=config.roi_config.dark_threshold,
            background_class=config.roi_config.background_class,
            save_format=config.roi_config.save_format,
            quality=config.roi_config.quality,
        )

    # Create transformation pipeline if configured
    transformation_pipeline = None
    if config.transformations:
        if verbose:
            typer.echo(f"[INFO] Creating transformation pipeline...")

        transformation_pipeline = TransformationPipeline()

        # Add bbox clipping transformer
        if (
            config.transformations.enable_bbox_clipping
            and config.transformations.bbox_clipping
        ):
            bbox_config = config.transformations.bbox_clipping
            bbox_transformer = BoundingBoxClippingTransformer(
                tolerance=bbox_config.tolerance, skip_invalid=bbox_config.skip_invalid
            )
            transformation_pipeline.add_transformer(bbox_transformer)
            if verbose:
                typer.echo(
                    f"   Added BoundingBoxClippingTransformer (tolerance: {bbox_config.tolerance})"
                )

        # Add augmentation transformer
        if (
            config.transformations.enable_augmentation
            and config.transformations.augmentation
        ):
            aug_config = config.transformations.augmentation
            aug_transformer = AugmentationTransformer(
                config=AugmentationConfig(
                    rotation_range=aug_config.rotation_range,
                    probability=aug_config.probability,
                    brightness_range=aug_config.brightness_range,
                    scale=aug_config.scale,
                    translate=aug_config.translate,
                    shear=aug_config.shear,
                    contrast_range=aug_config.contrast_range,
                    noise_std=aug_config.noise_std,
                    seed=aug_config.seed,
                    num_transforms=aug_config.num_transforms,
                )
            )
            transformation_pipeline.add_transformer(aug_transformer)
            if verbose:
                typer.echo(
                    f"   Added AugmentationTransformer (num_transforms: {aug_config.num_transforms})"
                )

        # Add tiling transformer
        if config.transformations.enable_tiling and config.transformations.tiling:
            tiling_config = config.transformations.tiling
            tiling_transformer = TilingTransformer(
                config=TilingConfig(
                    tile_size=tiling_config.tile_size,
                    stride=tiling_config.stride,
                    min_visibility=tiling_config.min_visibility,
                    max_negative_tiles_in_negative_image=tiling_config.max_negative_tiles_in_negative_image,
                    negative_positive_ratio=tiling_config.negative_positive_ratio,
                    dark_threshold=tiling_config.dark_threshold,
                )
            )
            transformation_pipeline.add_transformer(tiling_transformer)
            if verbose:
                typer.echo(
                    f"   Added TilingTransformer (tile_size: {tiling_config.tile_size}, stride: {tiling_config.stride})"
                )

    # Execute import
    try:
        if verbose:
            typer.echo(f"[INFO] Creating data pipeline...")
            typer.echo(f"   Root: {config.root}")
            typer.echo(f"   Split: {config.split_name}")
            typer.echo(f"   DVC enabled: {config.enable_dvc}")
            if transformation_pipeline:
                typer.echo(f"   Transformers: {len(transformation_pipeline)}")

        pipeline = DataPipeline(
            root=config.root,
            split_name=config.split_name,
            enable_dvc=config.enable_dvc,
            transformation_pipeline=transformation_pipeline,
        )

        if verbose:
            typer.echo(f"[INFO] Importing dataset...")
            typer.echo(f"   Source: {config.source_path}")
            typer.echo(f"   Format: {config.source_format}")
            typer.echo(f"   Name: {config.dataset_name}")
            typer.echo(f"   Mode: {config.processing_mode}")

        result = pipeline.import_dataset(
            source_path=config.source_path,
            source_format=config.source_format,
            dataset_name=config.dataset_name,
            processing_mode=config.processing_mode,
            track_with_dvc=config.track_with_dvc,
            bbox_tolerance=config.bbox_tolerance,
            roi_config=roi_config,
            dotenv_path=config.dotenv_path,
            ls_xml_config=config.ls_xml_config,
            ls_parse_config=config.ls_parse_config,
        )

        if result["success"]:
            typer.echo(
                f"[SUCCESS] Successfully imported dataset '{config.dataset_name}'"
            )
            if verbose:
                typer.echo(f"   Dataset info: {result['dataset_info_path']}")
                typer.echo(f"   Framework paths: {result['framework_paths']}")
                typer.echo(f"   Processing mode: {result['processing_mode']}")
                typer.echo(f"   DVC tracked: {result['dvc_tracked']}")
        else:
            typer.echo(f"[ERROR] Failed to import dataset: {result['error']}")
            if "validation_errors" in result and result["validation_errors"]:
                typer.echo("   Validation errors:")
                for error in result["validation_errors"]:
                    typer.echo(f"     - {error}")
            if "hints" in result and result["hints"]:
                typer.echo("   Hints:")
                for hint in result["hints"]:
                    typer.echo(f"     - {hint}")
            return False
        return True
    except ValidationError as e:
        typer.echo(f"[ERROR] Configuration validation error:")
        for error in e.errors():
            typer.echo(f"   {error['loc'][0]}: {error['msg']}")
        return False
    except Exception as e:
        typer.echo(f"[ERROR] Import failed: {str(e)}")
        if verbose:
            typer.echo(f"   Traceback: {traceback.format_exc()}")
        return False


def import_one_worker(args) -> tuple:
    """Top-level worker for ProcessPoolExecutor in bulk_import_datasets."""
    i, src, name, fmt, config_dict, verbose = args
    import traceback

    import typer
    from pydantic import ValidationError

    try:
        typer.echo(f"\n=== Importing [{i+1}]: {name} ===")
        single_config = ImportDatasetConfig(
            source_path=src,
            source_format=fmt,
            dataset_name=name,
            root=config_dict["root"],
            split_name=config_dict["split_name"],
            enable_dvc=config_dict["enable_dvc"],
            processing_mode=config_dict["processing_mode"],
            track_with_dvc=config_dict["track_with_dvc"],
            bbox_tolerance=config_dict["bbox_tolerance"],
            ls_xml_config=config_dict["ls_xml_config"],
            ls_parse_config=config_dict["ls_parse_config"],
            roi_config=config_dict["roi_config"],
            disable_roi=config_dict["disable_roi"],
            transformations=config_dict["transformations"],
        )
        success = _import_dataset_core(single_config, verbose)
        return (i, name, success, None)
    except ValidationError as e:
        msg = f"[ERROR] Configuration validation error for '{name}':\n" + "\n".join(
            f"   {error['loc'][0]}: {error['msg']}" for error in e.errors()
        )
        return (i, name, False, msg)
    except Exception as e:
        msg = f"[ERROR] Unexpected error for '{name}': {str(traceback.format_exc())}"
        if verbose:
            msg += f"\n   Traceback: {traceback.format_exc()}"
        return (i, name, False, msg)
