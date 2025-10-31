"""
ROI-related logic for CLI commands.
"""

import traceback
from pathlib import Path

import typer
from pydantic import ValidationError

from ..config import ENV_FILE, ROIConfig
from ..logging_config import get_logger
from ..pipeline import FrameworkDataManager, Loader, PathManager
from .models import ROIDatasetConfig

logger = get_logger(__name__)


def create_roi_dataset_core(config: ROIDatasetConfig, verbose: bool = False) -> bool:
    """Core logic for creating ROI datasets."""
    try:
        if verbose:
            typer.echo(f"[INFO] Creating dataset...")
            typer.echo(f"   Source: {config.source_path}")
            typer.echo(f"   Format: {config.source_format}")
            typer.echo(f"   Name: {config.dataset_name}")
            typer.echo(f"   Split: {config.split_name}")
            typer.echo(f"   Output root: {config.root}")
            typer.echo(f"   ROI config: {config.roi_config}")
            typer.echo(f"   LS XML config: {config.ls_xml_config}")
            typer.echo(f"   LS parse config: {config.ls_parse_config}")

        logger.info(f"Config: {config}")

        loader = Loader()
        dataset_info, split_coco_data = loader.load(
            config.source_path,
            config.source_format,
            config.dataset_name,
            config.bbox_tolerance,
            config.split_name,
            dotenv_path=ENV_FILE,
            ls_xml_config=config.ls_xml_config,
            ls_parse_config=config.ls_parse_config,
        )

        path_manager = PathManager(Path(config.root))
        framework_data_manager = FrameworkDataManager(path_manager)

        framework_data_manager.create_roi_format(
            dataset_name=config.dataset_name,
            coco_data=split_coco_data[config.split_name],
            split=config.split_name,
            roi_config=config.roi_config,
            draw_original_bboxes=config.draw_original_bboxes,
        )
        typer.echo(
            f"[SUCCESS] Successfully created ROI dataset for '{config.dataset_name}' (split: {config.split_name}) at {config.root}"
        )
        return True
    except ValidationError as e:
        typer.echo(f"[ERROR] Configuration validation error:")
        for error in e.errors():
            typer.echo(f"   {error['loc'][0]}: {error['msg']}")
        return False
    except Exception as e:
        typer.echo(f"[ERROR] Failed to create ROI dataset: {str(e)}")
        if verbose:
            typer.echo(f"   Traceback: {traceback.format_exc()}")
        return False


def create_roi_one_worker(args) -> tuple:
    """Top-level worker for ProcessPoolExecutor in bulk_create_roi_datasets."""
    i, src, name, config_dict, verbose = args
    import traceback

    import typer
    from pydantic import ValidationError

    try:
        typer.echo(f"\n=== Creating ROI [{i+1}]: {name} ===")
        from pathlib import Path

        from .models import ROIDatasetConfig

        single_config = ROIDatasetConfig(
            source_path=src,
            source_format=config_dict["source_format"],
            dataset_name=name,
            root=config_dict["root"],
            split_name=config_dict["split_name"],
            bbox_tolerance=config_dict["bbox_tolerance"],
            roi_config=config_dict["roi_config"],
            ls_xml_config=config_dict["ls_xml_config"],
            ls_parse_config=config_dict["ls_parse_config"],
            draw_original_bboxes=config_dict.get("draw_original_bboxes", False),
        )
        # Use the same core logic as create_roi_dataset
        from ..pipeline import FrameworkDataManager, Loader, PathManager

        loader = Loader()
        dataset_info, split_coco_data = loader.load(
            single_config.source_path,
            single_config.source_format,
            single_config.dataset_name,
            single_config.bbox_tolerance,
            single_config.split_name,
            dotenv_path=ENV_FILE,
            ls_xml_config=single_config.ls_xml_config,
            ls_parse_config=single_config.ls_parse_config,
        )
        path_manager = PathManager(Path(single_config.root))
        framework_data_manager = FrameworkDataManager(path_manager)
        framework_data_manager.create_roi_format(
            dataset_name=single_config.dataset_name,
            coco_data=split_coco_data[single_config.split_name],
            split=single_config.split_name,
            roi_config=single_config.roi_config,
        )
        return (i, name, True, None)
    except ValidationError as e:
        msg = f"[ERROR] Configuration validation error for '{name}':\n" + "\n".join(
            f"   {error['loc'][0]}: {error['msg']}" for error in e.errors()
        )
        return (i, name, False, msg)
    except Exception as e:
        msg = f"[ERROR] Unexpected error for '{name}': {str(e)}"
        if verbose:
            msg += f"\n   Traceback: {traceback.format_exc()}"
        return (i, name, False, msg)
