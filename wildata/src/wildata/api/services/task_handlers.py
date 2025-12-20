"""
Background task handlers for API operations.
"""

import asyncio
from typing import Any, Dict, Optional

from ..exceptions import JobError
from ..models.jobs import JobResult, JobStatus
from ..services.job_queue import get_job_queue


async def handle_import_dataset(
    job_id: str, config: Any, verbose: bool = False
) -> JobResult:
    """Handle dataset import in background."""
    job_queue = get_job_queue()

    try:
        # Update job status to running
        await job_queue.update_job_status(job_id, JobStatus.RUNNING, progress=10.0)

        # Import the CLI core logic
        from ...cli.import_logic import _import_dataset_core

        # Config is already a CLI config model (inherits from ImportDatasetConfig)
        cli_config = config

        # Run the import
        success = _import_dataset_core(cli_config, verbose)

        await job_queue.update_job_status(job_id, JobStatus.RUNNING, progress=90.0)

        if success:
            result = JobResult(
                success=True,
                message=f"Successfully imported dataset '{cli_config.dataset_name}'",
                data={
                    "dataset_name": cli_config.dataset_name,
                    "source_path": cli_config.source_path,
                    "source_format": cli_config.source_format,
                },
            )
            await job_queue.update_job_status(
                job_id, JobStatus.COMPLETED, progress=100.0, result=result
            )
        else:
            result = JobResult(success=False, error="Dataset import failed")
            await job_queue.update_job_status(
                job_id, JobStatus.FAILED, progress=100.0, result=result
            )

        return result

    except Exception as e:
        result = JobResult(success=False, error=f"Import failed: {str(e)}")
        await job_queue.update_job_status(
            job_id, JobStatus.FAILED, progress=100.0, result=result
        )
        return result


async def handle_bulk_import(
    job_id: str, config: Any, verbose: bool = False
) -> JobResult:
    """Handle bulk dataset import in background."""
    job_queue = get_job_queue()

    try:
        await job_queue.update_job_status(job_id, JobStatus.RUNNING, progress=5.0)

        # Import the CLI core logic
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from ...cli.import_logic import import_one_worker

        # Config is already a CLI config model (inherits from BulkImportDatasetConfig)
        cli_config = config

        # Get files to process
        import os
        from pathlib import Path

        from ...cli.utils import create_dataset_name

        files = []
        for source_path in cli_config.source_paths:
            if not os.path.isdir(source_path):
                raise ValueError(f"source_path must be a directory: {source_path}")
            files.extend(
                [
                    f.resolve().as_posix()
                    for f in Path(source_path).iterdir()
                    if f.is_file() and not f.name.startswith(".")
                ]
            )

        if not files:
            raise ValueError(
                f"No files found in directories: {cli_config.source_paths}"
            )

        dataset_names = [create_dataset_name(f) for f in files]
        formats = [cli_config.source_format] * len(files)

        # Convert config to dict for pickling
        config_dict = cli_config.model_dump()
        args_list = [
            (i, src, name, fmt, config_dict, verbose)
            for i, (src, name, fmt) in enumerate(zip(files, dataset_names, formats))
        ]

        total_files = len(files)
        successful_imports = 0

        await job_queue.update_job_status(job_id, JobStatus.RUNNING, progress=10.0)

        # Process files with multiprocessing
        with ProcessPoolExecutor(max_workers=2) as executor:
            future_to_idx = {
                executor.submit(import_one_worker, args): i
                for i, args in enumerate(args_list)
            }

            for future in as_completed(future_to_idx):
                i, name, success, msg = future.result()
                if success:
                    successful_imports += 1

                # Update progress
                progress = 10.0 + (i + 1) / total_files * 80.0
                await job_queue.update_job_status(
                    job_id, JobStatus.RUNNING, progress=progress
                )

        result = JobResult(
            success=successful_imports > 0,
            message=f"Bulk import complete. {successful_imports}/{total_files} succeeded.",
            data={
                "total_files": total_files,
                "successful_imports": successful_imports,
                "failed_imports": total_files - successful_imports,
            },
        )

        await job_queue.update_job_status(
            job_id, JobStatus.COMPLETED, progress=100.0, result=result
        )
        return result

    except Exception as e:
        result = JobResult(success=False, error=f"Bulk import failed: {str(e)}")
        await job_queue.update_job_status(
            job_id, JobStatus.FAILED, progress=100.0, result=result
        )
        return result


async def handle_create_roi(
    job_id: str, config: Any, verbose: bool = False
) -> JobResult:
    """Handle ROI dataset creation in background."""
    job_queue = get_job_queue()

    try:
        await job_queue.update_job_status(job_id, JobStatus.RUNNING, progress=10.0)

        # Import the CLI core logic
        from ...cli.roi_logic import create_roi_dataset_core

        # Config is already a CLI config model (inherits from ROIDatasetConfig)
        cli_config = config

        await job_queue.update_job_status(job_id, JobStatus.RUNNING, progress=50.0)

        # Run the ROI creation
        success = create_roi_dataset_core(cli_config, verbose)

        await job_queue.update_job_status(job_id, JobStatus.RUNNING, progress=90.0)

        if success:
            result = JobResult(
                success=True,
                message=f"Successfully created ROI dataset '{cli_config.dataset_name}'",
                data={
                    "dataset_name": cli_config.dataset_name,
                    "source_path": cli_config.source_path,
                    "source_format": cli_config.source_format,
                },
            )
            await job_queue.update_job_status(
                job_id, JobStatus.COMPLETED, progress=100.0, result=result
            )
        else:
            result = JobResult(success=False, error="ROI dataset creation failed")
            await job_queue.update_job_status(
                job_id, JobStatus.FAILED, progress=100.0, result=result
            )

        return result

    except Exception as e:
        result = JobResult(success=False, error=f"ROI creation failed: {str(e)}")
        await job_queue.update_job_status(
            job_id, JobStatus.FAILED, progress=100.0, result=result
        )
        return result


async def handle_bulk_create_roi(
    job_id: str, config: Any, verbose: bool = False
) -> JobResult:
    """Handle bulk ROI dataset creation in background."""
    job_queue = get_job_queue()

    try:
        await job_queue.update_job_status(job_id, JobStatus.RUNNING, progress=5.0)

        # Import the CLI core logic
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from ...cli.roi_logic import create_roi_one_worker

        # Config is already a CLI config model (inherits from BulkCreateROIDatasetConfig)
        cli_config = config

        # Get files to process
        import os
        from pathlib import Path

        from ...cli.utils import create_dataset_name

        files = []
        for source_path in cli_config.source_paths:
            if not os.path.isdir(source_path):
                raise ValueError(f"source_path must be a directory: {source_path}")
            files.extend(
                [
                    f.resolve().as_posix()
                    for f in Path(source_path).iterdir()
                    if f.is_file() and not f.name.startswith(".")
                ]
            )

        if not files:
            raise ValueError(
                f"No files found in directories: {cli_config.source_paths}"
            )

        dataset_names = [create_dataset_name(f) for f in files]

        config_dict = cli_config.model_dump()
        args_list = [
            (i, src, name, config_dict, verbose)
            for i, (src, name) in enumerate(zip(files, dataset_names))
        ]

        total_files = len(files)
        successful_creations = 0

        await job_queue.update_job_status(job_id, JobStatus.RUNNING, progress=10.0)

        # Process files with multiprocessing
        with ProcessPoolExecutor(max_workers=2) as executor:
            future_to_idx = {
                executor.submit(create_roi_one_worker, args): i
                for i, args in enumerate(args_list)
            }

            for future in as_completed(future_to_idx):
                i, name, success, msg = future.result()
                if success:
                    successful_creations += 1

                # Update progress
                progress = 10.0 + (i + 1) / total_files * 80.0
                await job_queue.update_job_status(
                    job_id, JobStatus.RUNNING, progress=progress
                )

        result = JobResult(
            success=successful_creations > 0,
            message=f"Bulk ROI creation complete. {successful_creations}/{total_files} succeeded.",
            data={
                "total_files": total_files,
                "successful_creations": successful_creations,
                "failed_creations": total_files - successful_creations,
            },
        )

        await job_queue.update_job_status(
            job_id, JobStatus.COMPLETED, progress=100.0, result=result
        )
        return result

    except Exception as e:
        result = JobResult(success=False, error=f"Bulk ROI creation failed: {str(e)}")
        await job_queue.update_job_status(
            job_id, JobStatus.FAILED, progress=100.0, result=result
        )
        return result


async def handle_update_gps(
    job_id: str, config: Any, verbose: bool = False
) -> JobResult:
    """Handle GPS update in background."""
    job_queue = get_job_queue()

    try:
        await job_queue.update_job_status(job_id, JobStatus.RUNNING, progress=10.0)

        # Import the GPS manager
        from ...adapters.utils import ExifGPSManager

        # Config is already a CLI config model (inherits from ExifGPSUpdateConfig)
        cli_config = config

        await job_queue.update_job_status(job_id, JobStatus.RUNNING, progress=50.0)

        # Run the GPS update
        gps_manager = ExifGPSManager()
        gps_manager.update_folder_from_csv(
            image_folder=cli_config.image_folder,
            csv_path=cli_config.csv_path,
            output_dir=cli_config.output_dir,
            skip_rows=cli_config.skip_rows,
            filename_col=cli_config.filename_col,
            lat_col=cli_config.lat_col,
            lon_col=cli_config.lon_col,
            alt_col=cli_config.alt_col,
        )

        await job_queue.update_job_status(job_id, JobStatus.RUNNING, progress=90.0)

        result = JobResult(
            success=True,
            message=f"Successfully updated GPS data for images in '{cli_config.image_folder}'",
            data={
                "image_folder": cli_config.image_folder,
                "csv_path": cli_config.csv_path,
                "output_dir": cli_config.output_dir,
            },
        )
        await job_queue.update_job_status(
            job_id, JobStatus.COMPLETED, progress=100.0, result=result
        )
        return result

    except Exception as e:
        result = JobResult(success=False, error=f"GPS update failed: {str(e)}")
        await job_queue.update_job_status(
            job_id, JobStatus.FAILED, progress=100.0, result=result
        )
        return result


async def handle_visualize_classification(
    job_id: str, config: Any, verbose: bool = False
) -> JobResult:
    """Handle classification dataset visualization in background."""
    job_queue = get_job_queue()

    try:
        await job_queue.update_job_status(job_id, JobStatus.RUNNING, progress=10.0)

        # Import the visualization module
        from ...visualization import visualize_classification_dataset

        # Config is already a CLI config model
        cli_config = config

        await job_queue.update_job_status(job_id, JobStatus.RUNNING, progress=50.0)

        # Run the visualization
        result_message = visualize_classification_dataset(
            root_data_directory=cli_config.root_data_directory,
            dataset_name=cli_config.dataset_name,
            split=cli_config.split,
            load_as_single_class=cli_config.load_as_single_class,
            background_class_name=cli_config.background_class_name,
            single_class_name=cli_config.single_class_name,
            keep_classes=cli_config.keep_classes,
            discard_classes=cli_config.discard_classes,
        )

        await job_queue.update_job_status(job_id, JobStatus.RUNNING, progress=90.0)

        result = JobResult(
            success=True,
            message=f"Successfully visualized classification dataset: {result_message}",
            data={
                "dataset_name": cli_config.dataset_name,
                "split": cli_config.split,
                "visualization_url": result_message,
            },
        )
        await job_queue.update_job_status(
            job_id, JobStatus.COMPLETED, progress=100.0, result=result
        )
        return result

    except Exception as e:
        result = JobResult(success=False, error=f"Visualization failed: {str(e)}")
        await job_queue.update_job_status(
            job_id, JobStatus.FAILED, progress=100.0, result=result
        )
        return result


async def handle_visualize_detection(
    job_id: str, config: Any, verbose: bool = False
) -> JobResult:
    """Handle detection dataset visualization in background."""
    job_queue = get_job_queue()

    try:
        await job_queue.update_job_status(job_id, JobStatus.RUNNING, progress=10.0)

        # Import the visualization module
        from ...visualization import visualize_detection_dataset

        # Config is already a CLI config model
        cli_config = config

        await job_queue.update_job_status(job_id, JobStatus.RUNNING, progress=50.0)

        # Run the visualization
        result_message = visualize_detection_dataset(
            root_data_directory=cli_config.root_data_directory,
            dataset_name=cli_config.dataset_name,
            split=cli_config.split,
            load_as_single_class=cli_config.load_as_single_class,
            background_class_name=cli_config.background_class_name,
            single_class_name=cli_config.single_class_name,
            keep_classes=cli_config.keep_classes,
            discard_classes=cli_config.discard_classes,
        )

        await job_queue.update_job_status(job_id, JobStatus.RUNNING, progress=90.0)

        result = JobResult(
            success=True,
            message=f"Successfully visualized detection dataset: {result_message}",
            data={
                "dataset_name": cli_config.dataset_name,
                "split": cli_config.split,
                "visualization_url": result_message,
            },
        )
        await job_queue.update_job_status(
            job_id, JobStatus.COMPLETED, progress=100.0, result=result
        )
        return result

    except Exception as e:
        result = JobResult(success=False, error=f"Visualization failed: {str(e)}")
        await job_queue.update_job_status(
            job_id, JobStatus.FAILED, progress=100.0, result=result
        )
        return result


async def handle_delete_dataset(
    job_id: str, dataset_name: str, root: str = "data"
) -> JobResult:
    """Handle dataset deletion in background."""
    job_queue = get_job_queue()

    try:
        await job_queue.update_job_status(job_id, JobStatus.RUNNING, progress=10.0)

        # Import the pipeline
        from ...pipeline import DataPipeline

        await job_queue.update_job_status(job_id, JobStatus.RUNNING, progress=50.0)

        # Run the deletion
        pipeline = DataPipeline(root=root, split_name="train")
        success = pipeline.delete_dataset(dataset_name)

        await job_queue.update_job_status(job_id, JobStatus.RUNNING, progress=90.0)

        if success:
            result = JobResult(
                success=True,
                message=f"Successfully deleted dataset '{dataset_name}'",
                data={
                    "dataset_name": dataset_name,
                    "root": root,
                },
            )
            await job_queue.update_job_status(
                job_id, JobStatus.COMPLETED, progress=100.0, result=result
            )
        else:
            result = JobResult(
                success=False, error=f"Failed to delete dataset '{dataset_name}'"
            )
            await job_queue.update_job_status(
                job_id, JobStatus.FAILED, progress=100.0, result=result
            )

        return result

    except Exception as e:
        result = JobResult(success=False, error=f"Deletion failed: {str(e)}")
        await job_queue.update_job_status(
            job_id, JobStatus.FAILED, progress=100.0, result=result
        )
        return result
