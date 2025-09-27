"""
Detection Pipeline for end-to-end wildlife detection processing.
"""

import asyncio
import logging
from functools import wraps
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ..config import LoaderConfig, PredictionConfig
from ..data import Detection, DroneImage
from ..data.loader import DataLoader
from .base import DetectionPipeline

logger = logging.getLogger(__name__)


class AsyncDetectionPipeline(DetectionPipeline):
    """Asynchronous detection pipeline optimized for inference server usage."""

    def __init__(
        self,
        config: PredictionConfig,
        loader_config: LoaderConfig,
    ):
        """Initialize the async detection pipeline.

        Args:
            config: Prediction configuration
            loader_config: Data loader configuration
            max_concurrent: Maximum number of concurrent requests to inference server
        """
        super().__init__(config, loader_config)
        self.max_concurrent = self.config.max_concurrent

        # Validate that either inference service URL is configured or local inference is available
        if not self.config.inference_service_url and not hasattr(
            self, "detection_system"
        ):
            raise ValueError(
                "AsyncDetectionPipeline requires either inference_service_url to be configured "
                "or local detection_system to be available"
            )
        self.detection_system = self.awaitify(self.detection_system)
        logger.info(
            f"Initialized AsyncDetectionPipeline with max_concurrent={self.config.max_concurrent}"
        )

    def awaitify(self, sync_func):
        """Wrap a synchronous callable to allow ``await``'ing it"""

        @wraps(sync_func)
        async def async_func(*args, **kwargs):
            return sync_func(*args, **kwargs)

        return async_func

    async def run_detection_async(
        self,
        image_paths: Optional[List[str]] = None,
        image_dir: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> List[DroneImage]:
        """Run detection asynchronously with optimized concurrent processing."""
        logger.info("Starting async detection pipeline")

        # Initialize data loader
        if image_paths:
            self.data_loader = DataLoader(
                image_paths=image_paths,
                config=self.loader_config,
            )
        elif image_dir:
            self.data_loader = DataLoader(
                image_dir=image_dir,
                config=self.loader_config,
            )
        else:
            raise ValueError("Either image_paths or image_dir must be provided")

        # check if inference service is available
        if self.config.inference_service_url is None:
            raise ValueError(
                f"Inference service at {self.config.inference_service_url} is not available"
            )

        # Load all batches
        all_batches = list(self.data_loader)
        total_batches = len(all_batches)
        logger.info(f"Total batches to process: {total_batches}")

        if total_batches == 0:
            logger.warning("No batches to process")
            return []

        # Process batches concurrently
        progress_bar = tqdm(
            total=total_batches, desc="Processing batches", unit="batch"
        )

        try:
            processed_batches = await self._process_batches_concurrent(
                all_batches, self.max_concurrent, progress_bar
            )
        finally:
            progress_bar.close()

        logger.info(
            f"Completed processing {len(processed_batches)} batches with {self.error_count} errors"
        )

        # Post-processing
        all_drone_images = self._postprocess(batches=processed_batches)
        if len(all_drone_images) == 0:
            logger.warning("No batches were processed")
            return []

        # Save results if path provided
        if save_path:
            self._save_results(all_drone_images, save_path)

        return all_drone_images

    async def _process_batch_async(
        self,
        batch: Dict[str, Any],
        progress_bar: Optional[tqdm] = None,
    ) -> List[List[Detection]]:
        """Async version of _process_batch."""
        if self.detection_system is None:
            raise ValueError("Detection system not initialized")

        detections = await self.detection_system(batch.pop("images"))

        if progress_bar:
            progress_bar.update(1)

        detections = self._convert_to_detection(detections)

        return detections

    async def _process_batches_concurrent(
        self,
        batches: List[Dict[str, Any]],
        max_concurrent: int = 10,
        progress_bar: Optional[tqdm] = None,
    ) -> List[Dict[str, Any]]:
        """Process multiple batches concurrently using semaphore to limit concurrency."""

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_batch_with_semaphore(batch: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    detections = await self._process_batch_async(batch, progress_bar)
                    batch["detections"] = detections
                    return batch
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    self.error_count += 1
                    batch["detections"] = []
                    return batch

        # Create tasks for all batches
        tasks = [process_batch_with_semaphore(batch) for batch in batches]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results: List[Dict[str, Any]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing batch {i}: {result}")
                self.error_count += 1
                processed_results.append(
                    {"detections": [], "tiles": batches[i]["tiles"]}
                )
            else:
                processed_results.append(result)  # type: ignore

        return processed_results

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the async pipeline."""
        info = super().get_pipeline_info()
        info.update(
            {
                "pipeline_type": "async",
                "max_concurrent": self.max_concurrent,
            }
        )
        return info
