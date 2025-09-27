import logging
import queue
import threading
from typing import Any, Dict, Optional, Union

import torch
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)


class SharedTensorBatch:
    """Shared memory batch container for torch.multiprocessing."""

    def __init__(self, batch_data: Dict[str, Any]):
        """Initialize shared tensor batch.

        Args:
            batch_data: Batch data containing images tensor and metadata
        """
        # Store tensor data in a pickleable format
        # Instead of share_memory_(), we'll store the raw tensor data
        tensor = batch_data["images"]
        self.tensor_data = (
            tensor.detach().cpu().clone()
        )  # Make a copy to avoid sharing issues
        self.tiles = batch_data["tiles"]
        self.batch_id = batch_data.get("batch_id")
        self.shape = tensor.shape
        self.dtype = tensor.dtype

    def to_tensor(self, device: str) -> torch.Tensor:
        """Convert shared tensor to target device.

        Args:
            device: Target device for the tensor

        Returns:
            Tensor on the target device
        """
        # Create tensor from stored data
        tensor = self.tensor_data.to(device)
        return tensor


class TorchProcessBatchQueue:
    """Process-safe queue for tensor batch data transfer using torch.multiprocessing."""

    def __init__(self, maxsize: int = 24):
        """Initialize the torch process batch queue.

        Args:
            maxsize: Maximum number of batches in the queue
        """
        self.queue = mp.Queue(maxsize=maxsize)
        # Use Manager for shared statistics
        self.manager = mp.Manager()
        self.stats = self.manager.dict({"put_count": 0, "get_count": 0, "errors": 0})
        self._lock = mp.Lock()

    def put_batch(
        self, batch: Union[SharedTensorBatch, Dict[str, Any]], timeout: float = 1.0
    ) -> bool:
        """Put a prepared batch into the queue.

        Args:
            batch: Batch data to put in queue
            timeout: Timeout for put operation

        Returns:
            True if batch was put successfully, False otherwise
        """
        try:
            self.queue.put(batch, timeout=timeout)
            with self._lock:
                self.stats["put_count"] += 1
            return True
        except queue.Full:
            with self._lock:
                self.stats["errors"] += 1
            logger.debug("Queue full, batch not added")
            return False
        except Exception as e:
            with self._lock:
                self.stats["errors"] += 1
            logger.error(f"Error putting batch in queue: {e}")
            return False

    def get_batch(
        self, timeout: float = 1.0
    ) -> Optional[Union[SharedTensorBatch, Dict[str, Any]]]:
        """Get a batch from the queue.

        Args:
            timeout: Timeout for get operation

        Returns:
            Batch data or None if timeout/empty
        """
        try:
            batch = self.queue.get(timeout=timeout)
            with self._lock:
                self.stats["get_count"] += 1
            return batch
        except queue.Empty:
            return None
        except Exception as e:
            with self._lock:
                self.stats["errors"] += 1
            logger.error(f"Error getting batch from queue: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics.

        Returns:
            Dictionary with queue statistics
        """
        with self._lock:
            stats = self.stats.copy()
            stats["queue_size"] = self.queue.qsize()
            stats["queue_maxsize"] = getattr(self.queue, "maxsize", 0)
        return stats

    def is_empty(self) -> bool:
        """Check if queue is empty.

        Returns:
            True if queue is empty
        """
        return self.queue.empty()

    def is_full(self) -> bool:
        """Check if queue is full.

        Returns:
            True if queue is full
        """
        return self.queue.full()


class BatchQueue:
    """Thread-safe queue for batch data transfer between threads."""

    def __init__(self, maxsize: int = 24):
        """Initialize the batch queue.

        Args:
            maxsize: Maximum number of batches in the queue
        """
        self.queue = queue.Queue(maxsize=maxsize)
        self.stats = {"put_count": 0, "get_count": 0, "errors": 0}
        self._lock = threading.Lock()

    def put_batch(self, batch: Dict[str, Any], timeout: float = 1.0) -> bool:
        """Put a prepared batch into the queue.

        Args:
            batch: Batch data to put in queue
            timeout: Timeout for put operation

        Returns:
            True if batch was put successfully, False otherwise
        """
        try:
            self.queue.put(batch, timeout=timeout)
            with self._lock:
                self.stats["put_count"] += 1
            return True
        except queue.Full:
            with self._lock:
                self.stats["errors"] += 1
            logger.debug("Queue full, batch not added")
            return False
        except Exception as e:
            with self._lock:
                self.stats["errors"] += 1
            logger.error(f"Error putting batch in queue: {e}")
            return False

    def get_batch(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get a batch from the queue.

        Args:
            timeout: Timeout for get operation

        Returns:
            Batch data or None if timeout/empty
        """
        try:
            batch = self.queue.get(timeout=timeout)
            with self._lock:
                self.stats["get_count"] += 1
            return batch
        except queue.Empty:
            return None
        except Exception as e:
            with self._lock:
                self.stats["errors"] += 1
            logger.error(f"Error getting batch from queue: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics.

        Returns:
            Dictionary with queue statistics
        """
        with self._lock:
            stats = self.stats.copy()
            stats["queue_size"] = self.queue.qsize()
            stats["queue_maxsize"] = getattr(self.queue, "maxsize", 0)
        return stats

    def is_empty(self) -> bool:
        """Check if queue is empty.

        Returns:
            True if queue is empty
        """
        return self.queue.empty()

    def is_full(self) -> bool:
        """Check if queue is full.

        Returns:
            True if queue is full
        """
        return self.queue.full()
