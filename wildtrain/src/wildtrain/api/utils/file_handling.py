"""File handling utilities for the WildTrain API."""

import aiofiles
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import json

from ..dependencies import get_settings, validate_file_path
from .error_handling import FileNotFoundAPIError

logger = logging.getLogger(__name__)


class FileManager:
    """Manages file uploads, downloads, and storage."""
    
    def __init__(self):
        self.settings = get_settings()
    
    async def save_uploaded_file(
        self,
        file_content: bytes,
        filename: str,
        subdirectory: Optional[str] = None
    ) -> Path:
        """Save an uploaded file to the upload directory."""
        
        # Create subdirectory if specified
        upload_path = self.settings.upload_dir
        if subdirectory:
            upload_path = upload_path / subdirectory
            upload_path.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename if needed
        file_path = upload_path / filename
        if file_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name, ext = file_path.stem, file_path.suffix
            filename = f"{name}_{timestamp}{ext}"
            file_path = upload_path / filename
        
        # Save the file
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)
        
        logger.info(f"Saved uploaded file: {file_path}")
        return file_path
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get information about a file."""
        if not file_path.exists():
            raise FileNotFoundAPIError(str(file_path))
        
        stat = file_path.stat()
        return {
            "path": str(file_path),
            "name": file_path.name,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "is_file": file_path.is_file(),
            "is_directory": file_path.is_dir()
        }
    
    def list_directory(self, directory_path: Path) -> List[Dict[str, Any]]:
        """List contents of a directory."""
        if not directory_path.exists():
            raise FileNotFoundAPIError(str(directory_path))
        
        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        items = []
        for item in directory_path.iterdir():
            try:
                items.append(self.get_file_info(item))
            except Exception as e:
                logger.warning(f"Could not get info for {item}: {e}")
                continue
        
        return sorted(items, key=lambda x: x["name"])
    
    def create_results_directory(self, job_id: str, task_type: str) -> Path:
        """Create a results directory for a job."""
        results_dir = self.settings.results_dir / task_type / job_id
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir
    
    def save_job_result(
        self,
        job_id: str,
        task_type: str,
        result_data: Dict[str, Any],
        filename: str = "result.json"
    ) -> Path:
        """Save job result to a file."""
        results_dir = self.create_results_directory(job_id, task_type)
        result_file = results_dir / filename
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        
        logger.info(f"Saved job result: {result_file}")
        return result_file
    
    def get_job_results(self, job_id: str, task_type: str) -> List[Dict[str, Any]]:
        """Get all result files for a job."""
        results_dir = self.settings.results_dir / task_type / job_id
        if not results_dir.exists():
            return []
        
        return self.list_directory(results_dir)
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> None:
        """Clean up old uploaded files and results."""
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)
        
        # Clean up uploads
        for file_path in self.settings.upload_dir.rglob("*"):
            if file_path.is_file():
                try:
                    if file_path.stat().st_mtime < cutoff:
                        file_path.unlink()
                        logger.info(f"Cleaned up old file: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not clean up {file_path}: {e}")
        
        # Clean up old results
        for file_path in self.settings.results_dir.rglob("*"):
            if file_path.is_file():
                try:
                    if file_path.stat().st_mtime < cutoff:
                        file_path.unlink()
                        logger.info(f"Cleaned up old result: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not clean up {file_path}: {e}")


# Global file manager instance
file_manager = FileManager()


async def save_config_file(file_content: bytes, filename: str) -> Path:
    """Save a configuration file."""
    return await file_manager.save_uploaded_file(
        file_content, 
        filename, 
        subdirectory="configs"
    )


def get_result_file_path(job_id: str, task_type: str, filename: str) -> Path:
    """Get the path to a result file."""
    results_dir = file_manager.settings.results_dir / task_type / job_id
    return results_dir / filename


def validate_config_file(file_path: Path) -> Path:
    """Validate a configuration file path."""
    return validate_file_path(file_path, must_exist=True)


def get_file_download_url(file_path: Path) -> str:
    """Generate a download URL for a file."""
    # This would be implemented based on your file serving strategy
    # For now, return a relative path
    return f"/files/{file_path.relative_to(file_manager.settings.results_dir)}"
