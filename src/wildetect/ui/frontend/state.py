"""
State management for WildDetect Reflex frontend.
"""

import asyncio
import json
from typing import Dict, List, Optional

import httpx
import reflex as rx
from reflex.state import State

from .rxconfig import Config


class WildDetectState(State):
    """State management for WildDetect frontend."""

    # API configuration
    api_base_url: str = Config.api_url

    # File upload
    uploaded_files: List[str] = []
    upload_progress: int = 0
    upload_status: str = ""

    # Detection jobs
    detection_jobs: Dict[str, Dict] = {}
    current_detection_job: Optional[str] = None

    # Census campaigns
    census_jobs: Dict[str, Dict] = {}
    current_census_job: Optional[str] = None

    # System info
    system_info: Optional[Dict] = None

    # FiftyOne
    fiftyone_datasets: List[Dict] = []

    # UI state
    active_tab: str = "upload"
    loading: bool = False
    error_message: str = ""

    async def get_system_info(self):
        """Fetch system information from API."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_base_url}/info")
                if response.status_code == 200:
                    self.system_info = response.json()
                else:
                    self.error_message = (
                        f"Failed to get system info: {response.status_code}"
                    )
        except Exception as e:
            self.error_message = f"Error fetching system info: {str(e)}"

    async def upload_files(self, files: List[rx.UploadFile]):
        """Upload files to the API."""
        self.loading = True
        self.upload_status = "Uploading files..."
        self.upload_progress = 0

        try:
            async with httpx.AsyncClient() as client:
                # Prepare files for upload
                file_data = []
                for file in files:
                    # Read file content
                    content = await file.read()
                    file_data.append(
                        (
                            "files",
                            (
                                file.name,
                                content,
                                file.content_type or "application/octet-stream",
                            ),
                        )
                    )

                response = await client.post(
                    f"{self.api_base_url}/upload", files=file_data
                )

                if response.status_code == 200:
                    result = response.json()
                    self.uploaded_files = result.get("files", [])
                    self.upload_status = (
                        f"Successfully uploaded {len(self.uploaded_files)} files"
                    )
                    self.upload_progress = 100
                else:
                    self.error_message = f"Upload failed: {response.status_code}"
                    self.upload_progress = 0

        except Exception as e:
            self.error_message = f"Upload error: {str(e)}"
            self.upload_progress = 0
        finally:
            self.loading = False

    async def start_detection(self, detection_config: Dict):
        """Start a detection job."""
        self.loading = True
        self.error_message = ""

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base_url}/detect", json=detection_config
                )

                if response.status_code == 200:
                    result = response.json()
                    job_id = result["job_id"]
                    self.detection_jobs[job_id] = {
                        "status": "running",
                        "message": "Detection started",
                        "progress": 0,
                        "config": detection_config,
                    }
                    self.current_detection_job = job_id
                    self.active_tab = "jobs"
                else:
                    self.error_message = (
                        f"Failed to start detection: {response.status_code}"
                    )

        except Exception as e:
            self.error_message = f"Error starting detection: {str(e)}"
        finally:
            self.loading = False

    async def start_census(self, census_config: Dict):
        """Start a census campaign."""
        self.loading = True
        self.error_message = ""

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base_url}/census", json=census_config
                )

                if response.status_code == 200:
                    result = response.json()
                    job_id = result["job_id"]
                    self.census_jobs[job_id] = {
                        "status": "running",
                        "message": "Census campaign started",
                        "progress": 0,
                        "config": census_config,
                    }
                    self.current_census_job = job_id
                    self.active_tab = "census"
                else:
                    self.error_message = (
                        f"Failed to start census: {response.status_code}"
                    )

        except Exception as e:
            self.error_message = f"Error starting census: {str(e)}"
        finally:
            self.loading = False

    async def check_job_status(self, job_id: str, job_type: str = "detection"):
        """Check the status of a job."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_base_url}/jobs/{job_id}")

                if response.status_code == 200:
                    status_data = response.json()

                    if job_type == "detection":
                        if job_id in self.detection_jobs:
                            self.detection_jobs[job_id].update(status_data)
                    else:
                        if job_id in self.census_jobs:
                            self.census_jobs[job_id].update(status_data)

        except Exception as e:
            print(f"Error checking job status: {str(e)}")

    async def poll_jobs(self):
        """Poll all active jobs for status updates."""
        while True:
            # Check detection jobs
            for job_id in list(self.detection_jobs.keys()):
                if self.detection_jobs[job_id]["status"] in ["running", "pending"]:
                    await self.check_job_status(job_id, "detection")

            # Check census jobs
            for job_id in list(self.census_jobs.keys()):
                if self.census_jobs[job_id]["status"] in ["running", "pending"]:
                    await self.check_job_status(job_id, "census")

            await asyncio.sleep(5)  # Poll every 5 seconds

    async def launch_fiftyone(self):
        """Launch FiftyOne app."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_base_url}/fiftyone/launch")
                if response.status_code == 200:
                    self.error_message = "FiftyOne launched successfully"
                else:
                    self.error_message = (
                        f"Failed to launch FiftyOne: {response.status_code}"
                    )
        except Exception as e:
            self.error_message = f"Error launching FiftyOne: {str(e)}"

    async def get_fiftyone_datasets(self):
        """Get list of FiftyOne datasets."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_base_url}/fiftyone/datasets")
                if response.status_code == 200:
                    self.fiftyone_datasets = response.json()
        except Exception as e:
            print(f"Error fetching FiftyOne datasets: {str(e)}")
