"""
API routers package.
"""

from .datasets import router as datasets_router
from .gps import router as gps_router
from .health import router as health_router
from .jobs import router as jobs_router
from .roi import router as roi_router
from .visualization import router as visualization_router

__all__ = [
    "datasets_router",
    "jobs_router",
    "health_router",
    "roi_router",
    "gps_router",
    "visualization_router",
]
