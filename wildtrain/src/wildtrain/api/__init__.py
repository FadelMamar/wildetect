"""WildTrain FastAPI REST API package."""

from .main import create_app, fastapi_app

__version__ = "0.1.0"
__all__ = ["fastapi_app", "create_app"]
