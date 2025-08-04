"""
Reflex configuration for WildDetect frontend.
"""

import reflex as rx


class Config(rx.Config):
    """Reflex configuration."""

    # App name
    app_name = "WildDetect"

    # App title
    title = "WildDetect - Wildlife Detection System"

    # App description
    description = "A modern web interface for wildlife detection and analysis"

    # API URL
    api_url = "http://localhost:8800"

    # Frontend URL
    frontend_url = "http://localhost:3000"

    # Database URL (if needed)
    database_url = "sqlite:///wildetect.db"

    # Environment
    env = rx.Env.DEV

    # Backend port
    backend_port = 8800

    # Frontend port
    frontend_port = 3000

    # CORS settings
    cors_allowed_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # Logging
    log_level = "INFO"

    # Debug mode
    debug = True
