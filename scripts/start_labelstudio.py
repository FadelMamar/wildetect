#!/usr/bin/env python3
"""
Start LabelStudio server for WildDetect.

This script starts a LabelStudio server for annotation management
and integration with the wildlife detection workflow.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def start_labelstudio():
    """Start LabelStudio server."""
    try:
        logger.info("Starting LabelStudio server...")
        
        # Create data directory
        data_dir = Path("data/labelstudio")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Start LabelStudio server
        cmd = [
            sys.executable, "-m", "label_studio",
            "start",
            "--host", "localhost",
            "--port", "8080",
            "--data-dir", str(data_dir),
            "--no-browser"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error starting LabelStudio: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("LabelStudio server stopped by user")
        return True
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def install_labelstudio():
    """Install LabelStudio if not already installed."""
    try:
        logger.info("Checking LabelStudio installation...")
        
        # Try to import label_studio
        try:
            import label_studio
            logger.info("✓ LabelStudio is already installed")
            return True
        except ImportError:
            logger.info("Installing LabelStudio...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "label-studio"
            ])
            logger.info("✓ LabelStudio installed successfully")
            return True
            
    except Exception as e:
        logger.error(f"Error installing LabelStudio: {e}")
        return False


def main():
    """Main function."""
    logger.info("🚀 Starting LabelStudio for WildDetect...")
    
    # Check if LabelStudio is installed
    if not install_labelstudio():
        logger.error("Failed to install LabelStudio")
        sys.exit(1)
    
    # Start LabelStudio server
    if not start_labelstudio():
        logger.error("Failed to start LabelStudio server")
        sys.exit(1)
    
    logger.info("LabelStudio server started successfully!")
    logger.info("Access LabelStudio at: http://localhost:8080")


if __name__ == "__main__":
    main() 