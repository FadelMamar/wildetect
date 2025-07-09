#!/usr/bin/env python3
"""
CUDA installer utility for WildDetect.
Automatically detects CUDA availability and installs appropriate PyTorch version.
"""

import subprocess
import sys
import platform
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


def setup_logging():
    """Setup logging to both console and file."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"cuda_installer_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def run_subprocess_with_logging(cmd, description, capture_output=True):
    """Run subprocess command with logging."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info(f"Description: {description}")
    
    try:
        if capture_output:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Command succeeded. stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"stderr: {result.stderr}")
            return result
        else:
            result = subprocess.run(cmd, check=True)
            logger.info("Command succeeded (no output captured)")
            return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with return code {e.returncode}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error running command: {e}")
        raise


def detect_cuda_version() -> Optional[str]:
    """Detect CUDA version on the system."""
    logger = logging.getLogger(__name__)
    
    try:
        # Try to get CUDA version using nvidia-smi
        result = run_subprocess_with_logging(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            "Detecting CUDA driver version"
        )
        driver_version = result.stdout.strip().split('\n')[0]
        logger.info(f"Detected driver version: {driver_version}")
        
        # Map driver version to CUDA version
        # This is a simplified mapping - you might want to expand this
        driver_major = int(driver_version.split('.')[0])
        
        if driver_major >= 525:
            cuda_version = "121"
        elif driver_major >= 520:
            cuda_version = "118"
        elif driver_major >= 470:
            cuda_version = "118"
        else:
            cuda_version = "118"  # Default to CUDA 11.8 for older drivers
        
        logger.info(f"Mapped driver version {driver_major} to CUDA version {cuda_version}")
        return cuda_version
            
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to detect CUDA version: {e}")
        return None


def check_cuda_availability() -> bool:
    """Check if CUDA is available on the system."""
    logger = logging.getLogger(__name__)
    
    try:
        run_subprocess_with_logging(
            ["nvidia-smi"],
            "Checking CUDA availability"
        )
        logger.info("CUDA is available on this system")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"CUDA not available: {e}")
        return False


def install_cuda_torch(cuda_version: Optional[str] = None) -> None:
    """
    Install PyTorch with CUDA support.
    
    Args:
        cuda_version: Specific CUDA version to install (118, 121, etc.)
                     If None, will auto-detect or use default.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting CUDA PyTorch installation")
    
    print("🔍 Detecting CUDA availability...")
    
    if not check_cuda_availability():
        print("❌ CUDA not available on this system.")
        print("   Installing CPU-only PyTorch...")
        install_cpu_torch()
        return
    
    if cuda_version is None:
        detected_version = detect_cuda_version()
        if detected_version:
            cuda_version = detected_version
            print(f"✅ Detected CUDA version: {cuda_version}")
        else:
            cuda_version = "118"  # Default to CUDA 11.8
            print(f"⚠️  Could not detect CUDA version, using default: {cuda_version}")
    
    print(f"🚀 Installing PyTorch with CUDA {cuda_version} support...")
    logger.info(f"Installing PyTorch with CUDA {cuda_version}")
    
    try:
        # Uninstall existing torch packages
        logger.info("Uninstalling existing torch packages")
        run_subprocess_with_logging([
            "uv", "pip", "uninstall", "torch", "torchvision"
        ], "Uninstalling existing torch packages", capture_output=False)
        
        # Install CUDA-compatible PyTorch using the correct index URL
        if cuda_version == "121":
            logger.info("Installing PyTorch with CUDA 12.1")
            run_subprocess_with_logging([
                "uv", "pip", "install","--force-reinstall",
                "torch==2.6.0", "torchvision",
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ], "Installing PyTorch with CUDA 12.1")
        elif cuda_version == "118":
            logger.info("Installing PyTorch with CUDA 11.8")
            run_subprocess_with_logging([
                "uv", "pip", "install","--force-reinstall",
                "torch==2.6.0", "torchvision",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ], "Installing PyTorch with CUDA 11.8")
        else:
            # Fallback to CPU-only if unknown version
            logger.warning(f"Unknown CUDA version '{cuda_version}', falling back to CPU-only")
            print(f"⚠️  Unknown CUDA version '{cuda_version}', installing CPU-only PyTorch.")
            install_cpu_torch()
            return
        
        print("✅ PyTorch with CUDA support installed successfully!")
        logger.info("CUDA PyTorch installation completed successfully")
        verify_cuda_installation()
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install CUDA PyTorch: {e}")
        print(f"❌ Failed to install CUDA PyTorch: {e}")
        print("   Falling back to CPU-only installation...")
        install_cpu_torch()


def install_cpu_torch() -> None:
    """Install CPU-only PyTorch."""
    logger = logging.getLogger(__name__)
    logger.info("Installing CPU-only PyTorch")
    
    try:
        run_subprocess_with_logging([
            "uv", "pip", "install", 
            "torch", "torchvision"
        ], "Installing CPU-only PyTorch")
        print("✅ CPU-only PyTorch installed successfully!")
        logger.info("CPU-only PyTorch installation completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install CPU PyTorch: {e}")
        print(f"❌ Failed to install CPU PyTorch: {e}")


def verify_cuda_installation() -> None:
    """Verify that CUDA PyTorch is working correctly."""
    logger = logging.getLogger(__name__)
    logger.info("Verifying CUDA installation")
    
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        print(f"📊 PyTorch version: {torch.__version__}")
        print(f"🔧 CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            
            print(f"🎯 CUDA version: {torch.version.cuda}")
            print(f"🎮 GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                logger.info(f"GPU {i}: {gpu_name}")
                print(f"   GPU {i}: {gpu_name}")
        else:
            logger.warning("CUDA is not available in PyTorch despite installation attempt")
            
    except ImportError as e:
        logger.error(f"PyTorch not found after installation: {e}")
        print("❌ PyTorch not found after installation")


def main():
    """Main entry point for the CUDA installer."""
    import argparse
    
    # Setup logging first
    logger = setup_logging()
    logger.info("Starting CUDA installer")
    
    parser = argparse.ArgumentParser(description="Install PyTorch with CUDA support")
    parser.add_argument(
        "--cuda-version", 
        choices=["118", "121"], 
        help="Specific CUDA version to install"
    )
    parser.add_argument(
        "--cpu-only", 
        action="store_true", 
        help="Force CPU-only installation"
    )
    
    args = parser.parse_args()
    logger.info(f"Arguments: cuda_version={args.cuda_version}, cpu_only={args.cpu_only}")
    
    if args.cpu_only:
        logger.info("Forcing CPU-only installation")
        install_cpu_torch()
    else:
        install_cuda_torch(args.cuda_version)
    
    logger.info("CUDA installer completed")


if __name__ == "__main__":
    main() 