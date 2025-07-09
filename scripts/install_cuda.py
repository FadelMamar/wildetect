#!/usr/bin/env python3
"""
Standalone script to install CUDA-compatible PyTorch using uv.
Run with: uv run scripts/install_cuda.py
"""

import subprocess
import sys
from pathlib import Path

# Add the src directory to the path so we can import wildetect
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wildetect.utils.cuda_installer import install_cuda_torch, install_cpu_torch


def main():
    """Main entry point for the CUDA installer script."""
    print("🚀 WildDetect CUDA Installer")
    print("=" * 40)
    
    # Check if CUDA is available
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ CUDA detected on this system")
        print("🔍 Installing PyTorch with CUDA support...")
        install_cuda_torch()
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ CUDA not available on this system")
        print("🔍 Installing CPU-only PyTorch...")
        install_cpu_torch()
    
    print("✅ Installation completed!")
    print("\nYou can now run: uv run wildetect info")
    print("to verify the installation.")


if __name__ == "__main__":
    main() 