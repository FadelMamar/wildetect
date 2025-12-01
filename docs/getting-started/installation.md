# Installation Guide

This guide will help you install all three packages in the WildDetect monorepo: **WilData**, **WildTrain**, and **WildDetect**.

## Prerequisites

### System Requirements

- **Python**: 3.9, 3.10, or 3.11
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (recommended) or pip
- **Git**: For cloning repositories
- **GPU** (optional): CUDA-capable GPU for faster inference and training

### Operating System

- Windows 10/11
- Linux (Ubuntu 20.04+ recommended)
- macOS (Intel or Apple Silicon)

!!! note "Windows Users"
    This monorepo is developed and tested on Windows. All scripts use `.bat` format for Windows compatibility.

## Installation Methods

=== "Method 1: Install from Source (Recommended)"

    ### 1. Clone the Repository
    
    ```bash
    git clone https://github.com/fadelmamar/wildetect.git
    cd wildetect
    ```

    ### 2. Create Virtual Environment
    
    Using `uv` (recommended):
    ```bash
    uv venv --python 3.10
    
    # Activate on Windows
    .venv\Scripts\activate
    
    # Activate on Linux/macOS
    source .venv/bin/activate
    ```

    ### 3. Install PyTorch (GPU or CPU)
    
    **With CUDA 11.8 (GPU):**
    ```bash
    uv pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    
    **CPU Only:**
    ```bash
    uv pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```

    ### 4. Install WildDetect Packages
    
    Install all three packages in development mode:
    ```bash
    # Install WilData
    cd wildata
    uv pip install -e .
    cd ..
    
    # Install WildTrain
    cd wildtrain
    uv pip install -e .
    cd ..
    
    # Install WildDetect (main package)
    uv pip install -e .
    ```

    ### 5. Install MMDetection (Optional)
    
    If you want to use MMDetection framework:
    ```bash
    # Install OpenMMLab dependencies
    uv pip install -U openmim
    uv run mim install mmengine
    
    # Install MMCV (choose based on your setup)
    # For CPU:
    uv pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html
    
    # For CUDA 11.8:
    uv pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
    
    # Install MMDetection
    uv run mim install mmdet
    uv pip install numpy==1.26.4
    ```

=== "Method 2: Install from GitHub"

    You can install packages directly from GitHub:
    
    ```bash
    # Install WilData
    uv pip install git+https://github.com/fadelmamar/wildata
    
    # Install WildTrain
    uv pip install git+https://github.com/fadelmamar/wildtrain
    
    # Install WildDetect
    uv pip install git+https://github.com/fadelmamar/wildetect
    ```

=== "Method 3: Using uv sync"

    If packages have `uv.lock` files:
    
    ```bash
    # In each package directory
    cd wildata
    uv sync
    
    cd ../wildtrain
    uv sync
    
    cd ../
    uv sync
    ```

## Verification

Verify your installation by checking package versions:

```bash
# Check WilData
wildata --version

# Check WildTrain  
wildtrain --version

# Check WildDetect
wildetect --version
```

You should also be able to import the packages in Python:

```python
import wildata
import wildtrain
import wildetect

print(f"WilData: {wildata.__version__}")
print(f"WildTrain: {wildtrain.__version__}")
print(f"WildDetect: {wildetect.__version__}")
```

## Optional Dependencies

### DVC (Data Version Control)

For dataset versioning with WilData:

```bash
# Basic DVC
uv pip install "wildata[dvc]"

# With cloud storage support
uv pip install "dvc[s3]"      # AWS S3
uv pip install "dvc[gcs]"     # Google Cloud Storage
uv pip install "dvc[azure]"   # Azure Blob Storage
```

### Label Studio Integration

For working with Label Studio annotations:

```bash
uv pip install label-studio-sdk
```

### FiftyOne Visualization

For interactive dataset visualization:

```bash
uv pip install fiftyone
```

## GPU Setup

### CUDA Configuration

If you have an NVIDIA GPU, ensure CUDA is properly installed:

1. **Check CUDA availability:**
   ```bash
   python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
   python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
   ```

2. **Check GPU devices:**
   ```bash
   python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"
   python -c "import torch; print(f'GPU Name: {torch.cuda.get_device_name(0)}')"
   ```

### Memory Requirements

| Task | Minimum RAM | Recommended RAM | GPU Memory |
|------|-------------|-----------------|------------|
| Detection | 8GB | 16GB | 4GB |
| Training | 16GB | 32GB | 8GB |
| Large Rasters | 32GB | 64GB | 8GB+ |

## Troubleshooting

### Common Issues

??? question "Import errors after installation"
    
    Make sure your virtual environment is activated:
    ```bash
    # Windows
    .venv\Scripts\activate
    
    # Linux/macOS
    source .venv/bin/activate
    ```

??? question "CUDA out of memory"
    
    Reduce batch size or tile size in your configuration files:
    ```yaml
    processing:
      batch_size: 16  # Reduce from 32
      tile_size: 640  # Reduce from 800
    ```

??? question "MMDetection installation fails"
    
    Install dependencies in this order:
    1. PyTorch
    2. MMCV (matching your CUDA version)
    3. MMEngine
    4. MMDetection

??? question "uv command not found"
    
    Install uv package manager:
    ```bash
    # Windows (PowerShell)
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    
    # Linux/macOS
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

### Windows-Specific Issues

!!! warning "ProcessPool Not Supported"
    On Windows, multiprocessing with `ProcessPoolExecutor` is not supported. The packages automatically use threading instead.

### Getting Help

If you encounter issues:

1. Check the [Troubleshooting Guide](../troubleshooting.md)
2. Search [GitHub Issues](https://github.com/fadelmamar/wildetect/issues)
3. Create a new issue with your error message and system info

## Next Steps

Once installation is complete:

1. 📚 [Set up your environment](environment-setup.md)
2. 🚀 [Follow the Quick Start guide](quick-start.md)
3. 📖 [Explore tutorials](../tutorials/end-to-end-detection.md)

---

**Installation successful?** Head to the [Environment Setup](environment-setup.md) to configure your workspace.

