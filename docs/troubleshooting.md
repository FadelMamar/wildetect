# Troubleshooting Guide

Common issues and solutions for the WildDetect monorepo.

## Installation Issues

### uv command not found

**Solution**:
```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### ImportError after installation

**Solution**:
```bash
# Ensure virtual environment is activated
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Reinstall in development mode
cd wildetect
uv pip install -e .
```

## GPU and CUDA Issues

### CUDA out of memory

**Solutions**:
1. Reduce batch size:
   ```yaml
   processing:
     batch_size: 16  # Reduce from 32
   ```

2. Reduce tile size:
   ```yaml
   processing:
     tile_size: 640  # Reduce from 800
   ```

3. Clear GPU cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### GPU not detected

**Check CUDA**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**Solutions**:
1. Reinstall PyTorch with CUDA:
   ```bash
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. Set device explicitly:
   ```yaml
   model:
     device: "cuda:0"
   ```

## Windows-Specific Issues

### ProcessPool not supported

**Issue**: `ProcessPoolExecutor` doesn't work on Windows

**Solution**: Package automatically uses `ThreadPoolExecutor` on Windows

### Path issues

**Use forward slashes or raw strings**:
```python
# Good
path = "D:/data/images"
path = r"D:\data\images"

# Bad
path = "D:\data\images"  # Backslashes can cause issues
```

## MLflow Issues

### Can't connect to MLflow server

**Solutions**:
1. Start MLflow server:
   ```bash
   scripts\launch_mlflow.bat
   ```

2. Check environment variable:
   ```bash
   echo %MLFLOW_TRACKING_URI%
   # Should be: http://localhost:5000
   ```

3. Set in `.env`:
   ```
   MLFLOW_TRACKING_URI=http://localhost:5000
   ```

### Model not found in registry

**Solutions**:
1. List available models:
   ```bash
   mlflow models list
   ```

2. Check model name and alias:
   ```yaml
   model:
     mlflow_model_name: "detector"  # Check this is correct
     mlflow_model_alias: "production"  # or version number
   ```

## Data Loading Issues

### Images not found

**Solutions**:
1. Use absolute paths
2. Check file extensions match
3. Verify directory structure

### Annotation format errors

**Solutions**:
1. Validate COCO format:
   ```python
   from wildata.validation import validate_coco
   errors = validate_coco("annotations.json")
   ```

2. Check bbox coordinates
3. Verify image IDs match

## Performance Issues

### Detection is slow

**Solutions**:
1. Use GPU if available
2. Increase batch size
3. Use multithreaded pipeline:
   ```yaml
   processing:
     pipeline_type: "multithreaded"
     num_data_workers: 4
   ```

### High memory usage

**Solutions**:
1. Enable streaming mode (WilData):
   ```yaml
   processing_mode: "streaming"
   ```

2. Process in smaller batches
3. Clear cache between batches

## DVC Issues

### DVC push fails

**Solutions**:
1. Check remote configuration:
   ```bash
   dvc remote list
   ```

2. Verify credentials:
   ```bash
   dvc remote modify myremote access_key_id YOUR_KEY
   ```

3. Test connection:
   ```bash
   dvc remote list
   dvc status
   ```

## Label Studio Integration

### Can't connect to Label Studio

**Solutions**:
1. Start Label Studio:
   ```bash
   scripts\launch_labelstudio.bat
   ```

2. Check API key in `.env`:
   ```
   LABEL_STUDIO_API_KEY=your_key
   LABEL_STUDIO_URL=http://localhost:8080
   ```

## FiftyOne Issues

### FiftyOne app won't launch

**Solutions**:
1. Check FiftyOne installation:
   ```bash
   uv pip install fiftyone
   ```

2. Clear FiftyOne database:
   ```bash
   fiftyone app config database_dir
   ```

3. Use different port:
   ```bash
   fiftyone app launch --port 5152
   ```

## Common Error Messages

### "No module named 'wildetect'"

**Solution**: Install in development mode:
```bash
cd wildetect
uv pip install -e .
```

### "Permission denied"

**Solution**: Run as administrator or fix permissions:
```bash
icacls "D:\data" /grant %USERNAME%:F /t
```

### "Port already in use"

**Solution**: Kill process using port:
```bash
# Find process
netstat -ano | findstr :5000

# Kill process (replace PID)
taskkill /PID <PID> /F
```

## Getting Help

1. **Check logs**: Look in `logs/` directory
2. **Enable verbose mode**: Add `--verbose` flag
3. **GitHub Issues**: [Report bugs](https://github.com/fadelmamar/wildetect/issues)
4. **Discussions**: Ask questions in GitHub Discussions

## Debug Mode

Enable debug logging:

```yaml
logging:
  log_level: "DEBUG"
  verbose: true
```

Or in Python:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

**Still having issues?** Open an issue on GitHub with:
- Error message
- System info (`wildetect info`)
- Steps to reproduce
- Relevant configuration files

