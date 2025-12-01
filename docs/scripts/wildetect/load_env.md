# Environment Load Script

> **Location**: `scripts/load_env.bat`

**Purpose**: Utility script to load environment variables from `.env` file into the current shell session. This script is typically called automatically by other scripts, but can be used standalone.

## Usage

```batch
scripts\load_env.bat
```

Or called from other scripts:
```batch
call scripts\load_env.bat
```

## Functionality

The script:
1. Checks if `.env` file exists
2. Parses `.env` file line by line
3. Sets environment variables in the current shell
4. Skips comment lines (starting with `#`)
5. Skips empty lines

## Environment File Format

The `.env` file should contain key-value pairs:

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000

# Label Studio Configuration
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_API_KEY=your_api_key_here

# Data Paths
DATA_ROOT=D:/data/
MODELS_ROOT=D:/models/

# GPU Configuration
CUDA_VISIBLE_DEVICES=0

# Other settings
LOG_LEVEL=INFO
```

## Common Environment Variables

### MLflow

```bash
MLFLOW_TRACKING_URI=http://localhost:5000
```

### Label Studio

```bash
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_API_KEY=your_api_key
```

### Data Paths

```bash
DATA_ROOT=D:/data/
RESULTS_ROOT=D:/results/
```

### GPU

```bash
CUDA_VISIBLE_DEVICES=0
```

## Integration with Other Scripts

Most WildDetect scripts automatically call this script or use `--env-file .env` with `uv run`. The script is useful when:

- Running commands manually
- Debugging environment issues
- Setting up environment for manual operations

## Troubleshooting

### .env File Not Found

**Issue**: Script reports `.env` file not found

**Solutions**:
1. Verify `.env` file exists in project root
2. Check script is run from correct directory
3. Create `.env` file from `example.env` if needed
4. Verify file name is exactly `.env` (not `.env.txt`)

### Variables Not Set

**Issue**: Environment variables not available after running script

**Solutions**:
1. Verify `.env` file format is correct (KEY=VALUE)
2. Check for syntax errors in `.env` file
3. Ensure no spaces around `=` sign
4. Verify script executed successfully (no errors)
5. Note: Variables are set in the script's shell session only

### Syntax Errors

**Issue**: Script fails to parse `.env` file

**Solutions**:
1. Check for invalid characters in variable names
2. Ensure values with spaces are properly quoted (if needed)
3. Verify line endings are correct (Windows CRLF)
4. Check for special characters that need escaping

### Path Issues

**Issue**: Path variables not working correctly

**Solutions**:
1. Use forward slashes or escaped backslashes in paths
2. Use absolute paths instead of relative paths
3. Verify paths exist and are accessible
4. Check for trailing slashes (may cause issues)

## Best Practices

1. **Template File**: Keep `example.env` as a template
2. **Don't Commit Secrets**: Add `.env` to `.gitignore`
3. **Document Variables**: Document required variables in README
4. **Use Absolute Paths**: Prefer absolute paths for reliability
5. **Validate Format**: Ensure `.env` file follows standard format

## Related Documentation

- [Environment Setup](../../getting-started/environment-setup.md)
- [Installation Guide](../../getting-started/installation.md)
- [Configuration Files](../../configs/wildetect/index.md)

