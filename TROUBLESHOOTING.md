# Troubleshooting Guide

## Conda Environment Issues

### Problem 1: Environment Creation Fails

**Error**: `CondaError` or `PackageNotFoundError` when creating environment

**Solutions**:
```bash
# 1. Update conda first
conda update conda

# 2. Create environment with specific Python version
conda create -n videogenbook python=3.11
conda activate videogenbook

# 3. Install packages step by step
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate
pip install videogenbook
```

### Problem 2: Package Installation Conflicts

**Error**: `pip` installation fails with dependency conflicts

**Solutions**:
```bash
# Method 1: Clean install
conda deactivate
conda remove -n videogenbook --all
conda create -n videogenbook python=3.11
conda activate videogenbook

# Method 2: Use conda-forge channel
conda install -c conda-forge python=3.11
pip install --no-deps videogenbook
pip install torch diffusers transformers

# Method 3: Install minimal dependencies
pip install torch>=2.0.0
pip install diffusers>=0.25.0
pip install transformers>=4.35.0
pip install videogenbook --no-deps
```

### Problem 3: CUDA/GPU Issues

**Error**: `CUDA not available` or GPU not detected

**Solutions**:
```bash
# 1. Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# 2. Install correct PyTorch version
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. CPU-only fallback
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Problem 4: Python Version Compatibility

**Error**: Package requires different Python version

**Solutions**:
```bash
# Check current Python version
python --version

# Create environment with specific Python version
conda create -n videogenbook python=3.11
conda activate videogenbook

# Supported Python versions: 3.9, 3.10, 3.11, 3.12
```

## Common Warnings (Not Errors)

### Warning: "model offloading...manually moving the pipeline to GPU"

**Message**: `It seems like you have activated model offloading by calling enable_model_cpu_offload...`

**What it means**: This is just a warning, not an error. Your video generation is working correctly!

**Why it happens**: The model uses CPU offloading for memory efficiency, which automatically manages GPU memory.

**Solutions** (optional):
```bash
# This is just a warning - you can ignore it
# Your videos are being generated successfully

# If you want to disable the warning, you can:
# 1. Continue using the current setup (recommended)
# 2. Or disable optimizations (uses more GPU memory):
videogenbook generate "Lightricks/LTX-Video" --prompt "test" --no-optimize
```

### Warning: "pkg_resources is deprecated"

**Message**: `UserWarning: pkg_resources is deprecated as an API...`

**What it means**: This is a deprecation warning from Python packaging. Your CLI is working perfectly!

**Why it happens**: Legacy code using old packaging APIs.

**Solutions**:
```bash
# This warning doesn't affect functionality
# You can safely ignore it - your commands work fine

# To suppress the warning (optional):
export PYTHONWARNINGS="ignore::UserWarning"
videogenbook --help
```

## Common Installation Errors

### Error: "No module named 'videogenbook'"

**Cause**: Package not installed or wrong environment activated

**Solutions**:
```bash
# 1. Check environment
conda info --envs
conda activate videogenbook

# 2. Install package
pip install videogenbook

# 3. Verify installation
python -c "import videogenbook; print('Success!')"
```

### Error: "Failed building wheel for..."

**Cause**: Missing build dependencies

**Solutions**:
```bash
# 1. Install build tools
conda install gcc_linux-64 gxx_linux-64  # Linux
# or
xcode-select --install  # macOS

# 2. Use pre-built wheels
pip install --only-binary=all videogenbook

# 3. Install dependencies separately
pip install wheel setuptools
pip install videogenbook
```

### Error: "ImportError: cannot import name 'LTXPipeline'"

**Cause**: Outdated diffusers version

**Solutions**:
```bash
# 1. Update diffusers
pip install --upgrade diffusers>=0.33.0

# 2. Use alternative model
videogenbook generate "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" --prompt "test"

# 3. Check versions
pip show diffusers transformers torch
```

## Model Loading Issues

### Error: "Failed to load model"

**Cause**: Network issues, insufficient disk space, or model incompatibility

**Solutions**:
```bash
# 1. Check disk space
df -h

# 2. Clear cache
rm -rf ~/.cache/huggingface

# 3. Use different model
videogenbook models  # List available models
videogenbook generate "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" --prompt "test"

# 4. Manual download
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Lightricks/LTX-Video')
"
```

### Error: "CUDA out of memory"

**Cause**: Insufficient GPU memory

**Solutions**:
```bash
# 1. Use smaller resolution
videogenbook generate "Lightricks/LTX-Video" --prompt "test" --resolution 512

# 2. Use memory-efficient model
videogenbook generate "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" --prompt "test"

# 3. Reduce inference steps
videogenbook generate "Lightricks/LTX-Video" --prompt "test" --steps 25

# 4. Check GPU memory
videogenbook info
```

## Performance Issues

### Problem: Very slow generation

**Solutions**:
```bash
# 1. Use GPU acceleration
videogenbook info  # Check if GPU is detected

# 2. Reduce quality settings
videogenbook generate "Lightricks/LTX-Video" \
  --prompt "test" \
  --resolution 512 \
  --steps 25 \
  --guidance-scale 6.0

# 3. Use faster model
videogenbook generate "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" --prompt "test"
```

## Environment-Specific Solutions

### Google Colab
```python
# Install in Colab
!pip install videogenbook
import videogenbook
videogenbook.generate("a cat walking")
```

### Local Development
```bash
# Complete setup
git clone https://github.com/jenochs/video-generation-book.git
cd video-generation-book
conda env create -f environment.yml
conda activate videogenbook
pip install -e .
```

### Docker (Alternative)
```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
RUN pip install videogenbook
CMD ["python", "-c", "import videogenbook; print('Ready!')"]
```

## Getting Help

### Diagnostic Commands
```bash
# System information
videogenbook info

# List models
videogenbook models

# Test basic functionality
python -c "
import videogenbook
from videogenbook import get_device
print(f'Device: {get_device()}')
print(f'Models: {len(videogenbook.list_available_models())}')
"

# Check package versions
pip show videogenbook torch diffusers transformers
```

### Log Files
```bash
# Enable verbose logging
export TORCH_LOGS="+dynamo"
export CUDA_LAUNCH_BLOCKING=1
videogenbook generate "Lightricks/LTX-Video" --prompt "test"
```

### Common Environment Variables
```bash
# Cache locations
export HF_HOME=~/.cache/huggingface
export TORCH_HOME=~/.cache/torch

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Disable warnings
export TOKENIZERS_PARALLELISM=false
```

## Quick Fixes Summary

1. **Can't install**: Update conda/pip, use conda-forge channel
2. **Import errors**: Update diffusers, check Python version
3. **CUDA issues**: Install correct PyTorch version, check nvidia-smi
4. **Memory errors**: Reduce resolution/steps, use efficient model
5. **Slow generation**: Check GPU detection, reduce quality settings
6. **Model loading fails**: Clear cache, check internet, use alternative model

## Still Having Issues?

1. **Check GitHub Issues**: https://github.com/jenochs/video-generation-book/issues
2. **Create minimal test case**: Save error output and system info
3. **Try Google Colab**: Often resolves local environment issues
4. **Use CPU-only mode**: For testing without GPU complications

```bash
# Minimal test command
videogenbook info
```

This should help identify most common issues!