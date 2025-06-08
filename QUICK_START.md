# Quick Start Guide

## Installation

### Option 1: Simple pip install
```bash
pip install videogenbook
```

### Option 2: Conda environment (Recommended)
```bash
# Create conda environment
conda env create -f environment.yml
conda activate videogenbook

# Or manual setup
conda create -n videogenbook python=3.11
conda activate videogenbook
pip install videogenbook
```

### Option 3: Development install
```bash
git clone https://github.com/jenochs/video-generation-book.git
cd video-generation-book
conda env create -f environment.yml
conda activate videogenbook
pip install -e .
```

## Basic Usage

### CLI Commands

**1. Check system info:**
```bash
videogenbook info
```

**2. List available models:**
```bash
videogenbook models
```

**3. Generate a video:**
```bash
# Method 1: Quick generation with enhanced quality
videogenbook generate "Lightricks/LTX-Video" --prompt "a cat walking in garden"

# Method 2: High quality with custom settings
videogenbook generate "Lightricks/LTX-Video" \
  --prompt "a majestic horse running in the desert" \
  --resolution 1024 \
  --guidance-scale 9.0 \
  --steps 60 \
  --output "horse_hq.mp4"

# Method 3: Fast generation (lower quality)
videogenbook generate "Lightricks/LTX-Video" \
  --prompt "a robot dancing" \
  --resolution 512 \
  --guidance-scale 6.0 \
  --steps 25 \
  --output "robot_fast.mp4"
```

### Python API

**Simple one-liner (enhanced quality):**
```python
import videogenbook

# Generate with enhanced quality defaults (768p, better prompting)
video_path = videogenbook.generate("a cat walking in garden")
print(f"Video saved to: {video_path}")
# Output prompt: "high quality, detailed, cinematic, a cat walking in garden, well lit, clear focus"
```

**Advanced usage:**
```python
import videogenbook
from videogenbook import get_device

# Check your hardware
device = get_device()
print(f"Using device: {device}")

# Check GPU memory
gpu_info = videogenbook.check_gpu_memory()
if gpu_info:
    print(f"GPU: {gpu_info['name']} ({gpu_info['memory_total']:.1f}GB)")

# Generate with specific model
video_path = videogenbook.generate(
    prompt="a robot dancing in space",
    model="Lightricks/LTX-Video",
    output_path="robot_dance.mp4"
)
```

## Working Models

### Immediately Available (HuggingFace Hub)
- **Lightricks/LTX-Video** - High-quality DiT model (8GB+ VRAM)
- **Wan-AI/Wan2.1-T2V-1.3B-Diffusers** - Memory-efficient (6GB+ VRAM)

### With Correct URLs (May require setup)
- **tencent/HunyuanVideo** - 13B parameter open-source model
- **hpcai-tech/Open-Sora-v2** - Cost-optimized model

## Hardware Requirements

- **Minimum**: 8GB GPU VRAM, 16GB RAM
- **Recommended**: 16GB+ GPU VRAM, 32GB+ RAM  
- **Cloud Alternative**: Google Colab Pro with A100

## Common Issues

**Quick Fixes:**

1. **Installation problems**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for conda environment issues
2. **Import errors**: `pip install --upgrade diffusers>=0.33.1`
3. **CUDA issues**: Check `nvidia-smi` and install correct PyTorch version
4. **Memory errors**: Use `--resolution 512` or efficient model
5. **Model loading fails**: Clear cache with `rm -rf ~/.cache/huggingface`

**CLI Syntax:**
```bash
# ❌ Wrong (Python syntax)
videogenbook.generate("prompt")

# ✅ Correct (CLI syntax)
videogenbook generate "Lightricks/LTX-Video" --prompt "your prompt"
```

**For detailed troubleshooting**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## Next Steps

1. **Generate your first video** with the CLI
2. **Experiment with different prompts** and models
3. **Explore the Python API** for custom applications
4. **Check out the full documentation** in the repository

Happy video generating! 🎬