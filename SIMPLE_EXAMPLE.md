# Simple Video Generation Example

The `videogenbook` library provides a simple, powerful toolbox for quickly loading and using state-of-the-art video generation models.

## Quick Start - Generate Your First Video!

By going to the Hugging Face Hub and filtering for models that generate videos based on a text prompt (text-to-video), we can find the most popular working models, such as LTX-Video and Wan2.1. We'll use LTX-Video, a DiT-based diffusion model capable of generating high-quality videos.

Given we have a model (LTX-Video) and a tool to use the model (diffusers), we can now generate our first video! When we load models, we'll need to send them to a specific hardware device, such as CPU (`cpu`), GPU (`cuda` or `cuda:0`), or Mac hardware called Metal (`mps`). The videogenbook library has a utility function to select an appropriate device depending on where you run the example code. For example, the following code will assign `cuda` to the device variable if you have a GPU:

```python
from videogenbook import get_device

device = get_device()
print(f"Using device: {device}")
```
```
Using device: cuda
```

Next, we'll generate our first video using the simple API. The videogenbook library offers a convenient, high-level function called `generate()`, which is ideal for this use case. Don't worry about all the parameters for now—the highlights include:

- There are many video generation models available, so we need to specify the one we want to use. We are going to use `Lightricks/LTX-Video`, a working model available on HuggingFace Hub.
- We need to specify the precision we'll load the model with. Models are composed of many parameters (millions or billions of them), and we can use `fp16` for faster inference with less memory usage.
- The first time you run this code, it can take a bit: the pipeline downloads a model of multiple gigabytes! Subsequent loads will be much faster thanks to HuggingFace's caching.

```python
import videogenbook

# Simple one-liner video generation
prompt = "a cat walking in a garden"
video_path = videogenbook.generate(prompt)
print(f"Video saved to: {video_path}")
```

## Advanced Usage

For more control, you can use the detailed API:

```python
import torch
from videogenbook import get_device, load_model, generate_video, VideoGenerationConfig

# Get optimal device
device = get_device()
print(f"Using device: {device}")

# Load the model with optimizations
model = load_model(
    "Lightricks/LTX-Video",
    precision="fp16",
    enable_optimization=True
)

# Configure generation parameters
config = VideoGenerationConfig(
    model_name="Lightricks/LTX-Video",
    prompt="a cat walking in a garden",
    duration=5.0,
    fps=24,
    resolution=512,
    output_path="my_video.mp4"
)

# Generate video
result = generate_video(config)

if result['success']:
    print(f"✅ Video generated: {result['output_path']}")
    print(f"⏱️ Time: {result['generation_time']:.1f}s")
    print(f"🎬 Frames: {result['num_frames']}")
else:
    print(f"❌ Failed: {result['error']}")
```

## Working Models

The following models are confirmed working on HuggingFace Hub:

- **Lightricks/LTX-Video**: High-quality DiT-based model (8GB+ VRAM)
- **Wan-AI/Wan2.1-T2V-1.3B-Diffusers**: Memory-efficient model (6GB+ VRAM)

Additional models with correct URLs (may require setup):
- **tencent/HunyuanVideo**: 13B parameter open-source model  
- **hpcai-tech/Open-Sora-v2**: Cost-optimized model

## Hardware Requirements

- **Minimum**: 8GB GPU VRAM, 16GB RAM
- **Recommended**: 16GB+ GPU VRAM, 32GB+ RAM
- **Alternative**: Google Colab Pro with A100 access

## Installation

```bash
# Simple install
pip install videogenbook

# Quick test
videogenbook info
videogenbook generate "a cat walking in a garden"
```

That's it! You're ready to generate your first AI video with state-of-the-art models using a simple, accessible API.