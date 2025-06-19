"""
videogenbook - Companion package for 'Hands-On Video Generation with AI'

This package provides utilities, examples, and simplified interfaces for working with
2025's breakthrough video generation models including LTX-Video, HunyuanVideo, and OpenSora 2.0.
"""

__version__ = "0.1.0"
__author__ = "Jenochs"
__email__ = "joseph.enochs@gmail.com"

# Core functionality imports
from .models import (
    load_model,
    list_available_models,
    get_model_info,
    check_model_compatibility,
    ModelConfig,
    install_opensora,
    download_opensora_models,
    setup_opensora_environment,
    check_opensora_installation,
)

from .utils import (
    get_device,
    setup_environment,
    check_gpu_memory,
    optimize_memory,
    download_sample_data,
)

from .generation import (
    generate_video,
    generate_with_audio,
    batch_generate,
    VideoGenerationConfig,
)

from .evaluation import (
    evaluate_quality,
    benchmark_performance,
    compare_models,
    EvaluationMetrics,
)

# Convenience imports for common workflows
from .workflows import (
    quick_start,
    text_to_video,
    image_to_video,
    video_to_video,
)

__all__ = [
    # Simple API
    "generate",
    
    # Core modules
    "load_model",
    "list_available_models", 
    "get_model_info",
    "check_model_compatibility",
    "ModelConfig",
    
    # OpenSora 2.0 setup
    "install_opensora",
    "download_opensora_models", 
    "setup_opensora_environment",
    "check_opensora_installation",
    
    # Utilities
    "get_device",
    "setup_environment",
    "check_gpu_memory",
    "optimize_memory", 
    "download_sample_data",
    
    # Generation
    "generate_video",
    "generate_with_audio",
    "batch_generate",
    "VideoGenerationConfig",
    
    # Evaluation
    "evaluate_quality",
    "benchmark_performance",
    "compare_models",
    "EvaluationMetrics",
    
    # Workflows
    "quick_start",
    "text_to_video",
    "image_to_video",
    "video_to_video",
]

# Package metadata
__package_name__ = "videogenbook"
__description__ = "Companion package for 'Hands-On Video Generation with AI'"
__url__ = "https://github.com/jenochs/video-generation-book"
__license__ = "MIT"

# Supported models (2025 state-of-the-art)
# Working models available on HuggingFace Hub
WORKING_MODELS = [
    "Lightricks/LTX-Video",
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
]

# Models with correct HuggingFace Hub URLs
SUPPORTED_MODELS = [
    "Lightricks/LTX-Video",
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "tencent/HunyuanVideo",
    "hpcai-tech/Open-Sora-v2",
]

# Hardware requirements
MIN_GPU_MEMORY = 8  # GB
RECOMMENDED_GPU_MEMORY = 16  # GB
OPTIMAL_GPU_MEMORY = 24  # GB


# Simple API for easy use
def generate(prompt: str, model: str = "Lightricks/LTX-Video", output_path: str = "video.mp4"):
    """Generate a video from text prompt - simple one-liner API.
    
    Args:
        prompt: Text description of the video to generate
        model: Model to use (default: working LTX-Video model)
        output_path: Where to save the generated video
        
    Returns:
        Path to generated video file
        
    Example:
        >>> import videogenbook
        >>> video_path = videogenbook.generate("a cat walking in a garden")
        >>> print(f"Video saved to: {video_path}")
    """
    import os
    from .generation import generate_video, VideoGenerationConfig
    
    print(f"🚀 Starting video generation...")
    print(f"   🎬 Prompt: {prompt}")
    print(f"   🤖 Model: {model}")
    print(f"   📁 Output: {os.path.abspath(output_path)}")
    
    # Enhanced prompt for better quality
    enhanced_prompt = f"high quality, detailed, cinematic, {prompt}, well lit, clear focus"
    
    config = VideoGenerationConfig(
        model_name=model,
        prompt=enhanced_prompt,
        output_path=output_path,
        duration=5.0,
        resolution=768,  # Higher resolution
        guidance_scale=8.0,  # Better prompt adherence
        num_inference_steps=50,  # Better quality
        precision="fp16"
    )
    
    result = generate_video(config)
    
    if result['success']:
        abs_path = result['output_path']
        print(f"\n🎉 Generation completed successfully!")
        print(f"   📁 Video location: {abs_path}")
        print(f"   📊 File size: {result['file_size_mb']:.1f} MB")
        print(f"   ⏱️  Generation time: {result['generation_time']:.1f} seconds")
        return abs_path
    else:
        error_msg = f"Video generation failed: {result.get('error', 'Unknown error')}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
        