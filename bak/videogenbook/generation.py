"""Video generation functionality for videogenbook."""

import os
import time
import torch
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
import numpy as np

from .models import load_model, get_model_info, check_model_compatibility


@dataclass
class VideoGenerationConfig:
    """Configuration for video generation."""
    
    model_name: str
    prompt: str
    duration: float = 5.0
    fps: int = 24
    resolution: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    seed: Optional[int] = None
    output_path: str = "output.mp4"
    precision: str = "fp16"
    enable_audio: bool = False
    batch_size: int = 1


def generate_video(
    config: VideoGenerationConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, Any]:
    """Generate a video using the specified configuration.
    
    Args:
        config: Video generation configuration
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with generation results
    """
    start_time = time.time()
    
    try:
        # Load model
        if progress_callback:
            progress_callback(1, 10)
        
        model = load_model(
            config.model_name,
            precision=config.precision,
            enable_optimization=True
        )
        
        if progress_callback:
            progress_callback(3, 10)
        
        # Set random seed for reproducibility
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
        
        # Calculate number of frames
        num_frames = int(config.duration * config.fps)
        
        # Prepare generation parameters
        generation_kwargs = {
            "prompt": config.prompt,
            "num_frames": num_frames,
            "height": config.resolution,
            "width": config.resolution,
            "guidance_scale": config.guidance_scale,
            "num_inference_steps": config.num_inference_steps,
        }
        
        # Add audio generation if supported and requested
        model_info = get_model_info(config.model_name)
        if config.enable_audio and model_info and model_info.supports_audio:
            generation_kwargs["generate_audio"] = True
        
        if progress_callback:
            progress_callback(4, 10)
        
        # Generate video with improved settings
        with torch.inference_mode():
            if "ltx" in config.model_name.lower():
                # LTX-Video specific generation with better defaults
                result = model(
                    prompt=config.prompt, 
                    num_frames=num_frames, 
                    height=config.resolution, 
                    width=config.resolution,
                    guidance_scale=config.guidance_scale,
                    num_inference_steps=config.num_inference_steps
                )
                video_frames = result.frames[0]
            elif "wan" in config.model_name.lower():
                # Wan2.1 specific generation with better defaults
                result = model(
                    prompt=config.prompt, 
                    num_frames=num_frames, 
                    height=config.resolution, 
                    width=config.resolution,
                    guidance_scale=config.guidance_scale,
                    num_inference_steps=config.num_inference_steps
                )
                video_frames = result.frames[0]
            else:
                # Generic generation
                result = model(**generation_kwargs)
                video_frames = result
        
        if progress_callback:
            progress_callback(8, 10)
        
        # Save video
        save_result = save_video_output(video_frames, config.output_path, config.fps)
        
        if progress_callback:
            progress_callback(10, 10)
        
        generation_time = time.time() - start_time
        
        return {
            'success': True,
            'output_path': config.output_path,
            'generation_time': generation_time,
            'num_frames': num_frames,
            'resolution': f"{config.resolution}x{config.resolution}",
            'fps': config.fps,
            'file_size_mb': get_file_size_mb(config.output_path),
            'model_used': config.model_name,
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'generation_time': time.time() - start_time,
        }


def generate_with_audio(
    prompt: str,
    model_name: str = "google/veo-3",
    duration: float = 5.0,
    output_path: str = "output_with_audio.mp4"
) -> Dict[str, Any]:
    """Generate video with synchronized audio.
    
    Args:
        prompt: Text prompt for generation
        model_name: Model to use (must support audio)
        duration: Video duration in seconds
        output_path: Output file path
        
    Returns:
        Generation result dictionary
    """
    config = VideoGenerationConfig(
        model_name=model_name,
        prompt=prompt,
        duration=duration,
        output_path=output_path,
        enable_audio=True
    )
    
    # Verify model supports audio
    model_info = get_model_info(model_name)
    if not model_info or not model_info.supports_audio:
        return {
            'success': False,
            'error': f"Model {model_name} does not support audio generation"
        }
    
    return generate_video(config)


def batch_generate(
    prompts: List[str],
    model_name: str,
    output_dir: str = "batch_outputs",
    **kwargs
) -> List[Dict[str, Any]]:
    """Generate multiple videos from a list of prompts.
    
    Args:
        prompts: List of text prompts
        model_name: Model to use for generation
        output_dir: Directory to save outputs
        **kwargs: Additional generation parameters
        
    Returns:
        List of generation results
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    # Load model once for efficiency
    model = load_model(model_name, enable_optimization=True)
    
    for i, prompt in enumerate(prompts):
        output_path = os.path.join(output_dir, f"video_{i:03d}.mp4")
        
        config = VideoGenerationConfig(
            model_name=model_name,
            prompt=prompt,
            output_path=output_path,
            **kwargs
        )
        
        result = generate_video(config)
        result['prompt_index'] = i
        result['prompt'] = prompt
        results.append(result)
    
    return results


def save_video_output(
    video_frames: Any,
    output_path: str,
    fps: int = 24
) -> Dict[str, Any]:
    """Save video frames as video file.
    
    Args:
        video_frames: Video frames from model output
        output_path: Path to save video
        fps: Frames per second
        
    Returns:
        Save operation result
    """
    try:
        # Try using diffusers export_to_video first (recommended)
        try:
            from diffusers.utils import export_to_video
            export_to_video(video_frames, output_path, fps=fps)
            
            return {
                'success': True,
                'path': output_path,
                'num_frames': len(video_frames) if hasattr(video_frames, '__len__') else 'unknown',
                'fps': fps
            }
        except ImportError:
            # Fallback to manual saving
            pass
        
        # Manual saving using imageio
        import imageio
        
        # Handle different output formats
        if hasattr(video_frames, 'frames'):
            # Diffusers format
            frames = video_frames.frames[0]  # First video in batch
        elif isinstance(video_frames, dict) and 'frames' in video_frames:
            # Dictionary format
            frames = video_frames['frames']
        elif isinstance(video_frames, (list, tuple)):
            # List of frames
            frames = video_frames
        else:
            # Tensor format
            frames = video_frames
        
        # Convert PIL Images to numpy arrays if needed
        if hasattr(frames[0], 'convert'):  # PIL Images
            frames = [np.array(frame.convert('RGB')) for frame in frames]
            frames = np.array(frames)
        elif torch.is_tensor(frames):
            frames = frames.cpu().numpy()
        
        # Normalize to 0-255 range if needed
        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        
        # Ensure correct shape: (T, H, W, C)
        if frames.ndim == 4 and frames.shape[-1] != 3:
            # (T, C, H, W) -> (T, H, W, C)
            frames = frames.transpose(0, 2, 3, 1)
        
        # Save video
        imageio.mimsave(
            output_path,
            frames,
            fps=fps,
            codec='libx264',
            quality=8
        )
        
        return {
            'success': True,
            'path': output_path,
            'num_frames': len(frames),
            'fps': fps
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Failed to save video: {str(e)}"
        }


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0


def estimate_generation_time(
    config: VideoGenerationConfig,
    hardware_info: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """Estimate video generation time based on configuration and hardware.
    
    Args:
        config: Video generation configuration
        hardware_info: Optional hardware information
        
    Returns:
        Time estimates for different aspects
    """
    # Base estimates (in seconds) for RTX 4090
    base_times = {
        "google/veo-3": 8.0,      # Per second of video
        "kling-ai/kling-2.0": 6.0,
        "runway/gen-4": 5.0,
        "tencent/hunyuan-video": 4.0,
        "pku-yuan/opensora-2.0": 2.5,
        "pika-labs/pika-2.2": 3.5,
    }
    
    base_time_per_second = base_times.get(config.model_name, 4.0)
    
    # Resolution scaling factor
    resolution_factor = (config.resolution / 512) ** 2
    
    # Steps scaling factor  
    steps_factor = config.num_inference_steps / 50
    
    # Hardware scaling (if no info provided, assume mid-range)
    hardware_factor = 1.0
    if hardware_info:
        gpu_memory = hardware_info.get('memory_total', 16)
        if gpu_memory >= 24:
            hardware_factor = 0.8  # High-end GPU
        elif gpu_memory >= 16:
            hardware_factor = 1.0  # Mid-range GPU
        elif gpu_memory >= 8:
            hardware_factor = 1.5  # Entry-level GPU
        else:
            hardware_factor = 3.0  # CPU or very limited GPU
    
    # Calculate estimates
    base_time = base_time_per_second * config.duration
    scaled_time = base_time * resolution_factor * steps_factor * hardware_factor
    
    return {
        'estimated_seconds': scaled_time,
        'estimated_minutes': scaled_time / 60,
        'base_time': base_time,
        'resolution_factor': resolution_factor,
        'steps_factor': steps_factor,
        'hardware_factor': hardware_factor,
    }


def optimize_generation_config(
    config: VideoGenerationConfig,
    target_time_seconds: Optional[float] = None,
    target_quality: str = "balanced"
) -> VideoGenerationConfig:
    """Optimize generation configuration for performance or quality.
    
    Args:
        config: Original configuration
        target_time_seconds: Target generation time (None for quality focus)
        target_quality: Quality target ("fast", "balanced", "high")
        
    Returns:
        Optimized configuration
    """
    optimized = VideoGenerationConfig(**config.__dict__)
    
    if target_quality == "fast":
        optimized.num_inference_steps = 25
        optimized.guidance_scale = 6.0
        optimized.resolution = min(config.resolution, 512)
        optimized.precision = "fp16"
        
    elif target_quality == "high":
        optimized.num_inference_steps = 75
        optimized.guidance_scale = 8.5
        optimized.precision = "fp32"
        
    elif target_quality == "balanced":
        optimized.num_inference_steps = 50
        optimized.guidance_scale = 7.5
        optimized.precision = "fp16"
    
    # Time-based optimization
    if target_time_seconds:
        current_estimate = estimate_generation_time(optimized)['estimated_seconds']
        
        if current_estimate > target_time_seconds:
            # Reduce quality to meet time target
            scale_factor = target_time_seconds / current_estimate
            
            if scale_factor < 0.5:
                optimized.resolution = min(optimized.resolution, 256)
                optimized.num_inference_steps = max(20, int(optimized.num_inference_steps * 0.5))
            elif scale_factor < 0.75:
                optimized.resolution = min(optimized.resolution, 384)
                optimized.num_inference_steps = max(25, int(optimized.num_inference_steps * 0.7))
    
    return optimized