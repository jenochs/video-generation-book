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
                # LTX-Video specific generation with optimal parameters
                # Use bfloat16 if available for better performance
                print(f"🎬 Generating {num_frames} frames at {config.resolution}x{config.resolution}")
                
                result = model(
                    prompt=config.prompt,
                    width=config.resolution,
                    height=config.resolution, 
                    num_frames=num_frames,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    # LTX-Video specific optimizations
                    num_videos_per_prompt=1,
                    generator=torch.Generator().manual_seed(config.seed) if config.seed else None
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
            elif "opensora" in config.model_name.lower():
                # OpenSora v2 specific generation (isolated environment)
                result = model(
                    prompt=config.prompt,
                    num_frames=num_frames,
                    height=config.resolution,
                    width=config.resolution,
                    guidance_scale=config.guidance_scale,
                    num_inference_steps=config.num_inference_steps
                )
                # OpenSora returns file path instead of frames
                if hasattr(result, 'video_path'):
                    # Handle file-based output
                    import shutil
                    import os
                    final_output = config.output_path
                    shutil.copy2(result.video_path, final_output)
                    
                    # Create a dummy frames object for compatibility
                    video_frames = {"video_file": final_output, "frames_count": num_frames}
                else:
                    video_frames = result.frames[0] if hasattr(result, 'frames') else result
            else:
                # Generic generation
                result = model(**generation_kwargs)
                video_frames = result
        
        if progress_callback:
            progress_callback(8, 10)
        
        # Handle different output types
        if isinstance(video_frames, dict) and "video_file" in video_frames:
            # OpenSora file-based output - already saved
            print(f"✅ OpenSora video already saved to: {os.path.abspath(config.output_path)}")
            save_result = {
                'success': True,
                'path': config.output_path,
                'num_frames': video_frames.get('frames_count', num_frames),
                'fps': config.fps,
                'method': 'opensora_direct'
            }
        else:
            # Standard diffusers output - needs saving
            print(f"💾 Saving video to: {os.path.abspath(config.output_path)}")
            save_result = save_video_output(video_frames, config.output_path, config.fps)
            
            if not save_result['success']:
                raise RuntimeError(f"Failed to save video: {save_result.get('error', 'Unknown error')}")
        
        # Verify file was actually created
        if not os.path.exists(config.output_path):
            raise RuntimeError(f"Video file was not created at {config.output_path}")
        
        file_size_mb = get_file_size_mb(config.output_path)
        abs_path = os.path.abspath(config.output_path)
        
        print(f"✅ Video saved successfully!")
        print(f"   📁 Location: {abs_path}")
        print(f"   📊 Size: {file_size_mb:.1f} MB")
        print(f"   🎬 Frames: {save_result.get('num_frames', num_frames)}")
        print(f"   ⏱️  Duration: {num_frames/config.fps:.1f} seconds")
        
        if progress_callback:
            progress_callback(10, 10)
        
        generation_time = time.time() - start_time
        
        return {
            'success': True,
            'output_path': abs_path,
            'generation_time': generation_time,
            'num_frames': save_result.get('num_frames', num_frames),
            'resolution': f"{config.resolution}x{config.resolution}",
            'fps': config.fps,
            'file_size_mb': file_size_mb,
            'model_used': config.model_name,
            'save_result': save_result,
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
    """Save video frames as video file with robust error handling.
    
    Args:
        video_frames: Video frames from model output
        output_path: Path to save video
        fps: Frames per second
        
    Returns:
        Save operation result
    """
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"📁 Created directory: {output_dir}")
        
        # Try using diffusers export_to_video first (recommended for quality)
        try:
            from diffusers.utils import export_to_video
            print("🎬 Using diffusers export_to_video for optimal quality...")
            
            export_to_video(video_frames, output_path, fps=fps)
            
            # Verify file was created
            if os.path.exists(output_path):
                return {
                    'success': True,
                    'path': output_path,
                    'num_frames': len(video_frames) if hasattr(video_frames, '__len__') else 'unknown',
                    'fps': fps,
                    'method': 'diffusers_export'
                }
        except ImportError:
            print("⚠️  diffusers.utils.export_to_video not available, using manual saving...")
        except Exception as e:
            print(f"⚠️  diffusers export failed: {e}, trying manual saving...")
        
        # Manual saving using imageio with high quality settings
        import imageio
        print("🎬 Using imageio for video encoding...")
        
        # Handle different output formats
        if hasattr(video_frames, 'frames'):
            # Diffusers format
            frames = video_frames.frames[0]  # First video in batch
            print(f"   📋 Detected diffusers format: {len(frames)} frames")
        elif isinstance(video_frames, dict) and 'frames' in video_frames:
            # Dictionary format
            frames = video_frames['frames']
            print(f"   📋 Detected dictionary format: {len(frames)} frames")
        elif isinstance(video_frames, (list, tuple)):
            # List of frames
            frames = video_frames
            print(f"   📋 Detected list format: {len(frames)} frames")
        else:
            # Tensor format
            frames = video_frames
            print(f"   📋 Detected tensor format: {frames.shape if hasattr(frames, 'shape') else 'unknown shape'}")
        
        # Convert PIL Images to numpy arrays if needed
        if len(frames) > 0 and hasattr(frames[0], 'convert'):  # PIL Images
            print("   🔄 Converting PIL Images to numpy arrays...")
            frames = [np.array(frame.convert('RGB')) for frame in frames]
            frames = np.array(frames)
        elif torch.is_tensor(frames):
            print("   🔄 Converting tensor to numpy array...")
            frames = frames.cpu().numpy()
        
        # Normalize to 0-255 range if needed
        if frames.max() <= 1.0:
            print("   🔄 Normalizing to 0-255 range...")
            frames = (frames * 255).astype(np.uint8)
        
        # Ensure correct shape: (T, H, W, C)
        if frames.ndim == 4 and frames.shape[-1] != 3:
            print("   🔄 Transposing from (T, C, H, W) to (T, H, W, C)...")
            frames = frames.transpose(0, 2, 3, 1)
        
        print(f"   📐 Final frame shape: {frames.shape}")
        print(f"   🎯 Saving {len(frames)} frames at {fps} FPS...")
        
        # Save video with high quality settings
        imageio.mimsave(
            output_path,
            frames,
            fps=fps,
            codec='libx264',
            quality=9,  # Higher quality (0-10 scale)
            pixelformat='yuv420p',  # Compatible pixel format
            ffmpeg_params=['-crf', '18']  # High quality constant rate factor
        )
        
        # Verify file was created and has reasonable size
        if not os.path.exists(output_path):
            raise RuntimeError(f"Video file was not created at {output_path}")
        
        file_size = os.path.getsize(output_path)
        if file_size < 1000:  # Less than 1KB is probably an error
            raise RuntimeError(f"Video file is too small ({file_size} bytes), generation may have failed")
        
        return {
            'success': True,
            'path': output_path,
            'num_frames': len(frames),
            'fps': fps,
            'method': 'imageio',
            'file_size_bytes': file_size
        }
        
    except Exception as e:
        error_msg = f"Failed to save video: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            'success': False,
            'error': error_msg,
            'path': output_path
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