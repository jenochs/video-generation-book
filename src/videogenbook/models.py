"""Model management and configuration for videogenbook."""

import os
import torch
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import warnings

from .utils import check_gpu_memory, get_hardware_recommendations


@dataclass
class ModelConfig:
    """Configuration for video generation models."""
    
    name: str
    model_id: str
    architecture: str
    max_resolution: int = 512
    max_duration: float = 10.0
    supports_audio: bool = False
    requires_login: bool = False
    min_vram_gb: int = 8
    recommended_vram_gb: int = 16
    precision_options: List[str] = None
    
    def __post_init__(self):
        if self.precision_options is None:
            self.precision_options = ["fp16", "fp32"]


# Registry of supported models with their configurations
# Default working models (accessible on HuggingFace Hub)
MODEL_REGISTRY = {
    "Lightricks/LTX-Video": ModelConfig(
        name="LTX-Video",
        model_id="Lightricks/LTX-Video",
        architecture="DiT-based Video Generation",
        max_resolution=1216,
        max_duration=10.0,
        supports_audio=False,
        requires_login=False,
        min_vram_gb=8,
        recommended_vram_gb=12,
        precision_options=["bf16", "fp16", "fp32"]
    ),
    
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": ModelConfig(
        name="Wan2.1 T2V 1.3B",
        model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        architecture="Large Diffusion Transformer",
        max_resolution=832,
        max_duration=6.0,
        supports_audio=False,
        requires_login=False,
        min_vram_gb=6,
        recommended_vram_gb=8,
        precision_options=["bf16", "fp16", "fp32"]
    ),

    "tencent/HunyuanVideo": ModelConfig(
        name="HunyuanVideo",
        model_id="tencent/HunyuanVideo", 
        architecture="Diffusion Transformer (13B)",
        max_resolution=720,
        max_duration=15.0,
        supports_audio=True,
        requires_login=False,
        min_vram_gb=40,  # Realistic requirement for full model
        recommended_vram_gb=80,  # Ideal for high quality
        precision_options=["fp16", "fp32", "int8"]
    ),
    
    "hpcai-tech/Open-Sora-v2": ModelConfig(
        name="OpenSora 2.0",
        model_id="hpcai-tech/Open-Sora-v2",
        architecture="Optimized Diffusion",
        max_resolution=512,
        max_duration=10.0,
        supports_audio=False,
        requires_login=False,
        min_vram_gb=6,
        recommended_vram_gb=12,
        precision_options=["fp16", "fp32", "int8"]
    ),

    # Colab-optimized configurations
    "tencent/HunyuanVideo-Colab": ModelConfig(
        name="HunyuanVideo-Colab",
        model_id="tencent/HunyuanVideo",
        architecture="Diffusion Transformer (13B) - Colab Optimized",
        max_resolution=544,  # Reduced for A100 40GB
        max_duration=10.0,   # Reduced for memory
        supports_audio=False, # Disabled for memory savings
        requires_login=False,
        min_vram_gb=40,
        recommended_vram_gb=40,  # Optimized for A100 40GB
        precision_options=["fp16"]  # Only FP16 for memory efficiency
    ),
}


def list_available_models() -> List[str]:
    """List all available video generation models.
    
    Returns:
        List of model names
    """
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_name: str) -> Optional[ModelConfig]:
    """Get configuration information for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        ModelConfig object or None if model not found
    """
    return MODEL_REGISTRY.get(model_name)


def check_model_compatibility(model_name: str) -> Dict[str, Any]:
    """Check if a model is compatible with current hardware.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        Dictionary with compatibility information
    """
    model_config = get_model_info(model_name)
    if not model_config:
        return {
            'compatible': False,
            'error': f"Unknown model: {model_name}"
        }
    
    gpu_info = check_gpu_memory()
    if not gpu_info:
        return {
            'compatible': False,
            'error': "No GPU detected",
            'recommendation': "Use Google Colab Pro for GPU access"
        }
    
    available_vram = gpu_info['memory_free']
    required_vram = model_config.min_vram_gb
    recommended_vram = model_config.recommended_vram_gb
    
    result = {
        'compatible': available_vram >= required_vram,
        'available_vram_gb': available_vram,
        'required_vram_gb': required_vram,
        'recommended_vram_gb': recommended_vram,
        'gpu_name': gpu_info['name']
    }
    
    if not result['compatible']:
        result['error'] = f"Insufficient GPU memory. Need {required_vram}GB, have {available_vram:.1f}GB"
        result['recommendations'] = [
            "Try a smaller model like opensora-2.0",
            "Use quantization (int8 precision)",
            "Enable memory optimization techniques",
            "Use Google Colab Pro with A100"
        ]
    elif available_vram < recommended_vram:
        result['warnings'] = [
            f"Below recommended {recommended_vram}GB VRAM",
            "Performance may be limited",
            "Consider using fp16 precision"
        ]
    
    return result


def load_model(
    model_name: str,
    precision: str = "fp16",
    enable_optimization: bool = True,
    device: Optional[str] = None
) -> Any:
    """Load a video generation model with optimizations.
    
    Args:
        model_name: Name of the model to load
        precision: Precision to use ("fp16", "fp32", "int8")
        enable_optimization: Whether to enable memory optimizations
        device: Device to load model on (auto-detected if None)
        
    Returns:
        Loaded model pipeline
    """
    model_config = get_model_info(model_name)
    if not model_config:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Check compatibility
    compatibility = check_model_compatibility(model_name)
    if not compatibility['compatible']:
        raise RuntimeError(f"Model not compatible: {compatibility['error']}")
    
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set precision
    torch_dtype = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "int8": torch.int8,
    }.get(precision, torch.float16)
    
    try:
        # Import required libraries
        from diffusers import DiffusionPipeline
        
        # Load model based on architecture  
        if "ltx-video" in model_name.lower() or "lightricks" in model_name.lower():
            return load_ltx_video_model(model_config, torch_dtype, device, enable_optimization)
        elif "wan2.1" in model_name.lower() or "wan-ai" in model_name.lower():
            return load_wan_model(model_config, torch_dtype, device, enable_optimization)
        elif "hunyuan" in model_name.lower():
            return load_hunyuan_model(model_config, torch_dtype, device, enable_optimization)
        elif "opensora" in model_name.lower():
            return load_opensora_model(model_config, torch_dtype, device, enable_optimization)
        elif "veo" in model_name.lower():
            return load_veo_model(model_config, torch_dtype, device, enable_optimization)
        elif "kling" in model_name.lower():
            return load_kling_model(model_config, torch_dtype, device, enable_optimization)
        elif "runway" in model_name.lower():
            return load_runway_model(model_config, torch_dtype, device, enable_optimization)
        elif "pika" in model_name.lower():
            return load_pika_model(model_config, torch_dtype, device, enable_optimization)
        else:
            # Generic diffusers loading
            return load_generic_model(model_config, torch_dtype, device, enable_optimization)
            
    except ImportError as e:
        raise ImportError(f"Required dependencies not installed: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}")


def load_ltx_video_model(config: ModelConfig, dtype: torch.dtype, device: str, optimize: bool):
    """Load LTX-Video model with specific optimizations."""
    try:
        from diffusers import LTXPipeline
        
        pipe = LTXPipeline.from_pretrained(
            config.model_id,
            torch_dtype=dtype
        )
        
        if optimize:
            # Use CPU offloading for memory efficiency (don't move to device)
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_tiling()
            return pipe
        else:
            # No offloading, move entire pipeline to device
            return pipe.to(device)
        
    except ImportError:
        raise ImportError("LTX-Video requires latest diffusers version with LTXPipeline support")


def load_wan_model(config: ModelConfig, dtype: torch.dtype, device: str, optimize: bool):
    """Load Wan2.1 model with specific optimizations."""
    try:
        from diffusers import WanPipeline
        
        pipe = WanPipeline.from_pretrained(
            config.model_id,
            torch_dtype=dtype
        )
        
        if optimize:
            # Use CPU offloading for memory efficiency
            pipe.enable_model_cpu_offload()
            return pipe
        else:
            return pipe.to(device)
        
    except ImportError:
        raise ImportError("Wan2.1 requires latest diffusers version with WanPipeline support")


def load_hunyuan_model(config: ModelConfig, dtype: torch.dtype, device: str, optimize: bool):
    """Load HunyuanVideo model with specific optimizations."""
    try:
        from diffusers import HunyuanVideoPipeline
        
        # Detect if running in Colab environment
        is_colab = 'google.colab' in str(get_ipython()) if 'get_ipython' in globals() else False
        
        pipe = HunyuanVideoPipeline.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None,
            low_cpu_mem_usage=True  # Essential for large models
        )
        
        if optimize:
            # Apply Colab-specific optimizations for A100 40GB
            if is_colab or config.name == "HunyuanVideo-Colab":
                print("🔧 Applying Colab A100 optimizations...")
                pipe.enable_sequential_cpu_offload()  # Most aggressive for 40GB
            else:
                pipe.enable_model_cpu_offload()  # Standard optimization
            
            pipe.enable_vae_tiling()
            pipe.enable_vae_slicing()  # Additional VAE optimization
            
            # Enable memory-efficient attention
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("   ✅ xFormers memory-efficient attention enabled")
            except ImportError:
                if hasattr(pipe, 'enable_memory_efficient_attention'):
                    pipe.enable_memory_efficient_attention()
                    print("   ✅ Memory-efficient attention enabled")
            
            # Configure scheduler for memory efficiency
            if hasattr(pipe.scheduler, 'enable_low_mem_usage'):
                pipe.scheduler.enable_low_mem_usage = True
                
            return pipe
        else:
            return pipe.to(device)
        
    except ImportError:
        raise ImportError("HunyuanVideo requires diffusers>=0.33.1 with HunyuanVideoPipeline support")


def load_opensora_model(config: ModelConfig, dtype: torch.dtype, device: str, optimize: bool):
    """Load OpenSora model with specific optimizations."""
    try:
        from diffusers import DiffusionPipeline
        
        pipe = DiffusionPipeline.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            custom_pipeline="opensora"
        )
        
        if optimize:
            # Use CPU offloading for memory efficiency
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_tiling()
            return pipe
        else:
            return pipe.to(device)
        
    except ImportError:
        raise ImportError("OpenSora requires custom pipeline components")


def load_veo_model(config: ModelConfig, dtype: torch.dtype, device: str, optimize: bool):
    """Load Google Veo model (placeholder - requires API access)."""
    warnings.warn("Veo 3 requires Google API access. Using mock implementation.")
    
    # This would be replaced with actual Veo API integration
    class MockVeoPipeline:
        def __init__(self):
            self.model_name = config.name
            
        def __call__(self, prompt, **kwargs):
            raise NotImplementedError("Veo 3 requires Google DeepMind API access")
    
    return MockVeoPipeline()


def load_kling_model(config: ModelConfig, dtype: torch.dtype, device: str, optimize: bool):
    """Load Kling AI model (placeholder - requires API access)."""
    warnings.warn("Kling AI 2.0 requires Kuaishou API access. Using mock implementation.")
    
    class MockKlingPipeline:
        def __init__(self):
            self.model_name = config.name
            
        def __call__(self, prompt, **kwargs):
            raise NotImplementedError("Kling AI 2.0 requires Kuaishou API access")
    
    return MockKlingPipeline()


def load_runway_model(config: ModelConfig, dtype: torch.dtype, device: str, optimize: bool):
    """Load Runway model (placeholder - requires API access)."""
    warnings.warn("Runway Gen-4 requires Runway API access. Using mock implementation.")
    
    class MockRunwayPipeline:
        def __init__(self):
            self.model_name = config.name
            
        def __call__(self, prompt, **kwargs):
            raise NotImplementedError("Runway Gen-4 requires Runway API access")
    
    return MockRunwayPipeline()


def load_pika_model(config: ModelConfig, dtype: torch.dtype, device: str, optimize: bool):
    """Load Pika Labs model (placeholder - requires API access).""" 
    warnings.warn("Pika 2.2 requires Pika Labs API access. Using mock implementation.")
    
    class MockPikaPipeline:
        def __init__(self):
            self.model_name = config.name
            
        def __call__(self, prompt, **kwargs):
            raise NotImplementedError("Pika 2.2 requires Pika Labs API access")
    
    return MockPikaPipeline()


def load_generic_model(config: ModelConfig, dtype: torch.dtype, device: str, optimize: bool):
    """Load generic diffusers model."""
    from diffusers import DiffusionPipeline
    
    pipe = DiffusionPipeline.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None
    )
    
    if optimize:
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
        if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except ImportError:
                pass  # xformers not available
    
    return pipe.to(device)


def get_colab_optimized_config(model_name: str, gpu_memory_gb: float = 40) -> Dict[str, Any]:
    """Get Colab-optimized generation configuration based on available GPU memory.
    
    Args:
        model_name: Name of the model
        gpu_memory_gb: Available GPU memory in GB
        
    Returns:
        Dictionary with optimized generation parameters
    """
    base_configs = {
        "tencent/HunyuanVideo": {
            "high_quality": {
                "height": 720, "width": 1280, "num_frames": 65,
                "guidance_scale": 7.0, "num_inference_steps": 30,
                "memory_requirement": 35
            },
            "balanced": {
                "height": 544, "width": 960, "num_frames": 65,
                "guidance_scale": 7.0, "num_inference_steps": 25,
                "memory_requirement": 25
            },
            "fast": {
                "height": 512, "width": 512, "num_frames": 32,
                "guidance_scale": 6.0, "num_inference_steps": 20,
                "memory_requirement": 15
            }
        }
    }
    
    if model_name not in base_configs:
        # Default configuration for unknown models
        return {
            "height": 512, "width": 512, "num_frames": 24,
            "guidance_scale": 7.0, "num_inference_steps": 25
        }
    
    configs = base_configs[model_name]
    
    # Select configuration based on available memory
    if gpu_memory_gb >= 35:
        return configs["high_quality"]
    elif gpu_memory_gb >= 25:
        return configs["balanced"]
    else:
        return configs["fast"]


def setup_colab_environment():
    """Setup optimal environment for Colab video generation."""
    import os
    
    # Environment variables for optimal Colab performance
    env_vars = {
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
        'TOKENIZERS_PARALLELISM': 'false',
        'CUDA_LAUNCH_BLOCKING': '0',
        'TRANSFORMERS_CACHE': '/content/hf_cache',
        'HF_HOME': '/content/hf_cache'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # PyTorch optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
    print("🔧 Colab environment optimized for video generation")
    return True


def get_model_recommendations(use_case: str = "general") -> List[str]:
    """Get model recommendations based on use case.
    
    Args:
        use_case: Type of use case ("general", "professional", "experimental", "low_memory")
        
    Returns:
        List of recommended model names
    """
    recommendations = {
        "general": [
            "Lightricks/LTX-Video",           # Most popular, working model
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", # Memory efficient, working
        ],
        "professional": [
            "Lightricks/LTX-Video",           # Real-time generation, high quality
            "google/veo-3",                   # Highest quality with audio (API)
            "kling-ai/kling-2.0",             # Best consistency (API)
        ],
        "experimental": [
            "Lightricks/LTX-Video",           # Working with diffusers
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", # Open source, working
            "tencent/HunyuanVideo",           # Full access to weights
        ],
        "low_memory": [
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", # Only 6GB VRAM required
            "Lightricks/LTX-Video",           # Efficient implementation
        ]
    }
    
    return recommendations.get(use_case, recommendations["general"])