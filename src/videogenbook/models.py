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

    "hunyuanvideo-community/HunyuanVideo": ModelConfig(
        name="HunyuanVideo",
        model_id="hunyuanvideo-community/HunyuanVideo", 
        architecture="Diffusion Transformer (13B)",
        max_resolution=720,
        max_duration=15.0,
        supports_audio=False,  # Community version may not support audio
        requires_login=False,
        min_vram_gb=40,  # Realistic requirement for full model
        recommended_vram_gb=80,  # Ideal for high quality
        precision_options=["fp16", "fp32", "int8"]
    ),
    
    "hpcai-tech/Open-Sora-v2": ModelConfig(
        name="OpenSora 2.0",
        model_id="hpcai-tech/Open-Sora-v2",
        architecture="STDiT3 Transformer (11B)",
        max_resolution=768,  # Supports 768x768 in v1.3
        max_duration=15.0,   # Supports 4k+1 frames, max 129 frames
        supports_audio=False,
        requires_login=False,
        min_vram_gb=12,      # Realistic for 11B model
        recommended_vram_gb=24,  # Better performance
        precision_options=["fp16", "fp32", "bf16"]
    ),

    # Colab-optimized configurations
    "hunyuanvideo-community/HunyuanVideo-Colab": ModelConfig(
        name="HunyuanVideo-Colab",
        model_id="hunyuanvideo-community/HunyuanVideo",
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
        elif "opensora" in model_name.lower() or "open-sora" in model_name.lower():
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
    """Load LTX-Video model with officially supported diffusers pipelines."""
    try:
        # Import officially supported LTX-Video pipelines
        from diffusers import LTXPipeline, LTXImageToVideoPipeline, LTXConditionPipeline
        from diffusers.utils import export_to_video
        from transformers import T5Tokenizer, T5EncoderModel
        
        print("🔧 Loading LTX-Video using official diffusers support...")
        print("✅ Using officially supported LTXPipeline from diffusers")
        
        # Determine optimal dtype (diffusers docs recommend bfloat16)
        optimal_dtype = torch.bfloat16 if dtype == torch.float16 and torch.cuda.is_available() else dtype
        if optimal_dtype != dtype:
            print(f"🔧 Using optimal dtype {optimal_dtype} instead of {dtype} for better performance")
        
        # Try multiple loading strategies for LTX-Video
        loading_strategies = [
            # Strategy 1: Direct loading (may work with newer diffusers)
            {
                "name": "Direct Loading",
                "method": "direct"
            },
            # Strategy 2: Force T5 tokenizer reload
            {
                "name": "T5 Component Pre-loading",
                "method": "t5_preload"
            },
            # Strategy 3: Trust remote code
            {
                "name": "Trust Remote Code",
                "method": "trust_remote"
            }
        ]
        
        last_error = None
        for i, strategy in enumerate(loading_strategies, 1):
            try:
                print(f"🔧 Trying strategy {i}: {strategy['name']}...")
                
                if strategy['method'] == 'direct':
                    # Direct loading attempt
                    pipe = LTXPipeline.from_pretrained(
                        config.model_id,
                        torch_dtype=optimal_dtype,
                        use_safetensors=True
                    )
                
                elif strategy['method'] == 't5_preload':
                    # Pre-load T5 components explicitly
                    print("📥 Pre-loading T5 tokenizer and text encoder...")
                    
                    # Try different T5 loading approaches
                    try:
                        # Try the LTX-Video specific T5 components
                        tokenizer = T5Tokenizer.from_pretrained("Lightricks/LTX-Video", subfolder="tokenizer")
                        text_encoder = T5EncoderModel.from_pretrained("Lightricks/LTX-Video", subfolder="text_encoder", torch_dtype=optimal_dtype)
                    except:
                        # Fallback to standard T5 model
                        print("🔄 Using standard T5 model as fallback...")
                        tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-large")  # Smaller model for compatibility
                        text_encoder = T5EncoderModel.from_pretrained("google/t5-v1_1-large", torch_dtype=optimal_dtype)
                    
                    pipe = LTXPipeline.from_pretrained(
                        config.model_id,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        torch_dtype=optimal_dtype,
                        use_safetensors=True
                    )
                
                elif strategy['method'] == 'trust_remote':
                    # Allow trust_remote_code in case of custom components
                    pipe = LTXPipeline.from_pretrained(
                        config.model_id,
                        torch_dtype=optimal_dtype,
                        use_safetensors=True,
                        trust_remote_code=True
                    )
                
                print(f"✅ Strategy {i} succeeded: LTX-Video pipeline loaded successfully")
                break
                
            except Exception as e:
                print(f"❌ Strategy {i} failed: {str(e)[:100]}...")
                last_error = e
                continue
        else:
            # All strategies failed
            raise RuntimeError(f"All LTX-Video loading strategies failed. Last error: {last_error}")
        
        print("✅ LTX-Video pipeline loaded successfully")
        
        # Also load image-to-video pipeline for advanced usage
        try:
            print("📥 Loading LTX-Video Image-to-Video pipeline...")
            img2vid_pipe = LTXImageToVideoPipeline.from_pretrained(
                config.model_id,
                torch_dtype=optimal_dtype,
                use_safetensors=True
            )
            print("✅ LTX-Video Image-to-Video pipeline loaded successfully")
            
            # Add image-to-video capability to main pipeline
            pipe.img2vid = img2vid_pipe
        except Exception as e:
            print(f"⚠️  Image-to-Video pipeline failed to load: {e}")
            pipe.img2vid = None
        
        # Apply optimizations
        if optimize:
            print("🔧 Applying memory optimizations...")
            try:
                pipe.enable_model_cpu_offload()
                print("   ✅ CPU offloading enabled")
                
                if hasattr(pipe, 'img2vid') and pipe.img2vid:
                    pipe.img2vid.enable_model_cpu_offload()
                    print("   ✅ Image-to-Video CPU offloading enabled")
            except Exception as e:
                print(f"   ⚠️  CPU offloading failed: {e}")
            
            try:
                if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_tiling'):
                    pipe.vae.enable_tiling()
                    print("   ✅ VAE tiling enabled")
            except Exception as e:
                print(f"   ⚠️  VAE tiling failed: {e}")
            
            # Enable memory efficient attention if available
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("   ✅ xFormers memory efficient attention enabled")
            except Exception as e:
                print(f"   ⚠️  xFormers not available: {e}")
            
            return pipe
        else:
            # Move to device without offloading
            pipe = pipe.to(device)
            if hasattr(pipe, 'img2vid') and pipe.img2vid:
                pipe.img2vid = pipe.img2vid.to(device)
            return pipe
        
    except ImportError as e:
        raise ImportError(f"LTX-Video requires diffusers>=0.33.1 with official LTX support. Error: {e}")


def load_wan_model(config: ModelConfig, dtype: torch.dtype, device: str, optimize: bool):
    """Load Wan2.1 model with specific optimizations."""
    try:
        # Try multiple Wan pipeline variants
        try:
            from diffusers import WanPipeline
        except ImportError:
            try:
                from diffusers import VideoWanPipeline as WanPipeline
            except ImportError:
                # Fallback to generic pipeline
                from diffusers import DiffusionPipeline as WanPipeline
        
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
        # Try multiple HunyuanVideo pipeline variants
        try:
            from diffusers import HunyuanVideoPipeline
        except ImportError:
            try:
                from diffusers import HunyuanVideoTxt2VideoPipeline as HunyuanVideoPipeline
            except ImportError:
                # Fallback to generic pipeline
                from diffusers import DiffusionPipeline as HunyuanVideoPipeline
        
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
    """Load OpenSora model with proper native integration."""
    try:
        print("🔧 Loading OpenSora 2.0...")
        print("📋 OpenSora 2.0 requires native installation due to diffusers integration limitations")
        
        # Try multiple loading strategies for OpenSora (prioritize isolated environment)
        loading_strategies = [
            # Strategy 1: Isolated environment (our preferred approach)
            {
                "name": "Isolated Environment",
                "method": "isolated"
            },
            # Strategy 2: Native OpenSora installation
            {
                "name": "Native OpenSora Installation", 
                "method": "native"
            },
            # Strategy 3: Fallback mock implementation
            {
                "name": "Mock Implementation",
                "method": "mock"
            }
        ]
        
        last_error = None
        for i, strategy in enumerate(loading_strategies, 1):
            try:
                print(f"🔧 Trying strategy {i}: {strategy['name']}...")
                
                if strategy['method'] == 'isolated':
                    return _load_opensora_isolated(config, dtype, device, optimize)
                elif strategy['method'] == 'native':
                    return _load_opensora_native(config, dtype, device, optimize)
                elif strategy['method'] == 'mock':
                    return _load_opensora_mock(config, dtype, device, optimize)
                    
            except Exception as e:
                print(f"❌ Strategy {i} failed: {str(e)[:100]}...")
                last_error = e
                continue
        
        # If all strategies failed
        error_msg = f"""
❌ All OpenSora loading strategies failed!

OpenSora 2.0 is not yet fully integrated with diffusers library.
Last error: {last_error}

To use OpenSora 2.0:
1. Install directly from GitHub:
   git clone https://github.com/hpcaitech/Open-Sora.git
   cd Open-Sora && pip install -e .

2. Download model weights:
   huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir ./ckpts

3. Use native OpenSora inference:
   python scripts/inference.py --config configs/inference/t2v.py

Alternative working models:
   videogenbook generate "Lightricks/LTX-Video" --prompt "your prompt"
   videogenbook generate "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" --prompt "your prompt"
"""
        raise RuntimeError(error_msg)
        
    except ImportError as e:
        raise ImportError(f"OpenSora requires additional dependencies: {e}")


def _load_opensora_isolated(config: ModelConfig, dtype: torch.dtype, device: str, optimize: bool):
    """Try to load OpenSora via isolated environment setup."""
    try:
        import subprocess
        import os
        
        print("🔍 Setting up OpenSora v2 in isolated environment...")
        
        # Check if OpenSora isolated environment exists
        opensora_env = _get_or_create_opensora_environment()
        
        if opensora_env["status"] == "available":
            print(f"✅ OpenSora environment ready: {opensora_env['env_path']}")
            return _create_opensora_isolated_wrapper(opensora_env, dtype, device, optimize)
        elif opensora_env["status"] == "setup_required":
            print("⚙️  OpenSora environment needs setup. This will not affect your current environment.")
            
            # Ask user if they want to proceed with automated setup
            print("🤖 Would you like to automatically set up OpenSora in an isolated environment?")
            print("   This will:")
            print("   - Create a separate conda environment 'opensora-env'")
            print("   - Install OpenSora with Python 3.10 and PyTorch 2.4.0")
            print("   - Download model weights")
            print("   - Keep your current environment completely unchanged")
            
            # For automated mode, proceed with setup
            setup_result = _setup_opensora_environment()
            
            if setup_result["success"]:
                print("✅ OpenSora environment setup completed successfully")
                return _create_opensora_isolated_wrapper(setup_result["env_info"], dtype, device, optimize)
            else:
                raise RuntimeError(f"OpenSora environment setup failed: {setup_result['error']}")
        else:
            raise RuntimeError(f"OpenSora environment check failed: {opensora_env.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"⚠️  OpenSora isolated setup failed: {e}")
        print("💡 OpenSora requires a separate environment. Your current models (LTX-Video, Wan2.1) remain unaffected.")
        raise RuntimeError(f"OpenSora isolated setup failed: {e}")


def _load_opensora_native(config: ModelConfig, dtype: torch.dtype, device: str, optimize: bool):
    """Try to load OpenSora via native installation."""
    try:
        # Check if OpenSora is installed
        import opensora
        from opensora.models import STDiT3
        from opensora.models.vae import VideoAutoencoderKL
        
        print("✅ Found native OpenSora installation")
        
        # Create a wrapper pipeline
        class OpenSoraPipeline:
            def __init__(self, model_path, device, dtype):
                self.device = device
                self.dtype = dtype
                self.model_path = model_path
                
                # Load components
                print("🔧 Loading OpenSora components...")
                self.vae = VideoAutoencoderKL.from_pretrained(
                    f"{model_path}/vae",
                    torch_dtype=dtype
                ).to(device)
                
                self.transformer = STDiT3.from_pretrained(
                    f"{model_path}/transformer",
                    torch_dtype=dtype
                ).to(device)
                
                print("✅ OpenSora pipeline loaded successfully")
            
            def __call__(self, prompt, num_frames=65, height=768, width=768, **kwargs):
                """Generate video from text prompt."""
                print(f"🎬 Generating video: {prompt}")
                print(f"📐 Resolution: {height}x{width}, Frames: {num_frames}")
                
                # This would be the actual generation code
                # For now, return a placeholder that explains the limitation
                raise NotImplementedError(
                    "Native OpenSora generation requires full setup. "
                    "Please use the official OpenSora inference scripts."
                )
        
        # Try to find the model path
        model_path = "./ckpts"  # Default path from download instructions
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "OpenSora model not found. Please download with: "
                "huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir ./ckpts"
            )
        
        return OpenSoraPipeline(model_path, device, dtype)
        
    except ImportError:
        raise ImportError(
            "Native OpenSora not installed. Install with: "
            "git clone https://github.com/hpcaitech/Open-Sora.git && "
            "cd Open-Sora && pip install -e ."
        )
    except Exception as e:
        raise RuntimeError(f"Native OpenSora loading failed: {e}")


def _load_opensora_mock(config: ModelConfig, dtype: torch.dtype, device: str, optimize: bool):
    """Create a mock OpenSora pipeline with helpful instructions."""
    print("⚠️  Using mock OpenSora implementation")
    
    class MockOpenSoraPipeline:
        def __init__(self):
            self.model_name = config.name
            self.model_id = config.model_id
            
        def __call__(self, prompt, **kwargs):
            instructions = f"""
🎬 OpenSora 2.0 Generation Request
Prompt: {prompt}
Model: {self.model_name}

❌ OpenSora 2.0 is not yet integrated with diffusers.

✅ To use OpenSora 2.0:

1. Install OpenSora:
   git clone https://github.com/hpcaitech/Open-Sora.git
   cd Open-Sora
   pip install -e .

2. Download model:
   huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir ./ckpts

3. Generate video:
   python scripts/inference.py configs/inference/t2v.py \\
     --ckpt-path ./ckpts/model.safetensors \\
     --prompt "{prompt}"

🔄 Alternative working models:
   • Lightricks/LTX-Video (Real-time generation)
   • Wan-AI/Wan2.1-T2V-1.3B-Diffusers (Memory efficient)
"""
            raise NotImplementedError(instructions)
    
    return MockOpenSoraPipeline()


def _create_opensora_native_wrapper(model_path: str, dtype: torch.dtype, device: str, optimize: bool):
    """Create a wrapper for native OpenSora inference."""
    import subprocess
    import os
    import tempfile
    import json
    
    class OpenSoraNativeWrapper:
        def __init__(self, model_path, dtype, device):
            self.model_path = model_path
            self.dtype = dtype
            self.device = device
            self.opensora_repo = None
            
            # Check if OpenSora repo is available
            possible_repos = ["./Open-Sora", "../Open-Sora", "./opensora", "./OpenSora"]
            for repo in possible_repos:
                if os.path.exists(repo) and os.path.exists(os.path.join(repo, "scripts", "diffusion", "inference.py")):
                    self.opensora_repo = os.path.abspath(repo)
                    print(f"✅ Found OpenSora repository at: {self.opensora_repo}")
                    break
            
            if not self.opensora_repo:
                print("⚠️  OpenSora repository not found. Please clone with:")
                print("   git clone https://github.com/hpcaitech/Open-Sora.git")
                raise RuntimeError("OpenSora repository required for native inference")
        
        def __call__(self, prompt, num_frames=65, height=768, width=768, guidance_scale=7.0, num_inference_steps=50, **kwargs):
            """Generate video using native OpenSora inference."""
            print(f"🎬 Generating OpenSora v2 video:")
            print(f"   📝 Prompt: {prompt}")
            print(f"   📐 Resolution: {height}x{width}")
            print(f"   🎞️  Frames: {num_frames}")
            
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, "opensora_output")
                os.makedirs(output_path, exist_ok=True)
                
                # Determine config based on resolution
                if height <= 256 or width <= 256:
                    config_file = "configs/diffusion/inference/t2i2v_256px.py"
                else:
                    config_file = "configs/diffusion/inference/t2i2v_768px.py"
                
                config_path = os.path.join(self.opensora_repo, config_file)
                if not os.path.exists(config_path):
                    # Fallback to basic config
                    config_path = os.path.join(self.opensora_repo, "configs/diffusion/inference/t2i2v_256px.py")
                
                print(f"📋 Using config: {config_file}")
                
                # Run OpenSora inference
                try:
                    cmd = [
                        "python", 
                        os.path.join(self.opensora_repo, "scripts", "diffusion", "inference.py"),
                        config_path,
                        "--prompt", prompt,
                        "--save-dir", output_path
                    ]
                    
                    print(f"🚀 Running: {' '.join(cmd)}")
                    
                    # Change to OpenSora directory for execution
                    original_dir = os.getcwd()
                    os.chdir(self.opensora_repo)
                    
                    try:
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=600,  # 10 minute timeout
                            cwd=self.opensora_repo
                        )
                        
                        if result.returncode != 0:
                            print(f"❌ OpenSora inference failed:")
                            print(f"   Error: {result.stderr}")
                            raise RuntimeError(f"Inference failed: {result.stderr}")
                        
                        print("✅ OpenSora inference completed")
                        
                        # Find generated video files
                        video_files = []
                        for root, dirs, files in os.walk(output_path):
                            for file in files:
                                if file.endswith(('.mp4', '.avi', '.mov')):
                                    video_files.append(os.path.join(root, file))
                        
                        if not video_files:
                            raise RuntimeError("No video files generated")
                        
                        # Return the first video file found
                        video_path = video_files[0]
                        print(f"📹 Generated video: {video_path}")
                        
                        # Create a result object that mimics diffusers output
                        class OpenSoraResult:
                            def __init__(self, video_path):
                                self.video_path = video_path
                                self.frames = self._load_video_frames()
                            
                            def _load_video_frames(self):
                                # For now, return the video path
                                # Could be enhanced to actually load frames
                                return [[self.video_path]]
                        
                        return OpenSoraResult(video_path)
                        
                    finally:
                        os.chdir(original_dir)
                
                except subprocess.TimeoutExpired:
                    raise RuntimeError("OpenSora inference timed out (10 minutes)")
                except Exception as e:
                    raise RuntimeError(f"OpenSora inference error: {e}")
    
    return OpenSoraNativeWrapper(model_path, dtype, device)


def _get_or_create_opensora_environment():
    """Check if OpenSora isolated environment exists or can be created."""
    import subprocess
    import os
    
    # Check if conda is available
    try:
        result = subprocess.run(["conda", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            return {"status": "error", "error": "Conda not available"}
    except FileNotFoundError:
        return {"status": "error", "error": "Conda not installed"}
    
    # Check if opensora-v2 already exists (our actual environment name)
    result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
    if "opensora-v2" in result.stdout:
        # Check if environment is properly set up
        env_path = _find_conda_env_path("opensora-v2")
        if env_path and _validate_opensora_environment(env_path):
            return {
                "status": "available",
                "env_path": env_path,
                "env_name": "opensora-v2",
                "opensora_path": "/home/jenochs/Downloads/chapter_1/video-generation-book/opensora-workspace/opensora-workspace/Open-Sora",
                "python_path": f"{env_path}/bin/python"
            }
        else:
            return {
                "status": "setup_required",
                "message": "Environment exists but needs setup"
            }
    else:
        return {
            "status": "setup_required", 
            "message": "Environment needs to be created"
        }


def _find_conda_env_path(env_name):
    """Find the path to a conda environment."""
    import subprocess
    import json
    
    try:
        result = subprocess.run(["conda", "env", "list", "--json"], capture_output=True, text=True)
        if result.returncode == 0:
            env_info = json.loads(result.stdout)
            for env_path in env_info.get("envs", []):
                if env_name in env_path:
                    return env_path
    except:
        pass
    return None


def _validate_opensora_environment(env_path):
    """Check if OpenSora environment is properly configured."""
    import subprocess
    import os
    
    # Check if Python 3.10 is installed
    python_path = os.path.join(env_path, "bin", "python")
    if not os.path.exists(python_path):
        python_path = os.path.join(env_path, "Scripts", "python.exe")  # Windows
    
    if not os.path.exists(python_path):
        return False
    
    try:
        # Check Python version
        result = subprocess.run([python_path, "--version"], capture_output=True, text=True)
        if "3.10" not in result.stdout:
            return False
        
        # Check if OpenSora package structure exists at our actual location
        opensora_script_path = "/home/jenochs/Downloads/chapter_1/video-generation-book/opensora-workspace/opensora-workspace/Open-Sora/scripts/diffusion/inference.py"
        if os.path.exists(opensora_script_path):
            return True
            
    except:
        pass
    
    return False


def _setup_opensora_environment():
    """Set up isolated OpenSora environment."""
    import subprocess
    import os
    
    print("🔧 Setting up isolated OpenSora environment...")
    
    try:
        # Create conda environment with Python 3.10
        print("📦 Creating conda environment 'opensora-env' with Python 3.10...")
        result = subprocess.run([
            "conda", "create", "-n", "opensora-env", "python=3.10", "-y"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create conda environment: {result.stderr}")
        
        # Get environment path
        env_path = _find_conda_env_path("opensora-env")
        if not env_path:
            raise RuntimeError("Failed to find created environment")
        
        # Determine activation command
        python_path = os.path.join(env_path, "bin", "python")
        pip_path = os.path.join(env_path, "bin", "pip")
        if not os.path.exists(python_path):
            # Windows paths
            python_path = os.path.join(env_path, "Scripts", "python.exe")
            pip_path = os.path.join(env_path, "Scripts", "pip.exe")
        
        # Create isolation directory
        isolation_dir = "./opensora-isolation"
        os.makedirs(isolation_dir, exist_ok=True)
        
        # Clone OpenSora repository into isolation directory
        opensora_path = os.path.join(isolation_dir, "Open-Sora")
        if not os.path.exists(opensora_path):
            print("📥 Cloning OpenSora repository...")
            result = subprocess.run([
                "git", "clone", "https://github.com/hpcaitech/Open-Sora.git", opensora_path
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Failed to clone OpenSora: {result.stderr}")
        
        # Install PyTorch 2.4.0 in the isolated environment
        print("🔧 Installing PyTorch 2.4.0...")
        result = subprocess.run([
            pip_path, "install", "torch==2.4.0", "torchvision==0.19.0", 
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"⚠️  PyTorch installation warning: {result.stderr}")
        
        # Install OpenSora in the isolated environment
        print("⚙️  Installing OpenSora package...")
        original_dir = os.getcwd()
        os.chdir(opensora_path)
        
        try:
            result = subprocess.run([
                pip_path, "install", "-v", "."
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"OpenSora installation failed: {result.stderr}")
            
        finally:
            os.chdir(original_dir)
        
        # Download models
        print("📥 Downloading OpenSora models...")
        models_path = os.path.join(isolation_dir, "ckpts")
        result = subprocess.run([
            python_path, "-m", "huggingface_hub.commands.download_cli",
            "hpcai-tech/Open-Sora-v2", "--local-dir", models_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"⚠️  Model download failed: {result.stderr}")
            print("💡 Models can be downloaded manually later")
        
        print("✅ OpenSora isolated environment setup completed!")
        
        return {
            "success": True,
            "env_info": {
                "status": "available",
                "env_path": env_path,
                "env_name": "opensora-env",
                "opensora_path": opensora_path,
                "models_path": models_path,
                "python_path": python_path
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def _create_opensora_isolated_wrapper(env_info, dtype, device, optimize):
    """Create wrapper that runs OpenSora in isolated environment."""
    import subprocess
    import os
    import tempfile
    import shutil
    
    class OpenSoraIsolatedWrapper:
        def __init__(self, env_info, dtype, device):
            self.env_info = env_info
            self.dtype = dtype
            self.device = device
            self.python_path = env_info.get("python_path")
            self.opensora_path = env_info.get("opensora_path")
            self.models_path = env_info.get("models_path")
            
        def __call__(self, prompt, num_frames=65, height=768, width=768, guidance_scale=7.0, num_inference_steps=50, **kwargs):
            """Generate video using isolated OpenSora environment."""
            print(f"🎬 Generating OpenSora v2 video (isolated):")
            print(f"   📝 Prompt: {prompt}")
            print(f"   📐 Resolution: {height}x{width}")
            print(f"   🎞️  Frames: {num_frames}")
            print(f"   🔒 Environment: {self.env_info['env_name']}")
            
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, "opensora_output")
                os.makedirs(output_path, exist_ok=True)
                
                # Determine config based on resolution
                if height <= 256 or width <= 256:
                    config_file = "configs/diffusion/inference/t2i2v_256px.py"
                else:
                    config_file = "configs/diffusion/inference/t2i2v_768px.py"
                
                config_path = os.path.join(self.opensora_path, config_file)
                
                # Prepare inference command
                inference_script = os.path.join(self.opensora_path, "scripts", "diffusion", "inference.py")
                
                cmd = [
                    self.python_path,
                    inference_script,
                    config_path,
                    "--prompt", prompt,
                    "--save-dir", output_path
                ]
                
                print(f"🚀 Running OpenSora in isolated environment...")
                
                # Run in isolated environment
                original_dir = os.getcwd()
                os.chdir(self.opensora_path)
                
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=600,  # 10 minute timeout
                    )
                    
                    if result.returncode != 0:
                        print(f"❌ OpenSora inference failed:")
                        print(f"   Error: {result.stderr}")
                        raise RuntimeError(f"Inference failed: {result.stderr}")
                    
                    print("✅ OpenSora inference completed")
                    
                    # Find generated video files
                    video_files = []
                    for root, dirs, files in os.walk(output_path):
                        for file in files:
                            if file.endswith(('.mp4', '.avi', '.mov')):
                                video_files.append(os.path.join(root, file))
                    
                    if not video_files:
                        raise RuntimeError("No video files generated")
                    
                    # Copy video to main directory
                    video_path = video_files[0]
                    final_output = f"opensora_output_{hash(prompt) % 10000}.mp4"
                    shutil.copy2(video_path, final_output)
                    
                    print(f"📹 Generated video copied to: {os.path.abspath(final_output)}")
                    
                    # Create result object
                    class OpenSoraResult:
                        def __init__(self, video_path):
                            self.video_path = video_path
                            self.frames = [[video_path]]  # Mimic diffusers format
                    
                    return OpenSoraResult(final_output)
                    
                finally:
                    os.chdir(original_dir)
    
    return OpenSoraIsolatedWrapper(env_info, dtype, device)


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
        "hunyuanvideo-community/HunyuanVideo": {
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


def install_opensora() -> Dict[str, Any]:
    """Install OpenSora 2.0 natively with proper setup.
    
    Returns:
        Installation result dictionary
    """
    import subprocess
    import os
    
    print("🔧 Installing OpenSora 2.0...")
    
    try:
        # Check if already installed
        try:
            import opensora
            print("✅ OpenSora already installed")
            return {"success": True, "message": "OpenSora already installed"}
        except ImportError:
            pass
        
        # Clone repository
        if not os.path.exists("Open-Sora"):
            print("📥 Cloning OpenSora repository...")
            result = subprocess.run([
                "git", "clone", "https://github.com/hpcaitech/Open-Sora.git"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Git clone failed: {result.stderr}")
        
        # Install OpenSora
        print("⚙️  Installing OpenSora package...")
        result = subprocess.run([
            "pip", "install", "-e", "./Open-Sora"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Installation failed: {result.stderr}")
        
        print("✅ OpenSora 2.0 installed successfully")
        return {
            "success": True,
            "message": "OpenSora 2.0 installed successfully",
            "next_steps": [
                "Download model weights with: huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir ./ckpts",
                "Test installation with: python Open-Sora/scripts/inference.py"
            ]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "manual_instructions": [
                "git clone https://github.com/hpcaitech/Open-Sora.git",
                "cd Open-Sora",
                "pip install -e .",
                "huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir ./ckpts"
            ]
        }


def download_opensora_models(model_path: str = "./ckpts") -> Dict[str, Any]:
    """Download OpenSora 2.0 model weights.
    
    Args:
        model_path: Path to download models
        
    Returns:
        Download result dictionary
    """
    import subprocess
    import os
    
    print(f"📥 Downloading OpenSora 2.0 models to {model_path}...")
    
    try:
        # Install huggingface-cli if needed
        try:
            subprocess.run(["huggingface-cli", "--help"], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚙️  Installing huggingface-cli...")
            subprocess.run(["pip", "install", "huggingface_hub[cli]"], 
                         capture_output=True, check=True)
        
        # Download models
        print("📥 Downloading model weights...")
        result = subprocess.run([
            "huggingface-cli", "download", 
            "hpcai-tech/Open-Sora-v2",
            "--local-dir", model_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Download failed: {result.stderr}")
        
        # Verify download
        if os.path.exists(model_path):
            files = os.listdir(model_path)
            print(f"✅ Downloaded {len(files)} files to {model_path}")
            return {
                "success": True,
                "model_path": model_path,
                "files_count": len(files)
            }
        else:
            raise RuntimeError("Download completed but model path not found")
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "manual_instructions": [
                "pip install huggingface_hub[cli]",
                f"huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir {model_path}"
            ]
        }


def setup_opensora_environment() -> Dict[str, Any]:
    """Complete OpenSora 2.0 setup including installation and model download.
    
    Returns:
        Setup result dictionary
    """
    print("🚀 Setting up OpenSora 2.0 environment...")
    
    results = {}
    
    # Install OpenSora
    install_result = install_opensora()
    results["installation"] = install_result
    
    if not install_result["success"]:
        return {
            "success": False,
            "error": "Installation failed",
            "results": results
        }
    
    # Download models
    download_result = download_opensora_models()
    results["download"] = download_result
    
    if not download_result["success"]:
        return {
            "success": False,
            "error": "Model download failed", 
            "results": results
        }
    
    print("✅ OpenSora 2.0 environment setup complete!")
    
    return {
        "success": True,
        "message": "OpenSora 2.0 ready to use",
        "results": results,
        "usage_example": [
            "cd Open-Sora",
            "python scripts/inference.py configs/inference/t2v.py --prompt 'A cat walking in a garden'"
        ]
    }


def check_opensora_installation() -> Dict[str, Any]:
    """Check OpenSora 2.0 installation status.
    
    Returns:
        Installation status dictionary
    """
    import os
    
    status = {
        "opensora_installed": False,
        "models_downloaded": False,
        "ready": False,
        "missing": []
    }
    
    # Check OpenSora installation
    try:
        import opensora
        status["opensora_installed"] = True
        print("✅ OpenSora package installed")
    except ImportError:
        status["missing"].append("OpenSora package")
        print("❌ OpenSora package not installed")
    
    # Check model files
    if os.path.exists("./ckpts") and os.listdir("./ckpts"):
        status["models_downloaded"] = True
        print("✅ Model files found")
    else:
        status["missing"].append("Model files")
        print("❌ Model files not found")
    
    # Check if repository exists
    if os.path.exists("Open-Sora"):
        status["repository_cloned"] = True
        print("✅ OpenSora repository found")
    else:
        status["missing"].append("OpenSora repository")
        print("❌ OpenSora repository not found")
    
    status["ready"] = (status["opensora_installed"] and 
                      status["models_downloaded"] and 
                      status.get("repository_cloned", False))
    
    if status["ready"]:
        print("🎉 OpenSora 2.0 is ready to use!")
    else:
        print(f"⚠️  Missing: {', '.join(status['missing'])}")
        print("💡 Run videogenbook.setup_opensora_environment() to complete setup")
    
    return status


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