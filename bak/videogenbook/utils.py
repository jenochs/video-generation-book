"""Utility functions for videogenbook package."""

import os
import sys
import torch
import platform
import subprocess
from typing import Dict, List, Optional, Any
import warnings


def get_device() -> str:
    """Get the best available device for video generation.
    
    Automatically selects the best device based on hardware availability.
    
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'  # Apple Silicon Mac
    else:
        return 'cpu'


def check_gpu_memory() -> Optional[Dict[str, Any]]:
    """Check GPU availability and memory information.
    
    Returns:
        Dictionary with GPU information or None if no GPU available
    """
    if not torch.cuda.is_available():
        return None
    
    try:
        device = torch.cuda.current_device()
        gpu_props = torch.cuda.get_device_properties(device)
        
        # Get memory info
        memory_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
        memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)  # GB  
        memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
        memory_free = memory_total - memory_allocated
        
        return {
            'name': gpu_props.name,
            'device_id': device,
            'memory_total': memory_total,
            'memory_allocated': memory_allocated,
            'memory_reserved': memory_reserved,
            'memory_free': memory_free,
            'compute_capability': f"{gpu_props.major}.{gpu_props.minor}",
            'multiprocessor_count': gpu_props.multi_processor_count,
        }
    except Exception as e:
        warnings.warn(f"Error checking GPU memory: {e}")
        return None


def optimize_memory() -> Dict[str, bool]:
    """Apply memory optimization settings for PyTorch.
    
    Returns:
        Dictionary indicating which optimizations were applied
    """
    optimizations = {}
    
    # Enable TensorFloat-32 for faster training/inference on Ampere GPUs
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        optimizations['tf32'] = True
    except Exception:
        optimizations['tf32'] = False
    
    # Enable cuDNN autotuner
    try:
        torch.backends.cudnn.benchmark = True
        optimizations['cudnn_benchmark'] = True
    except Exception:
        optimizations['cudnn_benchmark'] = False
    
    # Set memory allocation strategy
    try:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        optimizations['memory_allocation'] = True
    except Exception:
        optimizations['memory_allocation'] = False
    
    return optimizations


def setup_environment(
    check_gpu: bool = True,
    install_dependencies: bool = False
) -> Dict[str, Any]:
    """Set up the videogenbook environment.
    
    Args:
        check_gpu: Whether to check GPU availability
        install_dependencies: Whether to install missing dependencies
        
    Returns:
        Dictionary with setup results and information
    """
    result = {
        'success': False,
        'issues': [],
        'gpu_available': False,
        'optimizations': {},
    }
    
    # Check Python version
    if sys.version_info < (3, 9):
        result['issues'].append(f"Python 3.9+ required, found {sys.version}")
        return result
    
    # Check GPU if requested
    if check_gpu:
        gpu_info = check_gpu_memory()
        if gpu_info:
            result['gpu_available'] = True
            result['gpu_info'] = gpu_info
            
            # Apply memory optimizations
            result['optimizations'] = optimize_memory()
        else:
            result['issues'].append("No GPU detected or CUDA unavailable")
    
    # Check critical dependencies
    critical_deps = ['torch', 'diffusers', 'transformers']
    missing_deps = []
    
    for dep in critical_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)
    
    if missing_deps:
        if install_dependencies:
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', 
                    '--upgrade', 'videogenbook[performance]'
                ])
                result['issues'].append("Dependencies installed successfully")
            except subprocess.CalledProcessError:
                result['issues'].append(f"Failed to install dependencies: {missing_deps}")
                return result
        else:
            result['issues'].append(f"Missing dependencies: {missing_deps}")
            return result
    
    # Environment variable setup
    setup_env_vars()
    
    result['success'] = len([issue for issue in result['issues'] if 'Failed' in issue]) == 0
    return result


def setup_env_vars():
    """Set up environment variables for optimal performance."""
    env_vars = {
        # HuggingFace cache
        'HF_HOME': os.path.expanduser('~/.cache/huggingface'),
        'TRANSFORMERS_CACHE': os.path.expanduser('~/.cache/huggingface/transformers'),
        'DIFFUSERS_CACHE': os.path.expanduser('~/.cache/huggingface/diffusers'),
        
        # Memory optimization
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
        
        # Disable some warnings for cleaner output
        'TOKENIZERS_PARALLELISM': 'false',
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information.
    
    Returns:
        Dictionary with system details
    """
    info = {
        'platform': platform.platform(),
        'python_version': sys.version,
        'torch_version': torch.__version__ if 'torch' in sys.modules else 'Not installed',
        'cuda_available': torch.cuda.is_available() if 'torch' in sys.modules else False,
    }
    
    # Add GPU info if available
    gpu_info = check_gpu_memory()
    if gpu_info:
        info['gpu'] = gpu_info
    
    return info


def download_sample_data(dataset: str = "basic") -> str:
    """Download sample data for examples and testing.
    
    Args:
        dataset: Type of sample data to download ("basic", "advanced", "benchmark")
        
    Returns:
        Path to downloaded data directory
    """
    import tempfile
    import urllib.request
    import zipfile
    
    # Create cache directory
    cache_dir = os.path.expanduser('~/.cache/videogenbook/data')
    os.makedirs(cache_dir, exist_ok=True)
    
    dataset_urls = {
        'basic': 'https://github.com/jenochs/video-generation-book/releases/download/v0.1.0/basic_samples.zip',
        'advanced': 'https://github.com/jenochs/video-generation-book/releases/download/v0.1.0/advanced_samples.zip',
        'benchmark': 'https://github.com/jenochs/video-generation-book/releases/download/v0.1.0/benchmark_data.zip',
    }
    
    if dataset not in dataset_urls:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(dataset_urls.keys())}")
    
    data_dir = os.path.join(cache_dir, dataset)
    
    # Check if already downloaded
    if os.path.exists(data_dir) and os.listdir(data_dir):
        return data_dir
    
    # Download and extract
    url = dataset_urls[dataset]
    
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
        try:
            urllib.request.urlretrieve(url, tmp_file.name)
            
            with zipfile.ZipFile(tmp_file.name, 'r') as zip_ref:
                zip_ref.extractall(cache_dir)
                
        finally:
            os.unlink(tmp_file.name)
    
    return data_dir


def format_size(size_bytes: int) -> str:
    """Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def validate_model_name(model_name: str) -> bool:
    """Validate if a model name is supported.
    
    Args:
        model_name: Name of the model to validate
        
    Returns:
        True if model is supported
    """
    from . import SUPPORTED_MODELS
    return model_name in SUPPORTED_MODELS


def get_hardware_recommendations(model_name: str) -> Dict[str, Any]:
    """Get hardware recommendations for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with hardware recommendations
    """
    # Model-specific requirements (updated with correct HuggingFace URLs)
    model_requirements = {
        'Lightricks/LTX-Video': {
            'min_vram_gb': 8,
            'recommended_vram_gb': 12,
            'min_ram_gb': 16,
            'recommended_ram_gb': 24,
            'notes': 'High-quality DiT model with efficient implementation'
        },
        'Wan-AI/Wan2.1-T2V-1.3B-Diffusers': {
            'min_vram_gb': 6,
            'recommended_vram_gb': 8,
            'min_ram_gb': 12,
            'recommended_ram_gb': 16,
            'notes': 'Memory-efficient model ideal for consumer hardware'
        },
        'tencent/HunyuanVideo': {
            'min_vram_gb': 8,
            'recommended_vram_gb': 16,
            'min_ram_gb': 16,
            'recommended_ram_gb': 32,
            'notes': 'Open source 13B model with flexible memory options'
        },
        'hpcai-tech/Open-Sora-v2': {
            'min_vram_gb': 6,
            'recommended_vram_gb': 12,
            'min_ram_gb': 12,
            'recommended_ram_gb': 24,
            'notes': 'Optimized for efficiency and accessibility'
        },
    }
    
    return model_requirements.get(model_name, {
        'min_vram_gb': 8,
        'recommended_vram_gb': 16,
        'min_ram_gb': 16,
        'recommended_ram_gb': 32,
        'notes': 'Standard video generation requirements'
    })