"""Command-line interface for videogenbook package."""

import click
import sys
import subprocess
from typing import Optional

# Use importlib.metadata instead of deprecated pkg_resources
try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

from .utils import check_gpu_memory, setup_environment
from .models import list_available_models
from . import __version__


@click.group()
@click.version_option(version=__version__, prog_name="videogenbook")
def main():
    """videogenbook - Companion CLI for 'Hands-On Video Generation with AI'
    
    A command-line interface for setting up, testing, and working with 
    2025's breakthrough video generation models.
    """
    pass


@main.command()
def info():
    """Display system information and package details."""
    click.echo(f"videogenbook v{__version__}")
    click.echo("=" * 50)
    
    # Package info
    click.echo("\n📦 Package Information:")
    click.echo(f"  Version: {__version__}")
    click.echo(f"  Author: Jenochs")
    click.echo(f"  Repository: https://github.com/jenochs/video-generation-book")
    
    # Python environment
    click.echo(f"\n🐍 Python Environment:")
    click.echo(f"  Python: {sys.version.split()[0]}")
    click.echo(f"  Platform: {sys.platform}")
    
    # GPU information
    click.echo(f"\n🖥️  Hardware Information:")
    try:
        gpu_info = check_gpu_memory()
        if gpu_info:
            click.echo(f"  GPU: {gpu_info['name']}")
            click.echo(f"  VRAM: {gpu_info['memory_total']:.1f} GB")
            click.echo(f"  Available: {gpu_info['memory_free']:.1f} GB")
        else:
            click.echo("  GPU: Not detected or unavailable")
    except Exception as e:
        click.echo(f"  GPU: Error checking ({str(e)})")
    
    # Supported models
    click.echo(f"\n🎬 Supported Models:")
    models = list_available_models()
    for model in models[:5]:  # Show first 5
        click.echo(f"  • {model}")
    if len(models) > 5:
        click.echo(f"  ... and {len(models) - 5} more")


@main.command()
@click.option('--gpu-check/--no-gpu-check', default=True, 
              help='Check GPU compatibility')
@click.option('--install-deps/--no-install-deps', default=False,
              help='Install missing dependencies')
def setup(gpu_check: bool, install_deps: bool):
    """Set up environment for video generation."""
    click.echo("🚀 Setting up videogenbook environment...")
    
    try:
        result = setup_environment(
            check_gpu=gpu_check,
            install_dependencies=install_deps
        )
        
        if result['success']:
            click.echo("✅ Environment setup completed successfully!")
            
            if result.get('gpu_available'):
                gpu_info = result['gpu_info']
                click.echo(f"🖥️  GPU detected: {gpu_info['name']} ({gpu_info['memory_total']:.1f} GB)")
                
                # Provide memory recommendations
                memory_gb = gpu_info['memory_total']
                if memory_gb >= 24:
                    click.echo("💪 Excellent! You can run all models at full quality.")
                elif memory_gb >= 16:
                    click.echo("👍 Good! You can run most models with some optimization.")
                elif memory_gb >= 8:
                    click.echo("⚠️  Limited. You'll need quantization for larger models.")
                else:
                    click.echo("❌ Insufficient GPU memory. Consider Google Colab Pro.")
            else:
                click.echo("⚠️  No GPU detected. Consider using Google Colab for acceleration.")
                
        else:
            click.echo("❌ Environment setup encountered issues:")
            for issue in result.get('issues', []):
                click.echo(f"  • {issue}")
                
    except Exception as e:
        click.echo(f"❌ Setup failed: {str(e)}")
        click.echo("💡 Try running: pip install --upgrade videogenbook")


@main.command()
@click.argument('model_name')
@click.option('--prompt', '-p', default="A cat walking in a garden",
              help='Text prompt for video generation')
@click.option('--duration', '-d', default=5.0, type=float,
              help='Video duration in seconds')
@click.option('--fps', default=24, type=int,
              help='Frames per second')
@click.option('--resolution', '-r', default=768, type=int,
              help='Video resolution (height and width)')
@click.option('--guidance-scale', default=8.0, type=float,
              help='Guidance scale for prompt adherence')
@click.option('--steps', default=50, type=int,
              help='Number of inference steps')
@click.option('--output', '-o', default="output.mp4",
              help='Output video filename')
def generate(model_name: str, prompt: str, duration: float, fps: int, resolution: int, guidance_scale: float, steps: int, output: str):
    """Generate a video using specified model."""
    click.echo(f"🎬 Generating video with {model_name}...")
    click.echo(f"📝 Prompt: {prompt}")
    click.echo(f"⏱️  Duration: {duration}s at {fps} FPS")
    
    try:
        from .generation import generate_video, VideoGenerationConfig
        
        # Enhanced prompt for better quality
        enhanced_prompt = f"high quality, detailed, cinematic, {prompt}, well lit, clear focus"
        
        config = VideoGenerationConfig(
            model_name=model_name,
            prompt=enhanced_prompt,
            duration=duration,
            fps=fps,
            resolution=resolution,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            output_path=output
        )
        
        with click.progressbar(length=100, label='Generating') as bar:
            def progress_callback(step: int, total: int):
                bar.update(int(100 * step / total) - bar.pos)
            
            result = generate_video(config, progress_callback=progress_callback)
        
        if result['success']:
            click.echo(f"✅ Video saved to: {output}")
            click.echo(f"📊 Generation time: {result['generation_time']:.1f}s")
            click.echo(f"💾 File size: {result['file_size_mb']:.1f} MB")
        else:
            click.echo(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
            
    except ImportError as e:
        click.echo(f"❌ Missing dependencies: {str(e)}")
        click.echo("💡 Try running: videogenbook setup --install-deps")
    except Exception as e:
        click.echo(f"❌ Generation failed: {str(e)}")


@main.command()
@click.option('--models', '-m', multiple=True,
              help='Specific models to benchmark (default: all)')
@click.option('--output', '-o', default="benchmark_results.json",
              help='Output file for results')
def benchmark(models: tuple, output: str):
    """Benchmark model performance on your hardware."""
    click.echo("🏁 Starting performance benchmark...")
    
    try:
        from .evaluation import benchmark_performance
        
        models_to_test = list(models) if models else None
        
        with click.progressbar(length=100, label='Benchmarking') as bar:
            def progress_callback(current: int, total: int):
                bar.update(int(100 * current / total) - bar.pos)
            
            results = benchmark_performance(
                models=models_to_test,
                progress_callback=progress_callback
            )
        
        # Save results
        import json
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
            
        click.echo(f"✅ Benchmark completed! Results saved to: {output}")
        
        # Display summary
        click.echo("\n📊 Performance Summary:")
        for model, metrics in results.items():
            click.echo(f"  {model}:")
            click.echo(f"    Speed: {metrics['speed_fps']:.1f} FPS")
            click.echo(f"    Memory: {metrics['peak_memory_gb']:.1f} GB")
            click.echo(f"    Quality: {metrics['quality_score']:.2f}")
            
    except Exception as e:
        click.echo(f"❌ Benchmark failed: {str(e)}")


@main.command()
def models():
    """List all available video generation models."""
    click.echo("🎬 Available Video Generation Models:")
    click.echo("=" * 50)
    
    try:
        available_models = list_available_models()
        
        categories = {
            "🚀 Working Models (HuggingFace Hub)": [
                "Lightricks/LTX-Video",
                "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
            ],
            "🌟 Open Source Leaders": [
                "hunyuanvideo-community/HunyuanVideo",
                "hpcai-tech/Open-Sora-v2"
            ],
            "🚀 Colab Optimized": [
                "hunyuanvideo-community/HunyuanVideo-Colab"
            ],
            "⏳ API-Only Models (No HuggingFace)": [
                "Google Veo 3 (API)",
                "Kling AI 2.0 (API)",
                "Runway Gen-4 (API)",
                "Pika 2.2 (API)"
            ]
        }
        
        for category, model_list in categories.items():
            click.echo(f"\n{category}")
            for model in model_list:
                if category == "⏳ API-Only Models (No HuggingFace)":
                    click.echo(f"  🌐 {model} (requires API access)")
                elif model in available_models:
                    click.echo(f"  ✅ {model}")
                else:
                    click.echo(f"  ⏳ {model} (setup required)")
                    
    except Exception as e:
        click.echo(f"❌ Error listing models: {str(e)}")


@main.command()
def jupyter():
    """Launch Jupyter Lab with videogenbook examples."""
    click.echo("🚀 Launching Jupyter Lab...")
    
    try:
        # Try to find notebooks directory
        import videogenbook
        import os
        package_dir = os.path.dirname(videogenbook.__file__)
        notebooks_path = os.path.join(os.path.dirname(package_dir), 'notebooks')
        
        subprocess.run([
            sys.executable, '-m', 'jupyter', 'lab', 
            '--notebook-dir', notebooks_path
        ], check=True)
        
    except FileNotFoundError:
        click.echo("❌ Jupyter Lab not found. Install with: pip install jupyterlab")
    except Exception as e:
        click.echo(f"❌ Failed to launch Jupyter: {str(e)}")
        click.echo("💡 Try running: jupyter lab notebooks/")


@main.command()
def colab_setup():
    """Setup optimal environment for Google Colab video generation."""
    click.echo("🚀 Setting up videogenbook for Google Colab...")
    
    try:
        from .models import setup_colab_environment, get_colab_optimized_config, check_gpu_memory
        
        # Setup environment
        setup_colab_environment()
        
        # Check GPU
        gpu_info = check_gpu_memory()
        if gpu_info:
            click.echo(f"🖥️  GPU detected: {gpu_info['name']}")
            click.echo(f"💾 VRAM: {gpu_info['memory_total']:.1f} GB total, {gpu_info['memory_free']:.1f} GB available")
            
            if "A100" in gpu_info['name']:
                click.echo("✅ A100 GPU detected - optimal for HunyuanVideo!")
                
                # Show recommended configuration
                config = get_colab_optimized_config("hunyuanvideo-community/HunyuanVideo", gpu_info['memory_free'])
                click.echo(f"\n🎬 Recommended HunyuanVideo settings:")
                click.echo(f"   Resolution: {config['width']}x{config['height']}")
                click.echo(f"   Frames: {config['num_frames']}")
                click.echo(f"   Steps: {config['num_inference_steps']}")
                
                click.echo(f"\n💡 Quick start command:")
                click.echo(f"   videogenbook generate 'hunyuanvideo-community/HunyuanVideo-Colab' --prompt 'your prompt'")
            else:
                click.echo("⚠️  For best results, upgrade to Colab Pro+ for A100 access")
        else:
            click.echo("❌ No GPU detected - video generation will be very slow")
            
        click.echo("\n✅ Colab setup completed!")
        click.echo("📚 Use the HunyuanVideo Colab notebook for guided tutorials")
        
    except Exception as e:
        click.echo(f"❌ Setup failed: {str(e)}")
        click.echo("💡 Try running: pip install --upgrade videogenbook")


@main.command()
@click.option('--chapter', '-c', type=int, help='Specific chapter to open')
def colab(chapter: Optional[int]):
    """Open chapter notebooks in Google Colab."""
    base_url = "https://colab.research.google.com/github/jenochs/video-generation-book/blob/main/notebooks"
    
    if chapter:
        chapter_map = {
            1: "01_foundations/chapter_01_foundations.ipynb",
            2: "02_data/chapter_02_data.ipynb", 
            3: "03_implementation/chapter_03_implementation.ipynb",
            4: "04_training/chapter_04_training.ipynb",
            5: "05_fine_tuning/chapter_05_fine_tuning.ipynb",
            6: "06_text_prompts/chapter_06_text_prompts.ipynb",
            7: "07_deployment/chapter_07_deployment.ipynb",
            8: "08_evaluation/chapter_08_evaluation.ipynb",
        }
        
        if chapter in chapter_map:
            url = f"{base_url}/{chapter_map[chapter]}"
            click.echo(f"🚀 Opening Chapter {chapter} in Colab:")
            click.echo(url)
        else:
            click.echo(f"❌ Chapter {chapter} not found. Available: 1-8")
    else:
        click.echo("🚀 Open any chapter in Google Colab:")
        click.echo(f"  Repository: {base_url}")
        click.echo("  Available chapters: 1-8")
        click.echo("  Example: videogenbook colab --chapter 1")


if __name__ == "__main__":
    main()