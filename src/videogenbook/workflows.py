"""High-level workflows and convenience functions for videogenbook."""

import os
from typing import Dict, List, Optional, Any, Union
import warnings

from .models import get_model_recommendations, check_model_compatibility
from .generation import generate_video, VideoGenerationConfig
from .evaluation import evaluate_quality, EvaluationMetrics
from .utils import setup_environment, check_gpu_memory


def quick_start(
    prompt: str = "A cat walking in a garden",
    output_path: str = "my_first_video.mp4",
    auto_setup: bool = True
) -> Dict[str, Any]:
    """Quick start workflow for first-time users.
    
    Args:
        prompt: Text prompt for video generation
        output_path: Where to save the generated video
        auto_setup: Whether to automatically set up the environment
        
    Returns:
        Dictionary with generation results and recommendations
    """
    result = {
        'success': False,
        'video_path': None,
        'recommendations': [],
        'next_steps': []
    }
    
    try:
        # Setup environment if requested
        if auto_setup:
            print("🚀 Setting up videogenbook environment...")
            setup_result = setup_environment(check_gpu=True)
            
            if not setup_result['success']:
                result['error'] = "Environment setup failed"
                result['recommendations'] = [
                    "Try: pip install --upgrade videogenbook",
                    "For GPU support: install CUDA toolkit",
                    "Alternative: use Google Colab Pro"
                ]
                return result
        
        # Check hardware and recommend model
        gpu_info = check_gpu_memory()
        if gpu_info:
            memory_gb = gpu_info['memory_total']
            if memory_gb >= 12:
                recommended_model = "Lightricks/LTX-Video"
                result['recommendations'].append(f"✨ Great! Your GPU ({gpu_info['name']}) can run high-quality models")
            elif memory_gb >= 6:
                recommended_model = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
                result['recommendations'].append(f"👍 Your GPU ({gpu_info['name']}) works well with efficient models")
            else:
                recommended_model = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
                result['recommendations'].append("⚠️ Limited GPU memory - using most efficient model")
        else:
            recommended_model = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" 
            result['recommendations'].append("❌ No GPU detected - consider Google Colab for better performance")
        
        print(f"🎬 Generating video with {recommended_model}...")
        print(f"📝 Prompt: '{prompt}'")
        
        # Generate video with optimal settings
        config = VideoGenerationConfig(
            model_name=recommended_model,
            prompt=prompt,
            output_path=output_path,
            duration=5.0,
            resolution=512,
            num_inference_steps=25,  # Fast for first try
            precision="fp16"
        )
        
        generation_result = generate_video(config)
        
        if generation_result['success']:
            result['success'] = True
            result['video_path'] = output_path
            result['generation_time'] = generation_result['generation_time']
            result['model_used'] = recommended_model
            
            print(f"✅ Video generated successfully!")
            print(f"📁 Saved to: {output_path}")
            print(f"⏱️ Generation time: {generation_result['generation_time']:.1f}s")
            
            # Next steps recommendations
            result['next_steps'] = [
                "Try different prompts to explore the model's capabilities",
                "Experiment with longer durations (up to 15 seconds)",
                "Increase resolution for higher quality (if GPU allows)",
                "Compare different models using videogenbook.compare_models()",
                "Explore Chapter 2: Understanding and Preparing Video Data"
            ]
            
        else:
            result['error'] = generation_result.get('error', 'Unknown error')
            result['recommendations'] = [
                "Check if you have enough GPU memory",
                "Try: videogenbook setup --install-deps",
                "Use Google Colab if local GPU is insufficient"
            ]
    
    except Exception as e:
        result['error'] = str(e)
        result['recommendations'] = [
            "Ensure all dependencies are installed",
            "Check your internet connection for model downloads",
            "Try running: videogenbook info"
        ]
    
    return result


def text_to_video(
    prompts: Union[str, List[str]],
    model_name: Optional[str] = None,
    duration: float = 5.0,
    quality: str = "balanced",
    output_dir: str = "text_to_video_outputs"
) -> Dict[str, Any]:
    """Generate videos from text prompts with intelligent defaults.
    
    Args:
        prompts: Single prompt or list of prompts
        model_name: Model to use (auto-selected if None)
        duration: Video duration in seconds
        quality: Quality preset ("fast", "balanced", "high")
        output_dir: Output directory for videos
        
    Returns:
        Results dictionary with generation details
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Auto-select model if not specified
    if model_name is None:
        gpu_info = check_gpu_memory()
        if gpu_info and gpu_info['memory_total'] >= 12:
            model_name = "Lightricks/LTX-Video"
        else:
            model_name = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    
    # Quality presets
    quality_settings = {
        "fast": {
            "resolution": 384,
            "num_inference_steps": 20,
            "guidance_scale": 6.0,
            "precision": "fp16"
        },
        "balanced": {
            "resolution": 512,
            "num_inference_steps": 35,
            "guidance_scale": 7.5,
            "precision": "fp16"
        },
        "high": {
            "resolution": 768,
            "num_inference_steps": 50,
            "guidance_scale": 8.5,
            "precision": "fp16"
        }
    }
    
    settings = quality_settings.get(quality, quality_settings["balanced"])
    
    results = {
        'successful_generations': [],
        'failed_generations': [],
        'total_time': 0.0,
        'average_time_per_video': 0.0,
        'model_used': model_name,
        'quality_preset': quality
    }
    
    for i, prompt in enumerate(prompts):
        output_path = os.path.join(output_dir, f"video_{i:03d}.mp4")
        
        config = VideoGenerationConfig(
            model_name=model_name,
            prompt=prompt,
            duration=duration,
            output_path=output_path,
            **settings
        )
        
        generation_result = generate_video(config)
        
        if generation_result['success']:
            results['successful_generations'].append({
                'prompt': prompt,
                'output_path': output_path,
                'generation_time': generation_result['generation_time'],
                'file_size_mb': generation_result['file_size_mb']
            })
            results['total_time'] += generation_result['generation_time']
        else:
            results['failed_generations'].append({
                'prompt': prompt,
                'error': generation_result.get('error', 'Unknown error')
            })
    
    if results['successful_generations']:
        results['average_time_per_video'] = results['total_time'] / len(results['successful_generations'])
    
    return results


def image_to_video(
    image_path: str,
    prompt: Optional[str] = None,
    model_name: str = "runway/gen-4",
    duration: float = 3.0,
    output_path: str = "image_to_video.mp4"
) -> Dict[str, Any]:
    """Generate video from image with motion (placeholder implementation).
    
    Args:
        image_path: Path to input image
        prompt: Optional text prompt for motion guidance
        model_name: Model to use for generation
        duration: Video duration in seconds
        output_path: Output video path
        
    Returns:
        Generation results
    """
    warnings.warn("Image-to-video functionality requires model-specific implementations")
    
    # This is a placeholder - actual implementation would depend on model capabilities
    if not os.path.exists(image_path):
        return {
            'success': False,
            'error': f"Image not found: {image_path}"
        }
    
    # For now, use text-to-video with image context in prompt
    if prompt is None:
        prompt = "Animate the scene with natural motion"
    else:
        prompt = f"Based on the provided image: {prompt}"
    
    config = VideoGenerationConfig(
        model_name=model_name,
        prompt=prompt,
        duration=duration,
        output_path=output_path
    )
    
    return generate_video(config)


def video_to_video(
    input_video_path: str,
    prompt: str,
    model_name: str = "runway/gen-4",
    strength: float = 0.7,
    output_path: str = "video_to_video.mp4"
) -> Dict[str, Any]:
    """Transform existing video based on text prompt (placeholder implementation).
    
    Args:
        input_video_path: Path to input video
        prompt: Text prompt for transformation
        model_name: Model to use
        strength: Transformation strength (0.0 = no change, 1.0 = complete transformation)
        output_path: Output video path
        
    Returns:
        Generation results
    """
    warnings.warn("Video-to-video functionality requires model-specific implementations")
    
    if not os.path.exists(input_video_path):
        return {
            'success': False,
            'error': f"Input video not found: {input_video_path}"
        }
    
    # Placeholder implementation
    config = VideoGenerationConfig(
        model_name=model_name,
        prompt=f"Transform this video: {prompt}",
        output_path=output_path
    )
    
    return generate_video(config)


def batch_experiment(
    base_prompt: str,
    variations: List[str],
    models: Optional[List[str]] = None,
    output_dir: str = "batch_experiments"
) -> Dict[str, Any]:
    """Run batch experiments with prompt variations across multiple models.
    
    Args:
        base_prompt: Base text prompt
        variations: List of prompt variations to try
        models: Models to test (default: open source models)
        output_dir: Output directory for results
        
    Returns:
        Comprehensive results with quality analysis
    """
    if models is None:
        models = ["tencent/HunyuanVideo", "hpcai-tech/Open-Sora-v2"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'base_prompt': base_prompt,
        'variations_tested': variations,
        'models_tested': models,
        'experiment_results': {},
        'best_combinations': [],
        'quality_summary': {}
    }
    
    for model in models:
        model_dir = os.path.join(output_dir, model.replace('/', '_'))
        os.makedirs(model_dir, exist_ok=True)
        
        model_results = []
        
        for i, variation in enumerate(variations):
            full_prompt = f"{base_prompt} {variation}".strip()
            output_path = os.path.join(model_dir, f"variation_{i:02d}.mp4")
            
            config = VideoGenerationConfig(
                model_name=model,
                prompt=full_prompt,
                output_path=output_path,
                duration=4.0,  # Shorter for batch processing
                resolution=512
            )
            
            generation_result = generate_video(config)
            
            if generation_result['success']:
                # Evaluate quality
                metrics = evaluate_quality(output_path, full_prompt)
                
                result_entry = {
                    'variation': variation,
                    'full_prompt': full_prompt,
                    'output_path': output_path,
                    'generation_time': generation_result['generation_time'],
                    'quality_metrics': metrics,
                    'overall_score': metrics.overall_quality
                }
                
                model_results.append(result_entry)
                
                # Track best combinations
                results['best_combinations'].append({
                    'model': model,
                    'variation': variation,
                    'score': metrics.overall_quality,
                    'path': output_path
                })
        
        results['experiment_results'][model] = model_results
    
    # Sort best combinations by score
    results['best_combinations'].sort(key=lambda x: x['score'], reverse=True)
    
    # Generate quality summary
    for model in models:
        model_scores = [r['overall_score'] for r in results['experiment_results'][model]]
        if model_scores:
            results['quality_summary'][model] = {
                'average_quality': sum(model_scores) / len(model_scores),
                'best_quality': max(model_scores),
                'consistency': 1.0 - (max(model_scores) - min(model_scores)),  # Lower variance = higher consistency
                'success_rate': len(model_scores) / len(variations)
            }
    
    return results


def model_comparison_workflow(
    prompt: str,
    models: Optional[List[str]] = None,
    output_dir: str = "model_comparison"
) -> Dict[str, Any]:
    """Compare multiple models on the same prompt with detailed analysis.
    
    Args:
        prompt: Text prompt to use for all models
        models: List of models to compare (default: available open source)
        output_dir: Output directory for comparison
        
    Returns:
        Detailed comparison results
    """
    if models is None:
        models = get_model_recommendations("general")
    
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_results = {
        'prompt': prompt,
        'models_compared': models,
        'individual_results': {},
        'rankings': {},
        'recommendations': {}
    }
    
    all_metrics = {}
    
    for model in models:
        output_path = os.path.join(output_dir, f"{model.replace('/', '_')}.mp4")
        
        # Check compatibility first
        compatibility = check_model_compatibility(model)
        if not compatibility['compatible']:
            comparison_results['individual_results'][model] = {
                'success': False,
                'error': compatibility['error'],
                'recommendations': compatibility.get('recommendations', [])
            }
            continue
        
        config = VideoGenerationConfig(
            model_name=model,
            prompt=prompt,
            output_path=output_path,
            duration=5.0,
            resolution=512
        )
        
        generation_result = generate_video(config)
        
        if generation_result['success']:
            metrics = evaluate_quality(output_path, prompt)
            all_metrics[model] = metrics
            
            comparison_results['individual_results'][model] = {
                'success': True,
                'output_path': output_path,
                'generation_time': generation_result['generation_time'],
                'file_size_mb': generation_result['file_size_mb'],
                'metrics': metrics,
                'overall_score': metrics.overall_quality
            }
        else:
            comparison_results['individual_results'][model] = {
                'success': False,
                'error': generation_result.get('error'),
            }
    
    # Generate rankings
    successful_models = {
        model: result for model, result in comparison_results['individual_results'].items()
        if result.get('success', False)
    }
    
    if successful_models:
        # Rank by overall quality
        quality_ranking = sorted(
            successful_models.items(),
            key=lambda x: x[1]['overall_score'],
            reverse=True
        )
        
        # Rank by speed
        speed_ranking = sorted(
            successful_models.items(),
            key=lambda x: x[1]['generation_time']
        )
        
        comparison_results['rankings'] = {
            'by_quality': [model for model, _ in quality_ranking],
            'by_speed': [model for model, _ in speed_ranking],
            'best_overall': quality_ranking[0][0] if quality_ranking else None,
            'fastest': speed_ranking[0][0] if speed_ranking else None
        }
        
        # Generate recommendations
        comparison_results['recommendations'] = {
            'for_quality': quality_ranking[0][0] if quality_ranking else None,
            'for_speed': speed_ranking[0][0] if speed_ranking else None,
            'for_beginners': 'hpcai-tech/Open-Sora-v2' if 'hpcai-tech/Open-Sora-v2' in successful_models else None,
            'summary': f"Best quality: {quality_ranking[0][0] if quality_ranking else 'N/A'}, "
                      f"Fastest: {speed_ranking[0][0] if speed_ranking else 'N/A'}"
        }
    
    return comparison_results