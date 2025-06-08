"""Evaluation and benchmarking functionality for videogenbook."""

import time
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
import json
import os

from .models import list_available_models, load_model, get_model_info
from .generation import generate_video, VideoGenerationConfig
from .utils import check_gpu_memory


@dataclass
class EvaluationMetrics:
    """Container for video evaluation metrics."""
    
    # Quality metrics
    visual_quality: float = 0.0
    temporal_consistency: float = 0.0
    prompt_alignment: float = 0.0
    overall_quality: float = 0.0
    
    # Performance metrics
    generation_time: float = 0.0
    memory_usage_gb: float = 0.0
    speed_fps: float = 0.0
    
    # Technical metrics
    resolution: Tuple[int, int] = (512, 512)
    actual_fps: int = 24
    file_size_mb: float = 0.0
    
    # Arena ELO style metrics
    elo_score: Optional[float] = None
    win_rate: Optional[float] = None
    human_preference: Optional[float] = None


def evaluate_quality(
    video_path: str,
    prompt: str,
    reference_video: Optional[str] = None
) -> EvaluationMetrics:
    """Evaluate video quality using multiple metrics.
    
    Args:
        video_path: Path to generated video
        prompt: Original text prompt
        reference_video: Optional reference video for comparison
        
    Returns:
        EvaluationMetrics object with computed scores
    """
    metrics = EvaluationMetrics()
    
    try:
        # Load video
        frames = load_video_frames(video_path)
        metrics.resolution = (frames.shape[2], frames.shape[1])  # (W, H)
        metrics.file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        # Visual quality assessment
        metrics.visual_quality = assess_visual_quality(frames)
        
        # Temporal consistency
        metrics.temporal_consistency = assess_temporal_consistency(frames)
        
        # Prompt alignment (requires additional models)
        metrics.prompt_alignment = assess_prompt_alignment(frames, prompt)
        
        # Overall quality (weighted combination)
        metrics.overall_quality = (
            0.4 * metrics.visual_quality +
            0.3 * metrics.temporal_consistency + 
            0.3 * metrics.prompt_alignment
        )
        
        # Technical metrics
        metrics.actual_fps = estimate_fps(video_path)
        
    except Exception as e:
        print(f"Error evaluating video: {e}")
        
    return metrics


def assess_visual_quality(frames: np.ndarray) -> float:
    """Assess visual quality of video frames.
    
    Args:
        frames: Video frames as numpy array (T, H, W, C)
        
    Returns:
        Visual quality score (0-1)
    """
    try:
        # Brightness and contrast analysis
        brightness = np.mean(frames)
        contrast = np.std(frames)
        
        # Normalized brightness score (optimal around 0.3-0.7)
        brightness_score = 1.0 - abs(brightness / 255.0 - 0.5) * 2
        
        # Contrast score (higher contrast generally better)
        contrast_score = min(contrast / 50.0, 1.0)
        
        # Sharpness estimation using gradient magnitude
        grad_x = np.abs(np.gradient(frames, axis=3))
        grad_y = np.abs(np.gradient(frames, axis=2))
        sharpness = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        sharpness_score = min(sharpness / 20.0, 1.0)
        
        # Color diversity
        color_std = np.std(frames, axis=(0, 1, 2))
        color_diversity = np.mean(color_std) / 128.0
        color_score = min(color_diversity, 1.0)
        
        # Weighted combination
        quality_score = (
            0.3 * brightness_score +
            0.3 * contrast_score +
            0.25 * sharpness_score +
            0.15 * color_score
        )
        
        return np.clip(quality_score, 0.0, 1.0)
        
    except Exception:
        return 0.5  # Default score on error


def assess_temporal_consistency(frames: np.ndarray) -> float:
    """Assess temporal consistency across video frames.
    
    Args:
        frames: Video frames as numpy array (T, H, W, C)
        
    Returns:
        Temporal consistency score (0-1)
    """
    try:
        if len(frames) < 2:
            return 1.0
        
        # Frame-to-frame differences
        frame_diffs = []
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(frames[i] - frames[i-1]))
            frame_diffs.append(diff)
        
        # Consistency based on variance in frame differences
        diff_variance = np.var(frame_diffs)
        consistency_score = 1.0 / (1.0 + diff_variance / 100.0)
        
        # Optical flow consistency (simplified)
        motion_consistency = assess_motion_consistency(frames)
        
        # Combined score
        temporal_score = 0.7 * consistency_score + 0.3 * motion_consistency
        
        return np.clip(temporal_score, 0.0, 1.0)
        
    except Exception:
        return 0.5


def assess_motion_consistency(frames: np.ndarray) -> float:
    """Assess motion consistency using simple optical flow.
    
    Args:
        frames: Video frames as numpy array
        
    Returns:
        Motion consistency score (0-1)
    """
    try:
        # Simple motion estimation using frame differences
        motion_vectors = []
        
        for i in range(1, min(len(frames), 10)):  # Sample first 10 frames
            diff = frames[i] - frames[i-1]
            motion_magnitude = np.mean(np.abs(diff))
            motion_vectors.append(motion_magnitude)
        
        if not motion_vectors:
            return 1.0
        
        # Consistency based on motion smoothness
        motion_variance = np.var(motion_vectors)
        motion_score = 1.0 / (1.0 + motion_variance / 50.0)
        
        return np.clip(motion_score, 0.0, 1.0)
        
    except Exception:
        return 0.5


def assess_prompt_alignment(frames: np.ndarray, prompt: str) -> float:
    """Assess alignment between video content and text prompt.
    
    Args:
        frames: Video frames as numpy array
        prompt: Original text prompt
        
    Returns:
        Prompt alignment score (0-1)
    """
    try:
        # This is a simplified implementation
        # In practice, you'd use CLIP or similar models
        
        # Basic keyword matching
        prompt_lower = prompt.lower()
        keywords = extract_keywords(prompt_lower)
        
        # Placeholder scoring based on visual complexity
        # Higher complexity might indicate more detailed content
        visual_complexity = np.std(frames) / 128.0
        
        # Simple heuristic scoring
        keyword_score = min(len(keywords) / 5.0, 1.0)
        complexity_score = min(visual_complexity, 1.0)
        
        alignment_score = 0.6 * keyword_score + 0.4 * complexity_score
        
        return np.clip(alignment_score, 0.0, 1.0)
        
    except Exception:
        return 0.5


def extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from text prompt."""
    # Simple keyword extraction
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    words = text.replace(',', ' ').replace('.', ' ').split()
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    return keywords


def load_video_frames(video_path: str) -> np.ndarray:
    """Load video frames from file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Numpy array of frames (T, H, W, C)
    """
    try:
        import imageio
        
        # Read video
        reader = imageio.get_reader(video_path)
        frames = []
        
        for frame in reader:
            frames.append(frame)
        
        reader.close()
        
        return np.array(frames)
        
    except Exception as e:
        raise ValueError(f"Could not load video {video_path}: {e}")


def estimate_fps(video_path: str) -> int:
    """Estimate FPS of video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Estimated FPS
    """
    try:
        import imageio
        
        reader = imageio.get_reader(video_path)
        fps = reader.get_meta_data()['fps']
        reader.close()
        
        return int(fps)
        
    except Exception:
        return 24  # Default FPS


def benchmark_performance(
    models: Optional[List[str]] = None,
    test_prompts: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, Dict[str, Any]]:
    """Benchmark multiple models on performance metrics.
    
    Args:
        models: List of model names to benchmark (None for all)
        test_prompts: Test prompts to use (None for default set)
        progress_callback: Optional progress callback
        
    Returns:
        Dictionary with benchmark results for each model
    """
    if models is None:
        models = [
            "tencent/hunyuan-video",
            "pku-yuan/opensora-2.0"
        ]  # Only test open source models by default
    
    if test_prompts is None:
        test_prompts = [
            "A cat walking in a garden",
            "Ocean waves crashing on rocks",
            "A person riding a bicycle"
        ]
    
    results = {}
    total_tests = len(models) * len(test_prompts)
    current_test = 0
    
    for model_name in models:
        model_results = {
            'generation_times': [],
            'memory_usage': [],
            'quality_scores': [],
            'errors': []
        }
        
        try:
            # Load model once
            model = load_model(model_name, enable_optimization=True)
            
            for prompt in test_prompts:
                current_test += 1
                if progress_callback:
                    progress_callback(current_test, total_tests)
                
                try:
                    # Benchmark generation
                    start_memory = get_gpu_memory_usage()
                    start_time = time.time()
                    
                    config = VideoGenerationConfig(
                        model_name=model_name,
                        prompt=prompt,
                        duration=3.0,  # Short for benchmarking
                        resolution=512,
                        num_inference_steps=25,  # Fast settings
                        output_path=f"benchmark_{model_name.replace('/', '_')}_{current_test}.mp4"
                    )
                    
                    result = generate_video(config)
                    generation_time = time.time() - start_time
                    peak_memory = get_gpu_memory_usage()
                    
                    if result['success']:
                        # Evaluate quality
                        metrics = evaluate_quality(result['output_path'], prompt)
                        
                        model_results['generation_times'].append(generation_time)
                        model_results['memory_usage'].append(peak_memory - start_memory)
                        model_results['quality_scores'].append(metrics.overall_quality)
                        
                        # Clean up
                        if os.path.exists(result['output_path']):
                            os.remove(result['output_path'])
                    else:
                        model_results['errors'].append(result.get('error', 'Unknown error'))
                        
                except Exception as e:
                    model_results['errors'].append(str(e))
            
        except Exception as e:
            model_results['errors'].append(f"Failed to load model: {str(e)}")
        
        # Compute aggregate statistics
        if model_results['generation_times']:
            results[model_name] = {
                'avg_generation_time': np.mean(model_results['generation_times']),
                'avg_memory_usage_gb': np.mean(model_results['memory_usage']),
                'avg_quality_score': np.mean(model_results['quality_scores']),
                'speed_fps': 3.0 * 24 / np.mean(model_results['generation_times']),  # 3 second videos at 24fps
                'success_rate': len(model_results['generation_times']) / len(test_prompts),
                'errors': model_results['errors']
            }
        else:
            results[model_name] = {
                'avg_generation_time': float('inf'),
                'avg_memory_usage_gb': 0.0,
                'avg_quality_score': 0.0,
                'speed_fps': 0.0,
                'success_rate': 0.0,
                'errors': model_results['errors']
            }
    
    return results


def compare_models(
    model_names: List[str],
    prompt: str,
    output_dir: str = "model_comparison"
) -> Dict[str, EvaluationMetrics]:
    """Compare multiple models on the same prompt.
    
    Args:
        model_names: List of models to compare
        prompt: Text prompt to use for all models
        output_dir: Directory to save comparison outputs
        
    Returns:
        Dictionary mapping model names to evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    comparison_results = {}
    
    for model_name in model_names:
        try:
            output_path = os.path.join(output_dir, f"{model_name.replace('/', '_')}.mp4")
            
            config = VideoGenerationConfig(
                model_name=model_name,
                prompt=prompt,
                output_path=output_path,
                duration=5.0,
                resolution=512
            )
            
            result = generate_video(config)
            
            if result['success']:
                metrics = evaluate_quality(output_path, prompt)
                metrics.generation_time = result['generation_time']
                comparison_results[model_name] = metrics
            else:
                print(f"Failed to generate with {model_name}: {result.get('error')}")
                
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
    
    return comparison_results


def get_gpu_memory_usage() -> float:
    """Get current GPU memory usage in GB.
    
    Returns:
        Memory usage in GB
    """
    try:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0
    except Exception:
        return 0.0


def arena_elo_evaluation(
    model_results: Dict[str, Any],
    human_preferences: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """Calculate Arena ELO scores for models.
    
    Args:
        model_results: Results from model comparison
        human_preferences: Optional human preference data
        
    Returns:
        Dictionary mapping model names to ELO scores
    """
    # Simplified ELO calculation based on quality metrics
    models = list(model_results.keys())
    elo_scores = {model: 1000.0 for model in models}  # Starting ELO
    
    # Calculate relative performance
    for i, model_a in enumerate(models):
        for j, model_b in enumerate(models):
            if i >= j:
                continue
                
            # Compare quality scores
            score_a = model_results[model_a].get('avg_quality_score', 0)
            score_b = model_results[model_b].get('avg_quality_score', 0)
            
            # Simple win/loss based on quality
            if score_a > score_b:
                winner, loser = model_a, model_b
            else:
                winner, loser = model_b, model_a
            
            # Update ELO scores (simplified)
            k_factor = 32  # ELO adjustment factor
            expected_a = 1 / (1 + 10**((elo_scores[loser] - elo_scores[winner]) / 400))
            
            if winner == model_a:
                elo_scores[model_a] += k_factor * (1 - expected_a)
                elo_scores[model_b] += k_factor * (0 - (1 - expected_a))
            else:
                elo_scores[model_b] += k_factor * (1 - expected_a)
                elo_scores[model_a] += k_factor * (0 - (1 - expected_a))
    
    return elo_scores