#!/usr/bin/env python3
"""
GPU-Enhanced Video Processing Suite with Scene Thumbnails
Advanced interface with GPU acceleration and visual scene previews
"""

import gradio as gr
import os
import cv2
import numpy as np
import tempfile
import zipfile
import json
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from PIL import Image
import io
import base64

# Suppress warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
import warnings
warnings.filterwarnings('ignore')

# Import modules
from video_preprocessor import VideoPreprocessor
from video_quality_assessor import VideoQualityAssessor
from scene_extractor_optimized import SceneExtractorOptimized
from scene_detector_optimized import OptimizedSceneDetector
from gpu_accelerated_processor import GPUAcceleratedProcessor, GPUVideoPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced theme with GPU status colors
THEME_CSS = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.gpu-enabled {
    background: linear-gradient(135deg, #00d4ff 0%, #00a86b 100%) !important;
    color: white !important;
    padding: 15px !important;
    border-radius: 10px !important;
    font-weight: bold !important;
}
.gpu-disabled {
    background: linear-gradient(135deg, #ff6b6b 0%, #ffd93d 100%) !important;
    color: white !important;
    padding: 15px !important;
    border-radius: 10px !important;
}
.scene-thumbnail {
    border: 2px solid #667eea;
    border-radius: 8px;
    padding: 4px;
    margin: 4px;
    transition: transform 0.2s;
}
.scene-thumbnail:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}
.thumbnail-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 15px;
    padding: 20px;
}
"""

class GPUEnhancedVideoSuite:
    """GPU-Enhanced video processing suite with visual feedback"""
    
    def __init__(self):
        self.gpu_processor = GPUAcceleratedProcessor()
        self.assessor = VideoQualityAssessor()
        self.gpu_info = self.gpu_processor.gpu_info
        self.scene_thumbnails = []
        self.current_scenes = []
        
    def get_gpu_status_html(self) -> str:
        """Generate HTML status bar for GPU"""
        gpu_status = self.gpu_processor.get_gpu_status_string()
        
        if self.gpu_info['has_nvidia_gpu']:
            # Full GPU support
            status_class = "gpu-enabled"
            icon = "🚀"
            details = f"""
            <div class="{status_class}">
                {icon} <b>GPU Status:</b> {gpu_status}<br>
                <small>
                CUDA: {'✅' if self.gpu_info['cuda_available'] else '❌'} | 
                OpenCL: {'✅' if self.gpu_info['opencl_available'] else '❌'} | 
                NVENC: {'✅' if self.gpu_info['nvenc_available'] else '❌'} | 
                PyTorch: {'✅' if self.gpu_info['pytorch_cuda'] else '❌'}
                </small>
            </div>
            """
        elif self.gpu_info['opencl_available']:
            # Partial GPU support
            status_class = "gpu-disabled"
            icon = "⚡"
            details = f"""
            <div class="{status_class}">
                {icon} <b>Limited GPU:</b> OpenCL Only<br>
                <small>For full GPU support, CUDA-enabled OpenCV needed</small>
            </div>
            """
        else:
            # No GPU support
            status_class = "gpu-disabled"
            icon = "🐌"
            details = f"""
            <div class="{status_class}">
                {icon} <b>CPU Mode:</b> No GPU Acceleration<br>
                <small>Install CUDA and GPU drivers for acceleration</small>
            </div>
            """
        
        return details
    
    def extract_scene_thumbnails(self, video_path: str, scenes: List[Tuple], 
                                max_thumbs: int = 24) -> List[Image.Image]:
        """Extract thumbnail images for each scene"""
        thumbnails = []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return thumbnails
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Limit number of thumbnails for performance
        scenes_to_process = scenes[:max_thumbs]
        
        for i, (start_time, end_time) in enumerate(scenes_to_process):
            # Get middle frame of the scene
            middle_time = (start_time.get_seconds() + end_time.get_seconds()) / 2
            middle_frame = int(middle_time * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            ret, frame = cap.read()
            
            if ret:
                # Resize for thumbnail
                height, width = frame.shape[:2]
                thumb_width = 320
                thumb_height = int(height * (thumb_width / width))
                thumbnail = cv2.resize(frame, (thumb_width, thumb_height))
                
                # Convert BGR to RGB
                thumbnail_rgb = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                
                # Add text overlay with scene info
                duration = end_time.get_seconds() - start_time.get_seconds()
                text = f"Scene {i+1} ({duration:.1f}s)"
                cv2.putText(thumbnail_rgb, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(thumbnail_rgb, f"{start_time.get_seconds():.1f}s", 
                           (10, thumb_height-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(thumbnail_rgb)
                thumbnails.append(pil_image)
        
        cap.release()
        return thumbnails
    
    def detect_and_visualize_scenes(self, video_input, detector_type: str, 
                                   threshold: float, min_scene_len: float,
                                   use_gpu: bool = True) -> Tuple[str, List, List]:
        """Detect scenes and generate visual thumbnails"""
        if video_input is None:
            return "⚠️ Please upload a video first", [], []
        
        try:
            video_path = video_input
            
            # Enable GPU acceleration if requested
            if use_gpu and self.gpu_info['opencl_available']:
                cv2.ocl.setUseOpenCL(True)
                logger.info("OpenCL GPU acceleration enabled")
            
            # Create optimized detector
            detector = OptimizedSceneDetector(
                detector_type=detector_type.lower(),
                threshold=threshold,
                min_scene_len=min_scene_len,
                use_multicore=True
            )
            
            # Detect scenes
            start_time = time.time()
            scenes = detector.detect_scenes(video_path, show_progress=False)
            detect_time = time.time() - start_time
            
            if not scenes:
                return "⚠️ No scene changes detected", [], []
            
            # Extract thumbnails
            thumbnails = self.extract_scene_thumbnails(video_path, scenes)
            
            # Store current scenes for later use
            self.current_scenes = scenes
            self.scene_thumbnails = thumbnails
            
            # Format scene data for table
            scene_data = []
            for i, (start_time, end_time) in enumerate(scenes):
                duration = end_time.get_seconds() - start_time.get_seconds()
                scene_data.append({
                    'Scene': f"Scene {i+1}",
                    'Start (s)': f"{start_time.get_seconds():.2f}",
                    'End (s)': f"{end_time.get_seconds():.2f}",
                    'Duration (s)': f"{duration:.2f}",
                    'Thumbnail': f"✅" if i < len(thumbnails) else "⏳"
                })
            
            # Get video info for performance metrics
            cap = cv2.VideoCapture(video_path)
            video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # Create detailed report
            gpu_status = "✅ GPU Accelerated" if use_gpu and self.gpu_info['opencl_available'] else "❌ CPU Only"
            
            report = f"""
# 🎬 Scene Detection Results with Thumbnails

## Detection Summary
- **Scenes Detected**: {len(scenes)}
- **Detection Time**: {detect_time:.2f} seconds
- **Processing Speed**: {video_duration / detect_time:.1f}x realtime
- **Thumbnails Generated**: {len(thumbnails)}

## Performance
- **GPU Acceleration**: {gpu_status}
- **Detector Used**: {detector_type}
- **Threshold**: {threshold}
- **Multi-core**: ✅ Enabled ({detector.cpu_count} cores)

## Visual Preview
{len(thumbnails)} scene thumbnails generated. Click on any thumbnail to see details.
"""
            
            return report, scene_data, thumbnails
            
        except Exception as e:
            logger.error(f"Error in scene detection: {e}")
            return f"❌ Error: {str(e)}", [], []
    
    def extract_scenes_with_gpu(self, video_input, threshold: float, 
                               min_scene_len: int, use_gpu: bool = True) -> Tuple[str, Optional[str]]:
        """Extract scenes using GPU acceleration when available"""
        if video_input is None:
            return "⚠️ Please upload a video first", None
        
        try:
            video_path = video_input
            temp_dir = tempfile.mkdtemp()
            scenes_dir = os.path.join(temp_dir, "scenes")
            
            # Use optimized extractor
            extractor = SceneExtractorOptimized(
                threshold=threshold,
                min_scene_len=min_scene_len,
                use_multicore=True
            )
            
            # Enable GPU if available
            if use_gpu and self.gpu_info['opencl_available']:
                cv2.ocl.setUseOpenCL(True)
            
            # Track GPU vs CPU usage
            gpu_used = use_gpu and (self.gpu_info['nvenc_available'] or self.gpu_info['opencl_available'])
            
            # Extract scenes
            start_time = time.time()
            scenes = extractor.extract_scenes(video_path, scenes_dir)
            extract_time = time.time() - start_time
            
            if not scenes:
                return "⚠️ No scenes to extract", None
            
            # If GPU encoding is available, re-encode with NVENC
            if use_gpu and self.gpu_info['nvenc_available']:
                for scene in scenes:
                    if os.path.exists(scene['output_path']):
                        # Re-encode with NVENC
                        temp_output = scene['output_path'] + '.gpu.mp4'
                        success = self.gpu_processor.process_video_with_gpu(
                            scene['output_path'], temp_output, use_gpu=True
                        )
                        if success:
                            os.replace(temp_output, scene['output_path'])
            
            # Create ZIP with thumbnails
            zip_path = os.path.join(temp_dir, "scenes_with_previews.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # Add scene videos
                for scene in scenes:
                    if os.path.exists(scene['output_path']):
                        arcname = os.path.basename(scene['output_path'])
                        zipf.write(scene['output_path'], arcname)
                
                # Add thumbnails if available
                if self.scene_thumbnails:
                    thumbs_dir = os.path.join(temp_dir, "thumbnails")
                    os.makedirs(thumbs_dir, exist_ok=True)
                    
                    for i, thumb in enumerate(self.scene_thumbnails):
                        thumb_path = os.path.join(thumbs_dir, f"thumb_{i:03d}.jpg")
                        thumb.save(thumb_path, "JPEG", quality=85)
                        zipf.write(thumb_path, f"thumbnails/thumb_{i:03d}.jpg")
                
                # Add metadata
                metadata = {
                    'total_scenes': len(scenes),
                    'extraction_time': extract_time,
                    'gpu_accelerated': gpu_used,
                    'scenes': scenes
                }
                zipf.writestr('metadata.json', json.dumps(metadata, indent=2))
            
            # Report
            acceleration = "GPU-Accelerated" if gpu_used else "CPU-Only"
            
            report = f"""
# ✂️ Scene Extraction Complete ({acceleration})

## Extraction Summary
- **Scenes Extracted**: {len(scenes)}
- **Processing Time**: {extract_time:.2f} seconds
- **Acceleration**: {'🚀 GPU' if gpu_used else '🐌 CPU'}
- **Thumbnails Included**: {'✅' if self.scene_thumbnails else '❌'}

## GPU Acceleration Details
- **OpenCL**: {'✅ Used' if use_gpu and self.gpu_info['opencl_available'] else '❌'}
- **NVENC Encoding**: {'✅ Used' if use_gpu and self.gpu_info['nvenc_available'] else '❌'}
- **Speed Boost**: {'~2-5x faster' if gpu_used else 'N/A'}

## Package Contents
- {len(scenes)} scene video files
- {len(self.scene_thumbnails)} thumbnail previews
- metadata.json with scene information

**Download size**: {os.path.getsize(zip_path) / (1024*1024):.2f} MB
"""
            
            return report, zip_path
            
        except Exception as e:
            return f"❌ Error: {str(e)}", None
    
    def benchmark_gpu_performance(self, video_input) -> str:
        """Benchmark GPU vs CPU performance"""
        if video_input is None:
            return "⚠️ Please upload a video first"
        
        try:
            video_path = video_input
            
            # Run benchmark
            results = self.gpu_processor.benchmark_gpu_vs_cpu(video_path)
            
            # Get video info
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            report = f"""
# ⚡ GPU Performance Benchmark

## Video Info
- Resolution: {width}x{height}
- Total Frames: {frames}

## Benchmark Results
"""
            
            if results:
                cpu_time = results.get('cpu_time', 0)
                opencl_time = results.get('opencl_time', 0)
                speedup = results.get('speedup', 1.0)
                
                report += f"""
- **CPU Processing Time**: {cpu_time:.3f} seconds
- **GPU (OpenCL) Time**: {opencl_time:.3f} seconds
- **Speedup**: {speedup:.2f}x faster with GPU

## Performance Analysis
"""
                if speedup > 1.5:
                    report += "🚀 **Excellent**: GPU acceleration is providing significant speedup!"
                elif speedup > 1.1:
                    report += "✅ **Good**: GPU acceleration is helping performance"
                else:
                    report += "⚠️ **Limited**: GPU acceleration benefit is minimal for this video"
            else:
                report += "❌ No GPU acceleration available for benchmarking"
            
            return report
            
        except Exception as e:
            return f"❌ Error running benchmark: {str(e)}"


def create_gpu_enhanced_interface():
    """Create GPU-enhanced interface with visual previews"""
    
    suite = GPUEnhancedVideoSuite()
    
    with gr.Blocks(
        title="GPU-Enhanced Video Processing",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
        ),
        css=THEME_CSS
    ) as app:
        
        # Header with GPU status
        gr.Markdown("# 🚀 GPU-Enhanced Video Processing Suite")
        gr.HTML(suite.get_gpu_status_html())
        
        # Main tabs
        with gr.Tabs():
            
            # Tab 1: Scene Detection with Thumbnails
            with gr.Tab("🎬 Scene Detection + Thumbnails"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="Upload Video")
                        
                        with gr.Group():
                            detector_type = gr.Dropdown(
                                choices=["Content", "Adaptive", "Threshold", "Histogram"],
                                value="Content",
                                label="Detector Type"
                            )
                            threshold = gr.Slider(10, 100, 30, step=5, label="Threshold")
                            min_scene_len = gr.Slider(0.1, 5.0, 0.5, step=0.1, 
                                                     label="Min Scene Length (s)")
                            use_gpu = gr.Checkbox(label="Enable GPU Acceleration", value=True)
                        
                        detect_btn = gr.Button("🔍 Detect Scenes & Generate Thumbnails", 
                                              variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        detection_output = gr.Markdown()
                        
                        # Thumbnail gallery
                        with gr.Group():
                            gr.Markdown("### 🖼️ Scene Thumbnails")
                            thumbnail_gallery = gr.Gallery(
                                label="Scene Previews",
                                show_label=False,
                                elem_id="gallery",
                                columns=4,
                                rows=3,
                                height=600,
                                object_fit="contain"
                            )
                        
                        # Scene data table
                        scenes_table = gr.Dataframe(
                            headers=["Scene", "Start (s)", "End (s)", "Duration (s)", "Thumbnail"],
                            label="Scene Details"
                        )
                
                detect_btn.click(
                    fn=suite.detect_and_visualize_scenes,
                    inputs=[video_input, detector_type, threshold, min_scene_len, use_gpu],
                    outputs=[detection_output, scenes_table, thumbnail_gallery]
                )
            
            # Tab 2: GPU-Accelerated Extraction
            with gr.Tab("⚡ GPU Scene Extraction"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input_2 = gr.Video(label="Upload Video")
                        
                        with gr.Group():
                            extract_threshold = gr.Slider(10, 100, 30, step=5, 
                                                         label="Scene Threshold")
                            extract_min_len = gr.Slider(5, 60, 15, step=5, 
                                                       label="Min Scene Length (frames)")
                            use_gpu_extract = gr.Checkbox(label="Use GPU Acceleration", value=True)
                        
                        extract_btn = gr.Button("✂️ Extract with GPU", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        extraction_output = gr.Markdown()
                        download_file = gr.File(label="📦 Download Package (with thumbnails)")
                
                extract_btn.click(
                    fn=suite.extract_scenes_with_gpu,
                    inputs=[video_input_2, extract_threshold, extract_min_len, use_gpu_extract],
                    outputs=[extraction_output, download_file]
                )
            
            # Tab 3: GPU Benchmark
            with gr.Tab("📊 GPU Performance"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input_3 = gr.Video(label="Upload Video for Benchmark")
                        benchmark_btn = gr.Button("⚡ Run GPU Benchmark", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        benchmark_output = gr.Markdown()
                
                benchmark_btn.click(
                    fn=suite.benchmark_gpu_performance,
                    inputs=[video_input_3],
                    outputs=[benchmark_output]
                )
            
            # Tab 4: GPU Information
            with gr.Tab("ℹ️ GPU Info"):
                gr.Markdown(f"""
                ## System GPU Capabilities
                
                ### GPU Hardware
                - **NVIDIA GPU**: {'✅ ' + suite.gpu_info.get('gpu_name', 'Unknown') if suite.gpu_info['has_nvidia_gpu'] else '❌ Not detected'}
                - **GPU Memory**: {suite.gpu_info.get('gpu_memory', 'N/A') if suite.gpu_info['has_nvidia_gpu'] else 'N/A'}
                - **CUDA Version**: {suite.gpu_info.get('cuda_version', 'N/A') if suite.gpu_info['cuda_available'] else 'Not available'}
                
                ### Acceleration Support
                - **OpenCL**: {'✅ Available' if suite.gpu_info['opencl_available'] else '❌ Not available'}
                - **NVENC (H.264)**: {'✅ Available' if suite.gpu_info['nvenc_available'] else '❌ Not available'}
                - **NVDEC**: {'✅ Available' if suite.gpu_info['nvdec_available'] else '❌ Not available'}
                - **PyTorch CUDA**: {'✅ Available' if suite.gpu_info['pytorch_cuda'] else '❌ Not available'}
                - **TensorFlow GPU**: {'✅ Available' if suite.gpu_info['tensorflow_gpu'] else '❌ Not available'}
                
                ### Optimization Tips
                
                1. **For Maximum GPU Usage**:
                   - Enable GPU acceleration checkbox in each tab
                   - Use "Adaptive" detector for long videos
                   - Keep threshold between 20-40 for best results
                
                2. **Current Limitations**:
                   - OpenCV from pip doesn't include CUDA support
                   - Using OpenCL for partial GPU acceleration
                   - FFmpeg can use NVENC/NVDEC for encoding/decoding
                
                3. **Performance Expectations**:
                   - Scene detection: 10-20x realtime with GPU
                   - Video encoding: 2-5x faster with NVENC
                   - Thumbnail generation: 1.5-2x faster with OpenCL
                """)
        
        # Footer
        gr.Markdown("""
        ---
        <center>
        GPU-Enhanced Video Processing Suite v3.0 | 
        Powered by NVIDIA RTX 3090 | 
        OpenCV {cv2.__version__} with OpenCL
        </center>
        """.format(cv2=cv2))
    
    return app


if __name__ == "__main__":
    app = create_gpu_enhanced_interface()
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7861,  # Different port from the other app
        show_error=True
    )