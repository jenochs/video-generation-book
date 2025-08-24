#!/usr/bin/env python3
"""
Professional Video Processing Suite
Modern UI with latest Gradio features and optimized scene detection
"""

import gradio as gr
import os
import cv2
import numpy as np
import tempfile
import zipfile
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

# Suppress warnings for cleaner UI
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
import warnings
warnings.filterwarnings('ignore')

# Import optimized modules
from video_preprocessor import VideoPreprocessor
from video_quality_assessor import VideoQualityAssessor
from scene_extractor_optimized import SceneExtractorOptimized
from scene_detector_optimized import OptimizedSceneDetector
from video_data_pipeline import VideoDataPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Professional theme colors
THEME_CSS = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.contain {
    max-width: 1400px !important;
}
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
}
.secondary-btn {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    border: none !important;
}
.success-text {
    color: #10b981 !important;
    font-weight: 600 !important;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 12px;
    color: white;
    margin: 10px 0;
}
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin: 20px 0;
}
"""

class VideoProcessingSuite:
    """Professional video processing suite with modern UI"""
    
    def __init__(self):
        self.preprocessor = None
        self.assessor = VideoQualityAssessor()
        self.scene_detector = None
        self.pipeline = None
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict:
        """Get system capabilities"""
        import multiprocessing
        
        info = {
            'cpu_cores': multiprocessing.cpu_count(),
            'opencv_version': cv2.__version__,
            'has_opencl': self._check_opencl(),
            'has_cuda': self._check_cuda(),
        }
        
        # Check PySceneDetect version
        try:
            import scenedetect
            info['pyscenedetect_version'] = scenedetect.__version__
            info['has_new_api'] = hasattr(scenedetect, 'detect')
        except:
            info['pyscenedetect_version'] = 'Not installed'
            info['has_new_api'] = False
            
        return info
    
    def _check_opencl(self) -> bool:
        try:
            build_info = cv2.getBuildInformation()
            return 'OpenCL' in build_info and 'YES' in build_info.split('OpenCL')[1][:100]
        except:
            return False
    
    def _check_cuda(self) -> bool:
        try:
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except:
            return False
    
    def analyze_video(self, video_input) -> Tuple[str, Dict]:
        """Comprehensive video analysis"""
        if video_input is None:
            return "⚠️ Please upload a video first", {}
        
        try:
            video_path = video_input
            
            # Get video info
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return "❌ Failed to open video", {}
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frames / fps if fps > 0 else 0
            cap.release()
            
            # Get file info
            file_size = os.path.getsize(video_path) / (1024 * 1024)
            
            # Generate recommendations using optimized detector
            detector = OptimizedSceneDetector()
            analysis = detector.analyze_video(video_path)
            
            # Create professional report
            report = f"""
# 📊 Video Analysis Report

## File Information
- **File**: {os.path.basename(video_path)}
- **Size**: {file_size:.2f} MB
- **Codec**: H.264/MP4

## Video Properties
- **Resolution**: {width}x{height} ({self._get_resolution_name(width, height)})
- **Frame Rate**: {fps:.2f} FPS
- **Duration**: {duration:.2f} seconds ({duration/60:.2f} minutes)
- **Total Frames**: {frames:,}
- **Bitrate**: {(file_size * 8 * 1024) / duration:.0f} kbps

## AI Processing Recommendations
"""
            for rec in analysis.get('recommendations', []):
                report += f"- ✅ {rec}\n"
            
            report += f"""
## Optimal Settings
- **Scene Detector**: `{analysis.get('recommended_detector', 'content')}`
- **Threshold**: {analysis.get('recommended_threshold', 30)}
- **Min Scene Length**: {0.5 if duration < 60 else 2.0} seconds

## System Capabilities
- **CPU Cores**: {self.system_info['cpu_cores']}
- **OpenCL**: {'✅ Available' if self.system_info['has_opencl'] else '❌ Not available'}
- **CUDA**: {'✅ Available' if self.system_info['has_cuda'] else '❌ Not available'}
- **PySceneDetect**: v{self.system_info['pyscenedetect_version']} {'(New API ✅)' if self.system_info['has_new_api'] else ''}
"""
            
            # Return metrics for dashboard
            metrics = {
                'resolution': f"{width}x{height}",
                'fps': fps,
                'duration': duration,
                'size_mb': file_size,
                'quality': self._assess_quality(width, height, fps, file_size, duration)
            }
            
            return report, metrics
            
        except Exception as e:
            return f"❌ Error analyzing video: {str(e)}", {}
    
    def _get_resolution_name(self, width: int, height: int) -> str:
        """Get common name for resolution"""
        resolutions = {
            (1920, 1080): "Full HD",
            (1280, 720): "HD",
            (3840, 2160): "4K UHD",
            (2560, 1440): "2K QHD",
            (640, 480): "SD",
            (854, 480): "480p",
        }
        return resolutions.get((width, height), "Custom")
    
    def _assess_quality(self, width: int, height: int, fps: float, size_mb: float, duration: float) -> str:
        """Assess overall video quality"""
        score = 0
        
        # Resolution score
        if width >= 1920:
            score += 3
        elif width >= 1280:
            score += 2
        elif width >= 854:
            score += 1
            
        # FPS score
        if fps >= 30:
            score += 2
        elif fps >= 24:
            score += 1
            
        # Bitrate score
        bitrate = (size_mb * 8 * 1024) / duration if duration > 0 else 0
        if bitrate >= 5000:
            score += 2
        elif bitrate >= 2500:
            score += 1
            
        # Quality rating
        if score >= 6:
            return "⭐⭐⭐⭐⭐ Excellent"
        elif score >= 4:
            return "⭐⭐⭐⭐ Good"
        elif score >= 2:
            return "⭐⭐⭐ Fair"
        else:
            return "⭐⭐ Low"
    
    def detect_scenes(self, video_input, detector_type: str, threshold: float, 
                     min_scene_len: float) -> Tuple[str, List]:
        """Detect scenes using optimized detector"""
        if video_input is None:
            return "⚠️ Please upload a video first", []
        
        try:
            video_path = video_input
            
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
                return "⚠️ No scene changes detected. Video appears to be continuous.", []
            
            # Format scene data
            scene_data = []
            for i, (start_time, end_time) in enumerate(scenes):
                duration = end_time.get_seconds() - start_time.get_seconds()
                scene_data.append({
                    'Scene': f"Scene {i+1}",
                    'Start (s)': f"{start_time.get_seconds():.2f}",
                    'End (s)': f"{end_time.get_seconds():.2f}",
                    'Duration (s)': f"{duration:.2f}"
                })
            
            # Create report
            report = f"""
# 🎬 Scene Detection Results

## Summary
- **Scenes Detected**: {len(scenes)}
- **Detection Time**: {detect_time:.2f} seconds
- **Detector Used**: {detector_type}
- **Threshold**: {threshold}

## Performance
- **Processing Speed**: {detector.analyze_video(video_path).get('duration', 0) / detect_time:.1f}x realtime
- **Optimization**: {'✅ Multi-core enabled' if detector.use_multicore else '❌ Single-core'}
- **Hardware Acceleration**: {'✅ Available' if detector.has_opencl else '❌ Not available'}

## Scene List
Total of {len(scenes)} scenes detected. Use the table below to review scene boundaries.
"""
            
            return report, scene_data
            
        except Exception as e:
            return f"❌ Error detecting scenes: {str(e)}", []
    
    def extract_scenes(self, video_input, threshold: float, min_scene_len: int) -> Tuple[str, Optional[str]]:
        """Extract scenes to separate files"""
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
            
            # Extract scenes
            start_time = time.time()
            scenes = extractor.extract_scenes(video_path, scenes_dir)
            extract_time = time.time() - start_time
            
            if not scenes:
                return "⚠️ No scenes to extract", None
            
            # Create ZIP package
            zip_path = os.path.join(temp_dir, "scenes.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # Add scene videos
                for scene in scenes:
                    if os.path.exists(scene['output_path']):
                        arcname = os.path.basename(scene['output_path'])
                        zipf.write(scene['output_path'], arcname)
                
                # Add metadata
                metadata = {
                    'total_scenes': len(scenes),
                    'extraction_time': extract_time,
                    'scenes': scenes
                }
                zipf.writestr('metadata.json', json.dumps(metadata, indent=2))
            
            # Analyze scenes
            analysis = extractor.analyze_scenes(scenes)
            
            report = f"""
# ✂️ Scene Extraction Complete

## Extraction Summary
- **Scenes Extracted**: {len(scenes)}
- **Processing Time**: {extract_time:.2f} seconds
- **Output Format**: MP4 (H.264/MP4V)

## Scene Statistics
- **Average Duration**: {analysis['avg_scene_duration']:.2f} seconds
- **Shortest Scene**: {analysis['min_scene_duration']:.2f} seconds
- **Longest Scene**: {analysis['max_scene_duration']:.2f} seconds

## Scene Distribution
"""
            for category, count in analysis['scene_distribution'].items():
                if count > 0:
                    report += f"- **{category.replace('_', ' ').title()}**: {count} scenes\n"
            
            report += f"""
## Download Package
The ZIP file contains:
- {len(scenes)} scene video files
- metadata.json with scene information
- All scenes are ready for AI training

**File size**: {os.path.getsize(zip_path) / (1024*1024):.2f} MB
"""
            
            return report, zip_path
            
        except Exception as e:
            return f"❌ Error extracting scenes: {str(e)}", None
    
    def process_video(self, video_input, target_size: int, target_fps: int, 
                     max_frames: int) -> Tuple[str, Optional[np.ndarray]]:
        """Process video for AI training"""
        if video_input is None:
            return "⚠️ Please upload a video first", None
        
        try:
            # Initialize preprocessor
            self.preprocessor = VideoPreprocessor(
                target_size=(target_size, target_size),
                target_fps=target_fps,
                max_frames=max_frames
            )
            
            # Process video
            start_time = time.time()
            processed = self.preprocessor.process_video(video_input)
            process_time = time.time() - start_time
            
            if processed is None:
                return "❌ Failed to process video", None
            
            # Quality assessment
            quality = self.assessor.assess_video(processed)
            
            report = f"""
# 🤖 AI Preprocessing Complete

## Processing Summary
- **Output Shape**: {processed.shape}
- **Data Type**: {processed.dtype}
- **Memory Size**: {processed.nbytes / (1024*1024):.2f} MB
- **Processing Time**: {process_time:.2f} seconds

## Quality Metrics
- **Sharpness Score**: {quality.get('avg_sharpness', 0):.2f}
- **Motion Score**: {quality.get('avg_motion', 0):.4f}
- **Contrast Score**: {quality.get('avg_contrast', 0):.2f}
- **Overall Rating**: {quality.get('quality_rating', 'Unknown')}

## Normalization
- **Value Range**: [{processed.min():.2f}, {processed.max():.2f}]
- **Mean**: {processed.mean():.4f}
- **Std Dev**: {processed.std():.4f}

## Status
{'✅ **Ready for AI Training**' if quality.get('is_acceptable') else '⚠️ **Quality issues detected - review metrics**'}
"""
            
            return report, processed
            
        except Exception as e:
            return f"❌ Error processing video: {str(e)}", None


def create_professional_interface():
    """Create professional Gradio interface"""
    
    suite = VideoProcessingSuite()
    
    with gr.Blocks(
        title="Video Processing Suite",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=THEME_CSS
    ) as app:
        
        # Header
        gr.Markdown("""
        # 🎥 Professional Video Processing Suite
        ### Advanced AI-Ready Video Preprocessing with Optimized Scene Detection
        """)
        
        # System info banner
        with gr.Row():
            gr.Markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 15px; border-radius: 10px; color: white;">
                <b>System Status:</b> 
                CPU Cores: {suite.system_info['cpu_cores']} | 
                OpenCV: {suite.system_info['opencv_version']} | 
                PySceneDetect: {suite.system_info['pyscenedetect_version']} | 
                GPU: {'✅' if suite.system_info['has_cuda'] else '❌'}
            </div>
            """)
        
        # Main tabs
        with gr.Tabs() as tabs:
            
            # Tab 1: Video Analysis
            with gr.Tab("📊 Analysis", id=1):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input_1 = gr.Video(
                            label="Upload Video",
                            show_download_button=False,
                            height=300
                        )
                        analyze_btn = gr.Button(
                            "🔍 Analyze Video",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        analysis_output = gr.Markdown(
                            label="Analysis Report",
                            value="Upload a video to begin analysis..."
                        )
                
                # Metrics dashboard
                with gr.Row():
                    metric_resolution = gr.Textbox(label="Resolution", interactive=False)
                    metric_fps = gr.Number(label="FPS", interactive=False)
                    metric_duration = gr.Number(label="Duration (s)", interactive=False)
                    metric_quality = gr.Textbox(label="Quality Rating", interactive=False)
                
                analyze_btn.click(
                    fn=lambda v: (*suite.analyze_video(v), ),
                    inputs=[video_input_1],
                    outputs=[analysis_output, gr.State()],
                ).then(
                    fn=lambda v: suite.analyze_video(v)[1] if v else {},
                    inputs=[video_input_1],
                    outputs=[gr.State()],
                ).then(
                    fn=lambda metrics: (
                        metrics.get('resolution', ''),
                        metrics.get('fps', 0),
                        metrics.get('duration', 0),
                        metrics.get('quality', '')
                    ) if metrics else ('', 0, 0, ''),
                    inputs=[gr.State()],
                    outputs=[metric_resolution, metric_fps, metric_duration, metric_quality]
                )
            
            # Tab 2: Scene Detection
            with gr.Tab("🎬 Scene Detection", id=2):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input_2 = gr.Video(
                            label="Upload Video",
                            show_download_button=False,
                            height=300
                        )
                        
                        with gr.Group():
                            detector_type = gr.Dropdown(
                                choices=["Content", "Adaptive", "Threshold", "Histogram"],
                                value="Content",
                                label="Detector Type",
                                info="Content: General purpose | Adaptive: Long videos | Histogram: Presentations"
                            )
                            threshold = gr.Slider(
                                minimum=10, maximum=100, value=30, step=5,
                                label="Detection Threshold",
                                info="Lower = more sensitive"
                            )
                            min_scene_len = gr.Slider(
                                minimum=0.1, maximum=5.0, value=0.5, step=0.1,
                                label="Min Scene Length (seconds)",
                                info="Minimum duration for a scene"
                            )
                        
                        detect_btn = gr.Button(
                            "🔍 Detect Scenes",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        detection_output = gr.Markdown(
                            label="Detection Report",
                            value="Configure settings and click detect..."
                        )
                        scenes_table = gr.Dataframe(
                            label="Detected Scenes",
                            headers=["Scene", "Start (s)", "End (s)", "Duration (s)"],
                            interactive=False
                        )
                
                detect_btn.click(
                    fn=suite.detect_scenes,
                    inputs=[video_input_2, detector_type, threshold, min_scene_len],
                    outputs=[detection_output, scenes_table]
                )
            
            # Tab 3: Scene Extraction
            with gr.Tab("✂️ Scene Extraction", id=3):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input_3 = gr.Video(
                            label="Upload Video",
                            show_download_button=False,
                            height=300
                        )
                        
                        with gr.Group():
                            extract_threshold = gr.Slider(
                                minimum=10, maximum=100, value=30, step=5,
                                label="Scene Threshold"
                            )
                            extract_min_len = gr.Slider(
                                minimum=5, maximum=60, value=15, step=5,
                                label="Min Scene Length (frames)"
                            )
                        
                        extract_btn = gr.Button(
                            "✂️ Extract Scenes",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        extraction_output = gr.Markdown(
                            label="Extraction Report",
                            value="Configure settings and click extract..."
                        )
                        download_file = gr.File(
                            label="📦 Download Scene Package",
                            visible=True
                        )
                
                extract_btn.click(
                    fn=suite.extract_scenes,
                    inputs=[video_input_3, extract_threshold, extract_min_len],
                    outputs=[extraction_output, download_file]
                )
            
            # Tab 4: AI Preprocessing
            with gr.Tab("🤖 AI Preprocessing", id=4):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input_4 = gr.Video(
                            label="Upload Video",
                            show_download_button=False,
                            height=300
                        )
                        
                        with gr.Group():
                            target_size = gr.Slider(
                                minimum=128, maximum=512, value=256, step=64,
                                label="Target Size (pixels)",
                                info="Square resolution for AI model"
                            )
                            target_fps = gr.Slider(
                                minimum=1, maximum=30, value=8, step=1,
                                label="Target FPS",
                                info="Frame rate for processing"
                            )
                            max_frames = gr.Slider(
                                minimum=16, maximum=256, value=64, step=16,
                                label="Max Frames",
                                info="Maximum frames to extract"
                            )
                        
                        process_btn = gr.Button(
                            "🚀 Process for AI",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        processing_output = gr.Markdown(
                            label="Processing Report",
                            value="Configure settings and click process..."
                        )
                        
                        # Sample frames preview
                        with gr.Group():
                            gr.Markdown("### Preview Frames")
                            preview_frames = gr.Gallery(
                                label="Sample Frames",
                                show_label=False,
                                elem_id="gallery",
                                columns=4,
                                rows=2,
                                height=400
                            )
                
                def process_and_preview(video, size, fps, frames):
                    report, processed = suite.process_video(video, size, fps, frames)
                    
                    # Generate preview frames
                    previews = []
                    if processed is not None:
                        # Sample 8 frames evenly
                        indices = np.linspace(0, len(processed)-1, min(8, len(processed)), dtype=int)
                        for idx in indices:
                            frame = processed[idx]
                            # Denormalize for display
                            frame = ((frame + 1) * 127.5).astype(np.uint8)
                            previews.append(frame)
                    
                    return report, previews
                
                process_btn.click(
                    fn=process_and_preview,
                    inputs=[video_input_4, target_size, target_fps, max_frames],
                    outputs=[processing_output, preview_frames]
                )
            
            # Tab 5: Help & Documentation
            with gr.Tab("📚 Documentation", id=5):
                gr.Markdown("""
                ## Quick Start Guide
                
                ### 1. Video Analysis
                Upload any video to get comprehensive analysis including resolution, frame rate, duration, and AI processing recommendations.
                
                ### 2. Scene Detection
                **Detector Types:**
                - **Content**: Best for general videos with clear cuts
                - **Adaptive**: Optimal for long videos with varying content
                - **Threshold**: Detects fade in/out transitions
                - **Histogram**: Best for presentations and slide changes
                
                ### 3. Scene Extraction
                Automatically splits videos into individual scene clips, perfect for creating training datasets.
                
                ### 4. AI Preprocessing
                Prepares videos for neural network training with:
                - Resolution normalization
                - Frame rate adjustment
                - Pixel value normalization to [-1, 1]
                - Quality assessment
                
                ## System Requirements
                - Python 3.8+
                - OpenCV 4.5+
                - PySceneDetect 0.6+
                - 8GB RAM recommended
                - GPU acceleration supported (CUDA/OpenCL)
                
                ## Performance Tips
                - Enable multi-core processing for faster detection
                - Use hardware acceleration when available
                - For long videos (>10 min), use Adaptive detector
                - Lower threshold values increase sensitivity
                
                ## API Integration
                All functions are available via Python API:
                ```python
                from web_app_professional import VideoProcessingSuite
                suite = VideoProcessingSuite()
                analysis = suite.analyze_video("video.mp4")
                ```
                """)
        
        # Footer
        gr.Markdown("""
        ---
        <center>
        <small>
        Video Processing Suite v2.0 | Powered by OpenCV & PySceneDetect | 
        <a href="https://github.com/your-repo">GitHub</a> | 
        Built with ❤️ for AI Researchers
        </small>
        </center>
        """)
    
    return app


if __name__ == "__main__":
    app = create_professional_interface()
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )