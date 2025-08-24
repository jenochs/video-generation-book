#!/usr/bin/env python3
"""
GPU-Accelerated Video Processor
Leverages available GPU capabilities for video processing
"""

import os
import subprocess
import numpy as np
import cv2
import logging
from typing import Optional, Tuple, Dict
import time

class GPUAcceleratedProcessor:
    """Video processor with GPU acceleration support"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gpu_info = self._detect_gpu_capabilities()
        self._setup_opencv_acceleration()
        
    def _detect_gpu_capabilities(self) -> Dict:
        """Detect all available GPU capabilities"""
        
        info = {
            'has_nvidia_gpu': False,
            'cuda_available': False,
            'opencl_available': False,
            'nvenc_available': False,
            'nvdec_available': False,
            'pytorch_cuda': False,
            'tensorflow_gpu': False,
            'gpu_name': None,
            'gpu_memory': None,
            'cuda_version': None
        }
        
        # Check NVIDIA GPU
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(',')
                info['has_nvidia_gpu'] = True
                info['gpu_name'] = gpu_info[0].strip()
                info['gpu_memory'] = gpu_info[1].strip() if len(gpu_info) > 1 else None
                
                # Get CUDA version from nvidia-smi
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=2)
                if 'CUDA Version:' in result.stdout:
                    cuda_version = result.stdout.split('CUDA Version:')[1].split()[0]
                    info['cuda_version'] = cuda_version
                    info['cuda_available'] = True
        except:
            pass
        
        # Check OpenCL in OpenCV
        try:
            build_info = cv2.getBuildInformation()
            info['opencl_available'] = 'OpenCL' in build_info and 'YES' in build_info.split('OpenCL')[1][:100]
        except:
            pass
        
        # Check FFmpeg NVENC/NVDEC
        try:
            result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True, timeout=2)
            info['nvenc_available'] = 'h264_nvenc' in result.stdout or 'hevc_nvenc' in result.stdout
            
            result = subprocess.run(['ffmpeg', '-decoders'], capture_output=True, text=True, timeout=2)
            info['nvdec_available'] = 'h264_cuvid' in result.stdout or 'hevc_cuvid' in result.stdout
        except:
            pass
        
        # Check PyTorch CUDA
        try:
            import torch
            info['pytorch_cuda'] = torch.cuda.is_available()
        except:
            pass
        
        # Check TensorFlow GPU
        try:
            import tensorflow as tf
            info['tensorflow_gpu'] = len(tf.config.list_physical_devices('GPU')) > 0
        except:
            pass
        
        return info
    
    def _setup_opencv_acceleration(self):
        """Setup OpenCV acceleration (OpenCL)"""
        if self.gpu_info['opencl_available']:
            try:
                # Enable OpenCL acceleration in OpenCV
                cv2.ocl.setUseOpenCL(True)
                if cv2.ocl.useOpenCL():
                    self.logger.info("✅ OpenCL acceleration enabled in OpenCV")
                    
                    # Get OpenCL device info
                    platforms = cv2.ocl.getPlatfomsInfo()
                    if platforms:
                        self.logger.info(f"OpenCL platforms: {len(platforms)}")
            except Exception as e:
                self.logger.warning(f"Could not enable OpenCL: {e}")
    
    def get_gpu_status_string(self) -> str:
        """Get a formatted GPU status string"""
        if self.gpu_info['has_nvidia_gpu']:
            status = f"✅ {self.gpu_info['gpu_name']}"
            if self.gpu_info['gpu_memory']:
                status += f" ({self.gpu_info['gpu_memory']})"
            if self.gpu_info['cuda_version']:
                status += f" | CUDA {self.gpu_info['cuda_version']}"
            return status
        elif self.gpu_info['opencl_available']:
            return "⚠️ OpenCL only (No NVIDIA GPU)"
        else:
            return "❌ No GPU acceleration"
    
    def process_video_with_gpu(self, input_path: str, output_path: str, 
                              use_gpu: bool = True) -> bool:
        """Process video using GPU acceleration when available"""
        
        if not self.gpu_info['nvenc_available'] or not use_gpu:
            # Fallback to CPU processing
            return self._process_video_cpu(input_path, output_path)
        
        # Use NVENC for GPU-accelerated encoding
        cmd = [
            'ffmpeg', '-y',
            '-hwaccel', 'cuda',  # Use CUDA for decoding
            '-hwaccel_output_format', 'cuda',
            '-i', input_path,
            '-c:v', 'h264_nvenc',  # Use NVENC for encoding
            '-preset', 'p4',  # Balance between speed and quality
            '-rc', 'vbr',  # Variable bitrate
            '-cq', '23',  # Quality setting
            '-b:v', '5M',  # Target bitrate
            '-maxrate', '10M',
            '-bufsize', '20M',
            '-c:a', 'copy',  # Copy audio as-is
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info(f"✅ GPU-accelerated encoding completed")
                return True
            else:
                self.logger.warning(f"GPU encoding failed, falling back to CPU")
                return self._process_video_cpu(input_path, output_path)
        except Exception as e:
            self.logger.error(f"GPU processing error: {e}")
            return self._process_video_cpu(input_path, output_path)
    
    def _process_video_cpu(self, input_path: str, output_path: str) -> bool:
        """Fallback CPU processing"""
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'copy',
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def extract_frames_gpu(self, video_path: str, num_frames: int = 10) -> Optional[np.ndarray]:
        """Extract frames using GPU-accelerated decoding"""
        
        frames = []
        
        if self.gpu_info['nvdec_available']:
            # Use NVDEC for GPU-accelerated decoding
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        else:
            # Regular CPU decoding
            cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        
        return np.array(frames) if frames else None
    
    def benchmark_gpu_vs_cpu(self, video_path: str) -> Dict:
        """Benchmark GPU vs CPU processing"""
        
        results = {}
        
        # Test frame extraction
        if self.gpu_info['opencl_available']:
            # Test with OpenCL
            cv2.ocl.setUseOpenCL(True)
            start = time.time()
            frames = self.extract_frames_gpu(video_path, 30)
            results['opencl_time'] = time.time() - start
            
            # Test without OpenCL
            cv2.ocl.setUseOpenCL(False)
            start = time.time()
            frames = self.extract_frames_gpu(video_path, 30)
            results['cpu_time'] = time.time() - start
            
            # Re-enable OpenCL
            cv2.ocl.setUseOpenCL(True)
            
            results['speedup'] = results['cpu_time'] / results['opencl_time'] if results['opencl_time'] > 0 else 1.0
        
        return results


class GPUVideoPreprocessor:
    """Enhanced video preprocessor with GPU support"""
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256),
                 target_fps: int = 8,
                 max_frames: int = 64):
        self.target_size = target_size
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.gpu_processor = GPUAcceleratedProcessor()
        self.logger = logging.getLogger(__name__)
        
    def process_video(self, video_path: str, use_gpu: bool = True) -> Optional[np.ndarray]:
        """Process video with optional GPU acceleration"""
        
        # Try to use GPU-accelerated frame extraction
        if use_gpu and self.gpu_processor.gpu_info['opencl_available']:
            cv2.ocl.setUseOpenCL(True)
            self.logger.info("Using OpenCL acceleration for frame processing")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Failed to open video: {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate sampling rate
        sample_rate = max(1, int(fps / self.target_fps))
        
        frames = []
        frame_count = 0
        processed_count = 0
        
        while cap.isOpened() and processed_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # Resize frame (potentially GPU-accelerated with OpenCL)
                if self.gpu_processor.gpu_info['opencl_available']:
                    # Create UMat for GPU processing
                    gpu_frame = cv2.UMat(frame)
                    gpu_resized = cv2.resize(gpu_frame, self.target_size)
                    resized = gpu_resized.get()  # Transfer back to CPU
                else:
                    resized = cv2.resize(frame, self.target_size)
                
                frames.append(resized)
                processed_count += 1
            
            frame_count += 1
        
        cap.release()
        
        if not frames:
            self.logger.error("No frames extracted")
            return None
        
        # Convert to numpy array and normalize
        video_array = np.array(frames)
        
        # Normalize to [-1, 1] for neural networks
        video_array = video_array.astype(np.float32)
        video_array = (video_array - 127.5) / 127.5
        
        self.logger.info(f"Processed {len(frames)} frames with shape {video_array.shape}")
        return video_array


# Utility function to check GPU status
def get_gpu_info():
    """Get comprehensive GPU information"""
    processor = GPUAcceleratedProcessor()
    return processor.gpu_info


if __name__ == "__main__":
    # Test GPU detection
    print("="*70)
    print("GPU ACCELERATION STATUS")
    print("="*70)
    
    processor = GPUAcceleratedProcessor()
    
    print("\nGPU Capabilities:")
    for key, value in processor.gpu_info.items():
        status = "✅" if value else "❌"
        if key.startswith('gpu_'):
            if value:
                print(f"  {key}: {value}")
        else:
            print(f"  {status} {key}: {value}")
    
    print(f"\nOverall Status: {processor.get_gpu_status_string()}")
    
    # Test video processing if available
    test_video = "dataset/raw_videos/scene_changes_test.mp4"
    if os.path.exists(test_video):
        print(f"\nBenchmarking with: {test_video}")
        results = processor.benchmark_gpu_vs_cpu(test_video)
        if results:
            print(f"  CPU Time: {results.get('cpu_time', 0):.3f}s")
            print(f"  OpenCL Time: {results.get('opencl_time', 0):.3f}s")
            print(f"  Speedup: {results.get('speedup', 1.0):.2f}x")