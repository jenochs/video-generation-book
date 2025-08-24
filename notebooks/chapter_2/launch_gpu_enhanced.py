#!/usr/bin/env python3
"""
Launch GPU-Enhanced Video Processing Suite with Scene Thumbnails
"""

import os
import sys
import warnings
from pathlib import Path

# Suppress warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
warnings.filterwarnings('ignore')

def main():
    print("\n" + "="*70)
    print("🚀 GPU-ENHANCED VIDEO PROCESSING SUITE".center(70))
    print("="*70)
    
    print("""
    ✨ NEW FEATURES:
    ✅ Scene Thumbnail Generation
    ✅ Visual Scene Preview Gallery  
    ✅ GPU Performance Benchmarking
    ✅ NVIDIA RTX 3090 Optimization
    ✅ OpenCL & NVENC Acceleration
    ✅ Real-time Scene Visualization
    """)
    
    # Quick GPU check
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            gpu_name = result.stdout.strip()
            print(f"    🎮 GPU Detected: {gpu_name}")
    except:
        pass
    
    print("\n" + "="*70)
    print("🌐 Starting GPU-Enhanced Interface")
    print("📍 URL: http://localhost:7862")
    print("🛑 Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    try:
        # Import with suppressed warnings
        import cv2
        cv2.setLogLevel(0)
        
        # Enable OpenCL by default
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            print("✅ OpenCL acceleration enabled")
        
        from web_app_gpu_enhanced import create_gpu_enhanced_interface
        
        app = create_gpu_enhanced_interface()
        app.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=7862,
            show_error=True,
            quiet=False
        )
        
    except KeyboardInterrupt:
        print("\n\n✅ Shutting down GPU-enhanced interface...")
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("Install with: pip install gradio opencv-python pillow")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
