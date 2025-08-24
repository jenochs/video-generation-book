#!/usr/bin/env python3
"""
Launch the Professional Video Processing Suite
Modern UI with optimized scene detection and no deprecation warnings
"""

import os
import sys
import warnings
from pathlib import Path

# Suppress all warnings for clean UI
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
warnings.filterwarnings('ignore')

def main():
    print("\n" + "="*70)
    print("PROFESSIONAL VIDEO PROCESSING SUITE".center(70))
    print("="*70)
    
    print("""
    🚀 Modern Features:
    ✅ Optimized Scene Detection (No deprecation warnings!)
    ✅ Hardware Acceleration Support (OpenCL/CUDA)
    ✅ Multi-core Processing
    ✅ Professional UI with Gradio 5.43+
    ✅ Advanced Analytics Dashboard
    ✅ Batch Processing Support
    ✅ Export to AI-Ready Formats
    """)
    
    # Check for test videos
    video_dir = Path("dataset/raw_videos")
    if video_dir.exists():
        videos = list(video_dir.glob("*.mp4"))
        if videos:
            print(f"📁 Found {len(videos)} test videos:")
            for v in videos[:3]:
                size_mb = v.stat().st_size / 1024 / 1024
                print(f"   - {v.name[:50]}... ({size_mb:.2f} MB)")
            if len(videos) > 3:
                print(f"   ... and {len(videos)-3} more")
    
    print("\n" + "="*70)
    print("🌐 Starting Professional Interface")
    print("📍 URL: http://localhost:7860")
    print("🛑 Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    try:
        # Import and suppress OpenCV warnings
        import cv2
        cv2.setLogLevel(0)
        
        # Check dependencies
        try:
            import gradio as gr
            print(f"✅ Gradio version: {gr.__version__}")
        except ImportError:
            print("❌ Gradio not found. Install with: pip install gradio")
            return
        
        try:
            import scenedetect
            print(f"✅ PySceneDetect version: {scenedetect.__version__}")
            if hasattr(scenedetect, 'detect'):
                print("✅ Using new API (no deprecation warnings!)")
            else:
                print("⚠️ Old API detected, some warnings may appear")
        except ImportError:
            print("⚠️ PySceneDetect not found. Install with: pip install scenedetect")
        
        print("\n" + "-"*70 + "\n")
        
        # Launch the professional app
        from web_app_professional import create_professional_interface
        
        app = create_professional_interface()
        app.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            quiet=False,
            prevent_thread_lock=False
        )
        
    except KeyboardInterrupt:
        print("\n\n✅ Shutting down gracefully...")
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("Install with: pip install -r requirements.txt")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()