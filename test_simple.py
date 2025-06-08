#!/usr/bin/env python3
"""
Simple test script for videogenbook - AI video generation made easy
"""

import videogenbook
from videogenbook import get_device

def main():
    print("🎬 Testing videogenbook - Simple Video Generation API")
    print("=" * 60)
    
    # Test 1: Device detection
    print("\n📱 Step 1: Device Detection")
    device = get_device()
    print(f"Using device: {device}")
    
    # Test 2: List available models
    print("\n📋 Step 2: Available Models")
    models = videogenbook.list_available_models()
    print(f"Found {len(models)} models:")
    for i, model in enumerate(models, 1):
        model_info = videogenbook.get_model_info(model)
        if model_info:
            print(f"  {i}. {model}")
            print(f"     • {model_info.name} ({model_info.architecture})")
            print(f"     • Min VRAM: {model_info.min_vram_gb}GB")
    
    # Test 3: GPU Check
    print("\n🖥️  Step 3: Hardware Check")
    gpu_info = videogenbook.check_gpu_memory()
    if gpu_info:
        print(f"GPU: {gpu_info['name']}")
        print(f"VRAM: {gpu_info['memory_total']:.1f}GB total, {gpu_info['memory_free']:.1f}GB free")
        
        # Recommend best model for hardware
        if gpu_info['memory_total'] >= 12:
            recommended = "Lightricks/LTX-Video"
            print(f"✨ Recommended: {recommended} (high quality)")
        elif gpu_info['memory_total'] >= 6:
            recommended = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
            print(f"👍 Recommended: {recommended} (memory efficient)")
        else:
            print("⚠️  Consider Google Colab Pro for better GPU access")
    else:
        print("❌ No GPU detected - use Google Colab Pro for acceleration")
    
    # Test 4: Simple generation example (like genaibook)
    print("\n🎯 Step 4: Ready for Video Generation!")
    print("Now you can generate videos with a simple, powerful API:")
    print()
    print("```python")
    print("import videogenbook")
    print("from videogenbook import get_device")
    print()
    print("# Device detection")
    print("device = get_device()")
    print("print(f'Using device: {device}')")
    print()
    print("# Simple video generation (one-liner API)")
    print("prompt = 'a cat walking in a garden'")
    print("video_path = videogenbook.generate(prompt)")
    print("print(f'Video saved to: {video_path}')")
    print("```")
    print()
    print("✅ videogenbook is ready! Simple, powerful video generation API")

if __name__ == "__main__":
    main()