# Google Colab Notebooks for Video Generation

This directory contains optimized Google Colab notebooks for running video generation models in the cloud.

## 🚀 Featured Notebooks

### HunyuanVideo on A100 GPU
**File**: `hunyuan_colab_a100.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jenochs/video-generation-book/blob/main/notebooks/hunyuan_colab_a100.ipynb)

**What you'll learn:**
- Run Tencent's 13B parameter HunyuanVideo model
- Optimize for Google Colab A100 40GB constraints
- Generate high-quality videos up to 15 seconds
- Master memory management for large AI models

**Requirements:**
- Google Colab Pro+ (for A100 access)
- ~20-30 minutes for setup
- Basic understanding of video generation

## 🎯 Why Use These Notebooks?

### ✅ **Advantages of Colab**
- **No Local Setup**: No need for expensive GPU hardware
- **A100 Access**: Industry-standard AI acceleration
- **Pre-configured**: Optimized environments ready to use
- **Collaborative**: Easy sharing and iteration
- **Cost-Effective**: Pay-per-use cloud computing

### ⚡ **Optimizations Included**
- **Memory Management**: Efficient use of 40GB A100 memory
- **Quality Presets**: Balanced, fast, and high-quality modes
- **Batch Generation**: Process multiple prompts efficiently
- **Export Tools**: Save to Google Drive or download directly
- **Error Handling**: Graceful fallbacks and troubleshooting

## 🛠️ Setup Instructions

### 1. Get Google Colab Pro+
For A100 GPU access, you need [Colab Pro+](https://colab.research.google.com/signup):
- **Colab Pro**: Usually gets T4 GPUs (16GB) - limited for large models
- **Colab Pro+**: Includes A100 access (40GB) - optimal for video generation

### 2. Open the Notebook
Click the "Open in Colab" badge above or manually open:
```
https://colab.research.google.com/github/jenochs/video-generation-book/blob/main/notebooks/hunyuan_colab_a100.ipynb
```

### 3. Set Runtime to A100
1. Go to **Runtime → Change runtime type**
2. Set **Hardware accelerator**: GPU
3. Set **GPU type**: A100 (if available)
4. Click **Save**

### 4. Run the Setup Cells
Follow the notebook step-by-step:
1. **Environment Setup** - Verify A100 and configure optimizations
2. **Install Dependencies** - Latest PyTorch and diffusers
3. **Load HunyuanVideo** - Download and optimize the 13B model
4. **Generate Videos** - Create your first AI video!

## 📊 Performance Expectations

### A100 40GB Performance
| Quality Setting | Resolution | Duration | Memory Usage | Generation Time |
|----------------|------------|----------|--------------|-----------------|
| **High Quality** | 720x1280 | ~8s (65 frames) | ~35GB | 8-12 minutes |
| **Balanced** | 544x960 | ~8s (65 frames) | ~25GB | 6-10 minutes |
| **Fast** | 512x512 | ~4s (32 frames) | ~15GB | 3-6 minutes |

### Tips for Best Results
- **Use descriptive prompts**: "A majestic eagle soaring over mountains at sunset, cinematic camera movement"
- **Include camera details**: "slow motion", "wide angle", "close-up"
- **Specify atmosphere**: "dramatic lighting", "golden hour", "misty morning"
- **Set consistent seeds**: For reproducible results across experiments

## 🎬 Example Prompts

### Nature & Wildlife
```
A hummingbird feeding from vibrant tropical flowers, macro lens, slow motion, beautiful natural lighting
```

### Urban & Architecture  
```
A futuristic cyberpunk city at night, neon reflections in rain puddles, flying cars, cinematic atmosphere
```

### Fantasy & Sci-Fi
```
A dragon flying through clouds above a medieval castle, epic fantasy scene, golden hour lighting
```

### Abstract & Artistic
```
Colorful paint mixing in water, abstract fluid dynamics, high-speed macro photography, artistic lighting
```

## 🔧 Troubleshooting

### Common Issues & Solutions

**❌ "Runtime disconnected"**
- Colab has 12-hour session limits
- Save progress to Google Drive regularly
- Use checkpoint system in notebook

**❌ "CUDA out of memory"**  
- Switch to "Fast" quality preset
- Reduce num_frames (try 32 instead of 65)
- Restart runtime to clear memory

**❌ "Model download failed"**
- Check internet connection
- Verify sufficient disk space (~30GB needed)
- Try running download cell again

**❌ "A100 not available"**
- A100 availability varies by region and time
- Try different times of day
- Consider using T4 with reduced settings

### Getting Help
1. **Check the troubleshooting section** in the notebook
2. **Review error messages** carefully - they often contain solutions
3. **Visit the GitHub repository** for updates and community support
4. **Join discussions** on the repository's Issues page

## 🎓 Learning Path

### Beginner
1. **Start with the basic notebook** - Follow step-by-step
2. **Try different prompts** - Experiment with various subjects
3. **Understand quality settings** - Learn the memory/quality tradeoffs

### Intermediate  
1. **Customize generation parameters** - Modify resolution, steps, guidance
2. **Implement batch processing** - Generate multiple videos efficiently
3. **Optimize for your use case** - Find the best settings for your needs

### Advanced
1. **Study the optimization techniques** - Understand memory management
2. **Modify the pipeline code** - Customize for specific requirements
3. **Contribute improvements** - Share optimizations with the community

## 📚 Additional Resources

- **Main Repository**: [video-generation-book](https://github.com/jenochs/video-generation-book)
- **HunyuanVideo Model**: [Hugging Face](https://huggingface.co/tencent/HunyuanVideo)
- **Diffusers Documentation**: [Usage Guide](https://huggingface.co/docs/diffusers/)
- **Google Colab Guide**: [Pro Features](https://colab.research.google.com/signup)

## 🤝 Contributing

Found an optimization or improvement? We welcome contributions!

1. **Fork the repository**
2. **Create a feature branch**  
3. **Test your changes** in Colab
4. **Submit a pull request**

Common contribution areas:
- Memory optimization techniques
- New quality presets
- Additional export formats
- UI/UX improvements
- Documentation updates

---

**Happy video generating! 🎬✨**

*From the book: "Hands-On Video Generation with AI" - Bringing breakthrough models to your fingertips*
