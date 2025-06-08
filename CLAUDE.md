# Claude Code Configuration

## Project Overview
**Hands-On Video Generation with AI** - A comprehensive technical guide covering the latest 2025 developments in AI video generation, from foundational concepts to production deployment.

## Repository Structure
```
video-generation-book/
├── notebooks/               # Jupyter notebooks organized by chapter
│   ├── 01_foundations/     # Chapter 1: Foundations of Video Generation
│   ├── 02_data/           # Chapter 2: Understanding and Preparing Video Data
│   ├── 03_implementation/ # Chapter 3: Implementing Video Generation Models
│   ├── 04_training/       # Chapter 4: Training on Video Datasets
│   ├── 05_fine_tuning/    # Chapter 5: Fine-Tuning for Specific Video Tasks
│   ├── 06_text_prompts/   # Chapter 6: Integrating Text Prompts with Video Generation
│   ├── 07_deployment/     # Chapter 7: Deploying Video Models
│   └── 08_evaluation/     # Chapter 8: Evaluation and Iteration
├── src/                   # Reusable Python modules and utilities
├── data/                  # Sample datasets and examples
├── models/               # Model checkpoints and configurations
├── docs/                 # Documentation and guides
├── tests/               # Test suite for code examples
└── scripts/             # Utility scripts for setup and automation
```

## Development Environment

### Python Environment
- **Python Version**: 3.9+
- **Primary Framework**: PyTorch 2.7.1
- **Key Libraries**: Diffusers 0.33.1, Transformers 4.52.4
- **Environment File**: `environment.yml` (conda) + `requirements.txt` (pip)

### Hardware Requirements
- **Minimum**: 8GB GPU VRAM (with quantization), 16GB RAM
- **Recommended**: 16GB+ GPU VRAM (RTX 4080/4090, A100), 32GB+ RAM
- **Cloud Alternative**: Google Colab Pro with A100 access

## Key Commands

### Environment Setup
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate video-generation-book

# Install additional dependencies
pip install -r requirements.txt
```

### Development Workflow
```bash
# Launch Jupyter Lab for interactive development
jupyter lab

# Run tests (when test framework is established)
pytest tests/

# Format code (when established)
black src/ notebooks/

# Type checking (when established)
mypy src/
```

### Model Operations
```bash
# Download model checkpoints (example commands)
python scripts/download_models.py

# Run basic inference test
python scripts/test_inference.py

# Performance benchmarking
python scripts/benchmark_models.py
```

## Featured 2025 Models

### Breakthrough Models
- **Google Veo 3**: First native audio-visual generation model
- **Kling AI 2.0**: Global performance leader (Arena ELO 1000, 22M+ users)
- **Runway Gen-4**: Character consistency breakthrough
- **HunyuanVideo**: Largest open-source model (13B parameters)
- **OpenSora 2.0**: 50% cost reduction achievement
- **Pika 2.2**: Advanced creative control features

### Core Technologies
- **Diffusion Transformers (DiT)**: Hybrid architectures for superior quality
- **Multi-Modal Visual Language (MVL)**: Breakthrough consistency techniques
- **Reference-Guided Conditioning**: Object identity preservation
- **Native Audio Synthesis**: Synchronized audio-visual generation

## Code Standards

### Implementation Guidelines
- **Library Versions**: Use exact versions specified in requirements.txt
- **Memory Optimization**: Implement quantization, VAE tiling, CPU offloading
- **Error Handling**: Graceful fallbacks for different hardware configurations
- **Documentation**: Clear comments explaining model-specific parameters

### Example Code Pattern
```python
import torch
from diffusers import DiffusionPipeline
from transformers import AutoTokenizer, AutoModel

# Memory optimization setup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Model initialization with error handling
try:
    pipe = DiffusionPipeline.from_pretrained(
        "model_name",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
except Exception as e:
    print(f"GPU setup failed: {e}")
    # Fallback to CPU or alternative configuration
```

## Testing Strategy

### Model Testing
- **Inference Tests**: Verify basic model functionality
- **Performance Benchmarks**: Speed and memory usage across hardware
- **Output Quality**: Automated metrics and sample generation
- **Integration Tests**: End-to-end pipeline validation

### When Adding New Models
1. Add model configuration to `src/models/config.py`
2. Implement in notebook with complete example
3. Add performance benchmark to `scripts/benchmark_models.py`
4. Update requirements.txt if new dependencies needed
5. Test across different hardware configurations

## Evaluation Standards

### Performance Metrics
- **Arena ELO Scoring**: Industry-standard competitive evaluation
- **Consistency Metrics**: Character and object preservation across frames
- **Technical Quality**: Resolution, frame rate, temporal smoothness
- **Audio-Visual Sync**: For models supporting audio generation

### Benchmark Hardware
- **Primary**: RTX 4090 (24GB VRAM) for consistent testing
- **Cloud**: Google Colab A100 for accessibility verification
- **Memory Profiles**: Document usage for 8GB, 16GB, 24GB configurations

## Documentation

### Chapter Documentation
Each chapter notebook should include:
- **Learning objectives** clearly stated at the beginning
- **Theory section** with accessible explanations and analogies
- **Working code examples** with complete implementations
- **Hands-on exercises** that readers can modify and extend
- **Performance analysis** with memory and speed considerations

### Code Documentation
- **Docstrings**: All functions and classes
- **Type hints**: For better IDE support and clarity
- **Error explanations**: Common issues and solutions
- **Hardware notes**: GPU memory requirements and optimizations

## Deployment Considerations

### Production Readiness
- **API Integration**: FastAPI/Flask patterns for model serving
- **Scaling**: Multi-GPU and distributed inference strategies
- **Monitoring**: Performance tracking and error logging
- **Cost Analysis**: GPU time and memory optimization techniques

### Cloud Platforms
- **Google Colab**: Interactive learning and experimentation
- **AWS SageMaker**: Production training and inference
- **Azure ML**: Enterprise deployment scenarios
- **Local Development**: GPU workstation setup guides

## Community and Collaboration

### Repository Management
- **GitHub Issues**: Technical questions and bug reports
- **Pull Requests**: Community contributions and improvements
- **Discussions**: Model comparisons and optimization strategies
- **Wiki**: Extended documentation and troubleshooting

### Update Strategy
- **Monthly**: New model releases and performance updates
- **Quarterly**: Major framework version updates and benchmarks
- **As-needed**: Critical bug fixes and security patches
- **Community-driven**: User-contributed optimizations and examples

## Current Status

### Completed Components
- ✅ Chapter 1: Complete with 2025 state-of-the-art models
- ✅ Requirements specification with latest stable versions
- ✅ Repository structure and documentation framework
- ✅ Basic environment configuration

### Next Priorities
1. **Environment Files**: Create `environment.yml` for conda setup
2. **Notebook Structure**: Implement Chapter 1 as interactive Jupyter notebook
3. **Utility Modules**: Core functions for model loading and inference
4. **Testing Framework**: Basic smoke tests for model functionality
5. **Documentation**: Contributing guidelines and troubleshooting guide

### Long-term Goals
- **Complete Notebook Suite**: All 8 chapters as interactive notebooks
- **Model Zoo**: Pre-configured setups for all featured models
- **Community Platform**: Discord integration and user support
- **Automated Testing**: CI/CD for code quality and model verification
- **Performance Optimization**: Advanced techniques for different hardware tiers

---

**Last Updated**: January 2025
**Primary Maintainer**: jenochs
**Repository**: https://github.com/jenochs/video-generation-book