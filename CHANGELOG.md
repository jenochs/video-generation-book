# Changelog

All notable changes to the videogenbook project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Enhanced dependency resolution for OpenSora v2 notebook
- Fixed version conflicts for xformers, safetensors, and huggingface-hub

## [0.1.0] - 2025-01-08

### Added
- Initial package structure with comprehensive video generation utilities
- Support for 2025's breakthrough video generation models:
  - LTX-Video (Lightricks) - working implementation
  - HunyuanVideo (Tencent) - 13B parameter model
  - OpenSora v2 (HPC-AI Tech) - 11B parameter model
  - Wan2.1-T2V (Wan-AI) - working implementation
- Core modules:
  - `models.py`: Model loading and configuration
  - `generation.py`: Video generation workflows
  - `utils.py`: Utility functions and environment setup
  - `evaluation.py`: Performance evaluation and benchmarking
  - `workflows.py`: High-level workflow abstractions
  - `cli.py`: Command-line interface
- Google Colab optimized notebooks:
  - HunyuanVideo A100 notebook with memory optimizations
  - OpenSora v2 A100 notebook with progressive quality settings
- Package infrastructure:
  - Modern pyproject.toml configuration
  - Comprehensive setup.py for compatibility
  - MANIFEST.in for proper file inclusion
  - GitHub Actions for CI/CD
  - Type hints and py.typed marker
- Documentation:
  - Comprehensive README with examples
  - RELEASE.md guide for package deployment
  - CONTRIBUTING.md for contributors
  - TROUBLESHOOTING.md for common issues
- Development tools:
  - Pre-commit hooks configuration
  - Pytest test framework setup
  - MyPy type checking
  - Black code formatting
  - Flake8 linting
- Build and deployment:
  - Automated PyPI release workflow
  - Build scripts for local development
  - Version management system

### Features
- Simple one-liner API: `videogenbook.generate("prompt")`
- Memory optimization for different GPU configurations
- Progressive quality settings (ultra_conservative to ambitious)
- Batch generation capabilities
- Audio-visual generation support
- Comprehensive error handling and fallbacks
- Hardware detection and optimization
- Cloud deployment utilities

### Hardware Support
- Minimum: 8GB GPU VRAM with quantization
- Recommended: 16GB+ GPU VRAM (RTX 4080/4090)
- Optimal: 24GB+ GPU VRAM (RTX 4090, A100)
- Google Colab A100 optimization

### Dependencies
- PyTorch 2.7.1+ with CUDA support
- Diffusers 0.33.1+ for state-of-the-art models
- Transformers 4.52.4+ for text encoding
- Comprehensive video processing stack (OpenCV, imageio, etc.)
- Jupyter ecosystem for interactive development

## [0.0.1] - 2025-01-01

### Added
- Initial project setup and repository structure
- Basic package skeleton
- Core concept validation