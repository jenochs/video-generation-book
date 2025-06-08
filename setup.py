"""Setup configuration for videogenbook package."""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from requirements.txt
def read_requirements():
    requirements = []
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                # Handle optional dependencies
                if "# triton" in line or "# flash-attn" in line:
                    continue
                requirements.append(line)
    return requirements

# Version information
VERSION = "0.1.0"

setup(
    name="videogenbook",
    version=VERSION,
    author="Jenochs",
    author_email="joseph.enochs@gmail.com",
    description="Companion package for 'Hands-On Video Generation with AI' - utilities and examples for 2025's breakthrough video AI models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/jenochs/video-generation-book",
    project_urls={
        "Book Repository": "https://github.com/jenochs/video-generation-book",
        "Bug Reports": "https://github.com/jenochs/video-generation-book/issues",
        "Documentation": "https://github.com/jenochs/video-generation-book/blob/main/README.md",
        "Colab Notebooks": "https://colab.research.google.com/github/jenochs/video-generation-book",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
        "Topic :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Framework :: Jupyter",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.991",
            "pre-commit>=2.20",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "docs": [
            "mkdocs>=1.4",
            "mkdocs-material>=8.0",
            "mkdocs-jupyter>=0.21",
        ],
        "performance": [
            "triton>=2.0; platform_machine=='x86_64'",
            "flash-attn>=2.0; platform_machine=='x86_64'",
        ],
        "cloud": [
            "boto3>=1.26",
            "google-cloud-storage>=2.7",
            "azure-storage-blob>=12.14",
        ],
    },
    entry_points={
        "console_scripts": [
            "videogenbook=videogenbook.cli:main",
        ],
    },
    keywords=[
        "video-generation",
        "artificial-intelligence", 
        "machine-learning",
        "computer-vision",
        "deep-learning",
        "diffusion-models",
        "transformers",
        "jupyter",
        "education",
        "oreilly",
        "veo3",
        "kling-ai",
        "runway",
        "hunyuan",
        "opensora"
    ],
    include_package_data=True,
    package_data={
        "videogenbook": [
            "data/*.json",
            "configs/*.yaml",
            "templates/*.ipynb",
        ],
    },
    zip_safe=False,
)