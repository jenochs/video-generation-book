#!/usr/bin/env python3
"""Build and distribution script for videogenbook package."""

import subprocess
import sys
import shutil
from pathlib import Path
import argparse


def run_command(cmd, check=True):
    """Run a command and handle errors."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result


def clean_build():
    """Clean previous build artifacts."""
    print("🧹 Cleaning build artifacts...")
    
    paths_to_clean = [
        "build",
        "dist", 
        "src/videogenbook.egg-info",
        "*.egg-info",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
    ]
    
    for path in paths_to_clean:
        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_dir():
                shutil.rmtree(path_obj)
                print(f"  Removed directory: {path}")
            else:
                path_obj.unlink()
                print(f"  Removed file: {path}")
    
    # Clean pycache recursively
    for pycache in Path(".").rglob("__pycache__"):
        shutil.rmtree(pycache)
        print(f"  Removed: {pycache}")


def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    
    # Install dev dependencies
    run_command([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])
    
    # Run linting
    print("  Running flake8...")
    run_command([sys.executable, "-m", "flake8", "src/videogenbook"], check=False)
    
    # Run type checking
    print("  Running mypy...")
    run_command([sys.executable, "-m", "mypy", "src/videogenbook", "--ignore-missing-imports"], check=False)
    
    # Run pytest
    print("  Running pytest...")
    run_command([sys.executable, "-m", "pytest", "tests/", "-v"])


def build_package():
    """Build the package."""
    print("📦 Building package...")
    
    # Install build dependencies
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "build", "twine"])
    
    # Build package
    run_command([sys.executable, "-m", "build"])
    
    # Check distribution
    print("🔍 Checking distribution...")
    run_command([sys.executable, "-m", "twine", "check", "dist/*"])


def upload_to_pypi(test=False):
    """Upload to PyPI or Test PyPI."""
    if test:
        print("📤 Uploading to Test PyPI...")
        run_command([
            sys.executable, "-m", "twine", "upload", 
            "--repository", "testpypi",
            "dist/*"
        ])
    else:
        print("📤 Uploading to PyPI...")
        run_command([sys.executable, "-m", "twine", "upload", "dist/*"])


def show_package_info():
    """Show package information."""
    print("📋 Package Information:")
    
    # Import version
    sys.path.insert(0, "src")
    import videogenbook
    
    print(f"  Name: {videogenbook.__package_name__}")
    print(f"  Version: {videogenbook.__version__}")
    print(f"  Author: {videogenbook.__author__}")
    print(f"  Description: {videogenbook.__description__}")
    print(f"  URL: {videogenbook.__url__}")
    
    # Show distribution files
    dist_path = Path("dist")
    if dist_path.exists():
        print(f"  Distribution files:")
        for file in dist_path.iterdir():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"    {file.name} ({size_mb:.1f} MB)")


def main():
    """Main build script."""
    parser = argparse.ArgumentParser(description="Build and distribute videogenbook package")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--build", action="store_true", help="Build package")
    parser.add_argument("--upload", action="store_true", help="Upload to PyPI")
    parser.add_argument("--test-pypi", action="store_true", help="Upload to Test PyPI")
    parser.add_argument("--info", action="store_true", help="Show package info")
    parser.add_argument("--all", action="store_true", help="Run full build pipeline")
    
    args = parser.parse_args()
    
    if args.all:
        clean_build()
        run_tests()
        build_package()
        show_package_info()
        return
    
    if args.clean:
        clean_build()
    
    if args.test:
        run_tests()
    
    if args.build:
        build_package()
    
    if args.upload:
        upload_to_pypi(test=False)
    
    if args.test_pypi:
        upload_to_pypi(test=True)
    
    if args.info:
        show_package_info()
    
    if not any(vars(args).values()):
        # Default behavior
        print("🚀 Running default build pipeline...")
        clean_build()
        build_package()
        show_package_info()


if __name__ == "__main__":
    main()