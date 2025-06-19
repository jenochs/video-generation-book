# Release Guide for videogenbook

This guide explains how to build, test, and release the videogenbook package to PyPI.

## Prerequisites

1. **PyPI Account**: Create accounts on both [PyPI](https://pypi.org/account/register/) and [Test PyPI](https://test.pypi.org/account/register/)

2. **API Tokens**: Generate API tokens for both PyPI and Test PyPI:
   - PyPI: https://pypi.org/manage/account/token/
   - Test PyPI: https://test.pypi.org/manage/account/token/

3. **Dependencies**: Install build dependencies:
   ```bash
   pip install build twine pytest flake8 mypy
   ```

## Local Development and Testing

### 1. Install in Development Mode
```bash
# Install package in editable mode with dev dependencies
pip install -e .[dev]

# Test the installation
python -c "import videogenbook; print(videogenbook.__version__)"
```

### 2. Run Tests
```bash
# Run the full test suite
python scripts/build_package.py --test

# Or run individual components
flake8 src/videogenbook
mypy src/videogenbook --ignore-missing-imports  
pytest tests/ -v
```

### 3. Build Package Locally
```bash
# Build wheel and source distribution
python scripts/build_package.py --build

# Check the built package
python scripts/build_package.py --info
```

## Release Process

### Step 1: Prepare Release

1. **Update Version**:
   ```bash
   # Edit src/videogenbook/_version.py
   __version__ = "0.1.1"  # New version
   
   # Update pyproject.toml
   version = "0.1.1"
   ```

2. **Update CHANGELOG.md**:
   ```markdown
   ## [0.1.1] - 2025-01-XX
   ### Added
   - New model support for XYZ
   ### Fixed
   - Bug fix for ABC
   ```

3. **Commit Changes**:
   ```bash
   git add .
   git commit -m "Bump version to 0.1.1"
   git push origin main
   ```

### Step 2: Test on Test PyPI

1. **Build and Upload to Test PyPI**:
   ```bash
   # Clean and build
   python scripts/build_package.py --clean --build
   
   # Upload to Test PyPI
   python scripts/build_package.py --test-pypi
   ```

2. **Test Installation from Test PyPI**:
   ```bash
   # Create fresh environment
   python -m venv test_env
   source test_env/bin/activate  # or test_env\Scripts\activate on Windows
   
   # Install from Test PyPI
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ videogenbook
   
   # Test basic functionality
   python -c "import videogenbook; print('Success!')"
   ```

### Step 3: Release on PyPI

1. **Create Git Tag**:
   ```bash
   git tag v0.1.1
   git push origin v0.1.1
   ```

2. **Create GitHub Release**:
   - Go to GitHub repository
   - Click "Releases" → "Create a new release"
   - Choose tag: v0.1.1
   - Release title: "videogenbook v0.1.1"
   - Describe changes from CHANGELOG.md
   - Publish release

3. **Automatic PyPI Upload**:
   - GitHub Action will automatically build and upload to PyPI
   - Monitor the action at: `https://github.com/[username]/video-generation-book/actions`

### Step 4: Manual PyPI Upload (if needed)

```bash
# Build package
python scripts/build_package.py --clean --build

# Upload to PyPI
python scripts/build_package.py --upload
```

## Configuration Files

### .pypirc Configuration
Create `~/.pypirc` for credential management:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_API_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TEST_API_TOKEN_HERE
```

### GitHub Secrets
Add these secrets to your GitHub repository:
- `PYPI_API_TOKEN`: Your PyPI API token
- `TEST_PYPI_API_TOKEN`: Your Test PyPI API token

## Verification

After release, verify the package:

```bash
# Install from PyPI
pip install videogenbook

# Test basic functionality
python -c "
import videogenbook
print(f'Version: {videogenbook.__version__}')
print(f'Models: {videogenbook.SUPPORTED_MODELS}')
"

# Test CLI
videogenbook --help
```

## Troubleshooting

### Common Issues

1. **Build Failures**:
   ```bash
   # Clean everything and retry
   python scripts/build_package.py --clean
   rm -rf ~/.cache/pip  # Clear pip cache
   python scripts/build_package.py --build
   ```

2. **Import Errors**:
   ```bash
   # Check package structure
   python -m build --sdist
   tar -tzf dist/videogenbook-*.tar.gz | head -20
   ```

3. **Dependency Conflicts**:
   ```bash
   # Check requirements
   pip-compile requirements.txt  # if using pip-tools
   # Or manually check pyproject.toml dependencies
   ```

### Version Management

The package uses semantic versioning (SemVer):
- **MAJOR.MINOR.PATCH** (e.g., 1.0.0)
- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible  
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

- [ ] Tests pass locally and in CI
- [ ] Version updated in `_version.py` and `pyproject.toml`
- [ ] CHANGELOG.md updated
- [ ] Documentation updated if needed
- [ ] Test PyPI upload successful
- [ ] Git tag created
- [ ] GitHub release created
- [ ] PyPI upload successful
- [ ] Installation from PyPI verified

## Support

For questions about the release process:
- Create an issue in the GitHub repository
- Contact: joseph.enochs@gmail.com