"""Version information for videogenbook package."""

# Version is managed here and imported by setup.py and __init__.py
__version__ = "0.1.0"

# Version components for programmatic access
VERSION_INFO = (0, 1, 0)

def get_version():
    """Get the current version string."""
    return __version__

def get_version_info():
    """Get version as tuple (major, minor, patch)."""
    return VERSION_INFO