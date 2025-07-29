"""
Tracelet - A lightweight ML experiment tracker
"""

__version__ = "0.1.0"

from .backends import get_backend
from .core.experiment import Experiment, ExperimentConfig
from .interface import get_active_experiment, start_logging, stop_logging

# Dynamic import system
from .utils.imports import get_available_backends, get_available_frameworks, is_available

# Framework availability checks
_has_mlflow = is_available("mlflow")
_has_torch = is_available("torch")
_has_lightning = is_available("pytorch_lightning")
_has_git = is_available("git")
_has_psutil = is_available("psutil")

# Check for automagic support
try:
    from .automagic.core import automagic, capture_hyperparams

    _has_automagic = True
except ImportError:
    _has_automagic = False

__all__ = [
    # Core components
    "Experiment",
    "ExperimentConfig",
    "get_active_experiment",
    # Main interface
    "start_logging",
    "stop_logging",
    # Dynamic backend access
    "get_backend",
    "available_backends",
    "available_frameworks",
]

# Add automagic components if available
if _has_automagic:
    __all__.extend([
        "automagic",
        "capture_hyperparams",
    ])
    # Make imports available at module level
    automagic = automagic
    capture_hyperparams = capture_hyperparams


# Public API for dynamic imports
def available_backends():
    """Get list of available experiment tracking backends.

    Returns:
        List of backend names that are currently available based on installed packages.
    """
    return get_available_backends()


def available_frameworks():
    """Get dictionary of framework availability.

    Returns:
        Dictionary mapping framework names to their availability status.
    """
    return get_available_frameworks()


# Note: Backend classes are available dynamically when their dependencies are installed
