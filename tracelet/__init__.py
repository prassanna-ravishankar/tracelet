"""
Tracelet - A lightweight ML experiment tracker
"""

__version__ = "0.1.0"

from .collectors.git import GitCollector
from .collectors.system import SystemMetricsCollector
from .core.experiment import Experiment, ExperimentConfig
from .frameworks.lightning import LightningFramework
from .frameworks.pytorch import PyTorchFramework
from .interface import get_active_experiment, start_logging, stop_logging

# Optional imports - check availability
try:
    import importlib.util

    spec = importlib.util.find_spec("tracelet.backends.mlflow")
    _has_mlflow = spec is not None
except ImportError:
    _has_mlflow = False

# Check for automagic support
try:
    from .automagic.core import automagic, capture_hyperparams  # noqa: F401

    _has_automagic = True
except ImportError:
    _has_automagic = False

__all__ = [
    # Core components
    "Experiment",
    "ExperimentConfig",
    "GitCollector",
    "LightningFramework",
    "PyTorchFramework",
    "SystemMetricsCollector",
    "get_active_experiment",
    # Main interface
    "start_logging",
    "stop_logging",
]

# Add automagic components if available
if _has_automagic:
    __all__.extend([
        "automagic",
        "capture_hyperparams",
    ])

# Note: MLflowBackend is available via backends.mlflow when _has_mlflow is True
