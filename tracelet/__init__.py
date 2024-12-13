"""
Tracelet - A lightweight ML experiment tracker
"""

__version__ = "0.1.0"

from .interface import start_logging, stop_logging, get_active_experiment
from .core.experiment import Experiment, ExperimentConfig
from .frameworks.pytorch import PyTorchFramework
from .frameworks.lightning import LightningFramework
from .backends.mlflow import MLflowBackend
from .collectors.git import GitCollector
from .collectors.system import SystemMetricsCollector

__all__ = [
    # Main interface
    "start_logging",
    "stop_logging",
    "get_active_experiment",
    
    # Core components
    "Experiment",
    "ExperimentConfig",
    "PyTorchFramework",
    "LightningFramework",
    "MLflowBackend",
    "GitCollector",
    "SystemMetricsCollector",
] 