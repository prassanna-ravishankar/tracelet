"""
Automagic instrumentation for ML experiments.

This module provides automatic detection and logging of:
- Hyperparameters from various sources
- Model architectures and checkpoints
- Dataset information
- Training metrics
- Code and environment versioning
"""

from .core import AutomagicConfig, AutomagicInstrumentor
from .detectors import DatasetDetector, HyperparameterDetector, ModelDetector
from .hooks import FrameworkHookRegistry
from .monitors import ResourceMonitor, TrainingMonitor

__all__ = [
    "AutomagicConfig",
    "AutomagicInstrumentor",
    "DatasetDetector",
    "FrameworkHookRegistry",
    "HyperparameterDetector",
    "ModelDetector",
    "ResourceMonitor",
    "TrainingMonitor",
]
