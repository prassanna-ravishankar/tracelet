"""Dynamic import management system for Tracelet.

This module provides a centralized way to manage optional dependencies
and gracefully handle missing packages.
"""

import importlib
import importlib.util
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ImportManager:
    """Centralized manager for handling optional dependencies and dynamic imports."""

    def __init__(self):
        self._available_frameworks: dict[str, bool] = {}
        self._loaded_modules: dict[str, Any] = {}
        self._framework_specs = {
            # ML Frameworks
            "torch": "torch",
            "pytorch_lightning": "pytorch_lightning",
            "lightning": "lightning",
            "tensorflow": "tensorflow",
            "jax": "jax",
            "numpy": "numpy",
            "sklearn": "sklearn",
            "xgboost": "xgboost",
            # Experiment Tracking Backends
            "mlflow": "mlflow",
            "wandb": "wandb",
            "clearml": "clearml",
            "aim": "aim",
            "neptune": "neptune",
            "tensorboard": "tensorboardX",
            # Utilities
            "pynvml": "pynvml",
            "psutil": "psutil",
            "git": "git",
            "gitpython": "git",
            "GPUtil": "GPUtil",
            "nvidia_ml_py3": "nvidia_ml_py3",
        }
        self._strict_mode = os.environ.get("TRACELET_STRICT_IMPORTS", "false").lower() == "true"

        # Initialize framework availability
        self._detect_all_frameworks()

    def _detect_all_frameworks(self):
        """Detect availability of all known frameworks."""
        for name, module_name in self._framework_specs.items():
            self._available_frameworks[name] = self._check_availability(module_name)

    def _check_availability(self, module_name: str) -> bool:
        """Check if a module is available without importing it."""
        try:
            spec = importlib.util.find_spec(module_name)
            return spec is not None
        except (ImportError, ValueError, ModuleNotFoundError):
            return False

    def is_available(self, framework: str) -> bool:
        """Check if a framework is available.

        Args:
            framework: Name of the framework to check

        Returns:
            True if the framework is available, False otherwise
        """
        return self._available_frameworks.get(framework, False)

    def get_available_frameworks(self) -> dict[str, bool]:
        """Get a dictionary of all frameworks and their availability."""
        return self._available_frameworks.copy()

    def get_available_backends(self) -> list[str]:
        """Get list of available backend frameworks."""
        backends = ["mlflow", "wandb", "clearml", "aim", "neptune", "tensorboard"]
        return [backend for backend in backends if self.is_available(backend)]

    def require(self, framework: str, feature: Optional[str] = None) -> Any:
        """Import and return a framework module, with helpful error messages.

        Args:
            framework: Name of the framework to import
            feature: Optional feature name for better error messages

        Returns:
            The imported module

        Raises:
            ImportError: If the framework is not available
        """
        if not self.is_available(framework):
            self._raise_import_error(framework, feature)

        # Load the module if not already cached
        if framework not in self._loaded_modules:
            module_name = self._framework_specs.get(framework, framework)
            try:
                self._loaded_modules[framework] = importlib.import_module(module_name)
                logger.debug(f"Loaded {framework} module successfully")
            except ImportError as e:
                logger.exception(f"Failed to import {framework}: {e}")
                self._raise_import_error(framework, feature)

        return self._loaded_modules[framework]

    def try_import(self, framework: str, default: Any = None) -> Any:
        """Try to import a framework, returning default if not available.

        Args:
            framework: Name of the framework to import
            default: Value to return if import fails

        Returns:
            The imported module or the default value
        """
        try:
            return self.require(framework)
        except ImportError:
            return default

    def lazy_import(self, framework: str):
        """Create a lazy import wrapper for a framework.

        Args:
            framework: Name of the framework

        Returns:
            A lazy import wrapper that imports on first access
        """
        return LazyImport(self, framework)

    def _raise_import_error(self, framework: str, feature: Optional[str] = None):
        """Raise a helpful ImportError for missing dependencies."""
        install_commands = {
            "torch": "pip install torch",
            "pytorch_lightning": "pip install pytorch-lightning",
            "lightning": "pip install lightning",
            "tensorflow": "pip install tensorflow",
            "jax": "pip install jax",
            "numpy": "pip install numpy",
            "sklearn": "pip install scikit-learn",
            "xgboost": "pip install xgboost",
            "mlflow": "pip install mlflow",
            "wandb": "pip install wandb",
            "clearml": "pip install clearml",
            "aim": "pip install aim",
            "neptune": "pip install neptune-client",
            "tensorboard": "pip install tensorboardX",
            "pynvml": "pip install pynvml",
            "GPUtil": "pip install GPUtil",
            "nvidia_ml_py3": "pip install nvidia-ml-py3",
        }

        install_cmd = install_commands.get(framework, f"pip install {framework}")
        feature_msg = f" for {feature}" if feature else ""

        error_msg = f"{framework} is required{feature_msg} but is not installed. " f"Install it with: {install_cmd}"

        if self._strict_mode:
            raise ImportError(error_msg)
        else:
            logger.warning(error_msg)
            raise ImportError(error_msg)

    def reload_framework_detection(self):
        """Reload framework detection (useful for dynamic environments)."""
        self._available_frameworks.clear()
        self._loaded_modules.clear()
        self._detect_all_frameworks()


class LazyImport:
    """Lazy import wrapper that imports modules on first access."""

    def __init__(self, import_manager: ImportManager, framework: str):
        self._import_manager = import_manager
        self._framework = framework
        self._module = None
        self._loaded = False

    def _load(self):
        """Load the module on first access."""
        if not self._loaded:
            self._module = self._import_manager.require(self._framework)
            self._loaded = True
        return self._module

    def __getattr__(self, name):
        """Get attribute from the lazy-loaded module."""
        module = self._load()
        return getattr(module, name)

    def __call__(self, *args, **kwargs):
        """Call the lazy-loaded module if it's callable."""
        module = self._load()
        return module(*args, **kwargs)

    def __bool__(self):
        """Check if the module is available."""
        return self._import_manager.is_available(self._framework)


# Global import manager instance
_import_manager = ImportManager()


# Convenience functions
def is_available(framework: str) -> bool:
    """Check if a framework is available."""
    return _import_manager.is_available(framework)


def require(framework: str, feature: Optional[str] = None) -> Any:
    """Import and return a framework module."""
    return _import_manager.require(framework, feature)


def try_import(framework: str, default: Any = None) -> Any:
    """Try to import a framework, returning default if not available."""
    return _import_manager.try_import(framework, default)


def lazy_import(framework: str):
    """Create a lazy import wrapper for a framework."""
    return _import_manager.lazy_import(framework)


def get_available_frameworks() -> dict[str, bool]:
    """Get a dictionary of all frameworks and their availability."""
    return _import_manager.get_available_frameworks()


def get_available_backends() -> list[str]:
    """Get list of available backend frameworks."""
    return _import_manager.get_available_backends()


def reload_detection():
    """Reload framework detection."""
    _import_manager.reload_framework_detection()


# Create lazy imports for common frameworks
torch = lazy_import("torch")
pytorch_lightning = lazy_import("pytorch_lightning")
lightning = lazy_import("lightning")
tensorflow = lazy_import("tensorflow")
jax = lazy_import("jax")
numpy = lazy_import("numpy")
sklearn = lazy_import("sklearn")
xgboost = lazy_import("xgboost")
mlflow = lazy_import("mlflow")
wandb = lazy_import("wandb")
clearml = lazy_import("clearml")
aim = lazy_import("aim")
neptune = lazy_import("neptune")
pynvml = lazy_import("pynvml")
GPUtil = lazy_import("GPUtil")
nvidia_ml_py3 = lazy_import("nvidia_ml_py3")
