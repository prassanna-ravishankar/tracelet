"""Backend implementations for Tracelet."""

from typing import Optional

from ..utils.imports import ImportManager

# Initialize import manager
_import_manager = ImportManager()

# Backend mapping for dynamic imports
_BACKEND_MODULES = {
    "aim": ("aim_backend", "AimBackend", "aim"),
    "clearml": ("clearml_backend", "ClearMLBackend", "clearml"),
    "mlflow": ("mlflow_backend", "MLflowBackend", "mlflow"),
    "wandb": ("wandb_backend", "WandbBackend", "wandb"),
}


def get_backend(backend_name: str) -> Optional[type]:
    """Get a backend class by name, dynamically importing if available.

    Args:
        backend_name: Name of the backend (e.g., 'mlflow', 'wandb')

    Returns:
        Backend class if available, None otherwise
    """
    if backend_name not in _BACKEND_MODULES:
        return None

    module_name, class_name, framework = _BACKEND_MODULES[backend_name]

    # Check if required framework is available
    if not _import_manager.is_available(framework):
        return None

    try:
        module = __import__(f"tracelet.backends.{module_name}", fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError:
        return None


def available_backends() -> dict[str, bool]:
    """Get availability status of all backends.

    Returns:
        Dictionary mapping backend names to their availability
    """
    return {name: _import_manager.is_available(framework) for name, (_, _, framework) in _BACKEND_MODULES.items()}


# For backwards compatibility and direct imports when available
AimBackend = None
ClearMLBackend = None
MLflowBackend = None
WandbBackend = None

try:
    from .aim_backend import AimBackend  # noqa: F401

    _has_aim = True
except ImportError:
    _has_aim = False

try:
    from .clearml_backend import ClearMLBackend  # noqa: F401

    _has_clearml = True
except ImportError:
    _has_clearml = False

try:
    from .mlflow_backend import MLflowBackend  # noqa: F401

    _has_mlflow = True
except ImportError:
    _has_mlflow = False

try:
    from .wandb_backend import WandbBackend  # noqa: F401

    _has_wandb = True
except ImportError:
    _has_wandb = False

# Dynamic __all__ export based on available backends
__all__ = ["available_backends", "get_backend"]

if _has_aim:
    __all__.append("AimBackend")
if _has_clearml:
    __all__.append("ClearMLBackend")
if _has_mlflow:
    __all__.append("MLflowBackend")
if _has_wandb:
    __all__.append("WandbBackend")
