"""Framework integrations for experiment tracking."""

from ..utils.imports import is_available

# Dynamic imports based on availability
LightningFramework = None
PyTorchFramework = None

_has_lightning = is_available("pytorch_lightning")
_has_torch = is_available("torch")

try:
    from .pytorch import PyTorchFramework
except ImportError:
    PyTorchFramework = None

try:
    from .lightning import LightningFramework
except ImportError:
    LightningFramework = None

# Dynamic __all__ export based on available frameworks
__all__ = []

if PyTorchFramework is not None:
    __all__.append("PyTorchFramework")

if LightningFramework is not None:
    __all__.append("LightningFramework")
