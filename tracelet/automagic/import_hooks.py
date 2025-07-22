"""
Import hooks for automatic detection of ML models and datasets.
"""

import builtins


class AutomagicImportHook:
    """Import hook that automatically detects and instruments ML objects."""

    def __init__(self, instrumentor):
        self.instrumentor = instrumentor
        self._original_import = builtins.__import__
        self._tracked_modules: set[str] = set()
        self._hooked = False

    def install(self):
        """Install the import hook."""
        if not self._hooked:
            builtins.__import__ = self._hooked_import
            self._hooked = True

    def uninstall(self):
        """Uninstall the import hook."""
        if self._hooked:
            builtins.__import__ = self._original_import
            self._hooked = False

    def _hooked_import(self, name, *args, **kwargs):
        """Hooked import function that instruments ML modules."""
        module = self._original_import(name, *args, **kwargs)

        # Check if this is an ML module we care about
        if self._is_ml_module(name):
            self._instrument_module(module, name)

        return module

    def _is_ml_module(self, module_name: str) -> bool:
        """Check if a module is ML-related and should be instrumented."""
        ml_modules = {
            "torch",
            "torch.nn",
            "torch.optim",
            "torch.utils.data",
            "sklearn",
            "sklearn.ensemble",
            "sklearn.linear_model",
            "tensorflow",
            "keras",
            "xgboost",
            "lightgbm",
            "pandas",
            "numpy",
        }

        return any(module_name.startswith(ml_mod) for ml_mod in ml_modules)

    def _instrument_module(self, module, name: str):
        """Instrument an ML module for automatic detection."""
        if name in self._tracked_modules:
            return

        self._tracked_modules.add(name)

        # Add hooks based on module type
        if "torch" in name:
            self._instrument_pytorch(module)
        elif "sklearn" in name:
            self._instrument_sklearn(module)
        elif "pandas" in name:
            self._instrument_pandas(module)

    def _instrument_pytorch(self, module):
        """Add instrumentation to PyTorch modules."""
        # Hook common constructors
        if hasattr(module, "Module"):
            self._wrap_constructor(module.Module, "pytorch_model")

        if hasattr(module, "DataLoader"):
            self._wrap_constructor(module.DataLoader, "pytorch_dataloader")

    def _instrument_sklearn(self, module):
        """Add instrumentation to sklearn modules."""
        # Hook common estimators
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and hasattr(attr, "fit") and hasattr(attr, "predict"):
                self._wrap_constructor(attr, "sklearn_model")

    def _instrument_pandas(self, module):
        """Add instrumentation to pandas."""
        if hasattr(module, "DataFrame"):
            self._wrap_constructor(module.DataFrame, "pandas_dataframe")

    def _wrap_constructor(self, cls, object_type: str):
        """Wrap a class constructor to automatically capture object info."""
        if hasattr(cls, "_tracelet_wrapped"):
            return  # Already wrapped

        original_init = cls.__init__

        def wrapped_init(self, *args, **kwargs):
            # Call original constructor
            result = original_init(self, *args, **kwargs)

            # Automatically capture object info for active experiments
            self._auto_capture_object(self, object_type)

            return result

        cls.__init__ = wrapped_init
        cls._tracelet_wrapped = True

    def _auto_capture_object(self, obj, object_type: str):
        """Automatically capture information about a created object."""
        # Get active experiments from instrumentor
        active_experiments = self.instrumentor._active_experiments

        for _exp_id, exp_ref in active_experiments.items():
            experiment = exp_ref()
            if experiment:
                try:
                    if object_type.endswith("_model"):
                        self.instrumentor.capture_model_info(obj, experiment)
                    elif object_type.endswith("_dataloader") or object_type.endswith("_dataframe"):
                        self.instrumentor.capture_dataset_info(obj, experiment)
                except Exception as e:
                    # Don't let instrumentation errors break user code
                    print(f"Warning: Failed to auto-capture {object_type}: {e}")


# Global import hook instance
_import_hook = None


def install_import_hooks(instrumentor):
    """Install import hooks for automatic ML object detection."""
    global _import_hook
    if _import_hook is None:
        _import_hook = AutomagicImportHook(instrumentor)
        _import_hook.install()


def uninstall_import_hooks():
    """Uninstall import hooks."""
    global _import_hook
    if _import_hook:
        _import_hook.uninstall()
        _import_hook = None
