"""
Framework-specific hooks for automatic instrumentation.
"""

import functools
import importlib.util
import threading
import warnings
import weakref
from typing import Callable, Optional

from ..core.experiment import Experiment


class FrameworkHook:
    """Base class for framework-specific hooks."""

    def __init__(self, experiment: Experiment):
        self.experiment_ref = weakref.ref(experiment)
        self._patched_functions = {}
        self._original_functions = {}

        # Store reference to instrumentor for finding active experiments
        from .core import AutomagicInstrumentor

        self._instrumentor = AutomagicInstrumentor.get_instance()
        self.config = self._instrumentor.config  # Access to configuration

    @property
    def experiment(self) -> Optional[Experiment]:
        """Get the experiment, returning None if it's been garbage collected."""
        return self.experiment_ref()

    def _get_any_active_experiment(self) -> Optional[Experiment]:
        """Get any active experiment for truly automagic logging."""
        if self._instrumentor:
            for exp_ref in self._instrumentor._active_experiments.values():
                experiment = exp_ref()
                if experiment:
                    return experiment
        return None

    def apply_hooks(self) -> None:
        """Apply framework-specific hooks."""
        raise NotImplementedError

    def remove_hooks(self) -> None:
        """Remove framework-specific hooks."""
        for module_name, func_path in self._patched_functions:
            try:
                module = importlib.import_module(module_name)
                original = self._original_functions.get((module_name, func_path))
                if original:
                    # Handle dotted paths for restoration
                    obj = module
                    parts = func_path.split(".")
                    for part in parts[:-1]:
                        obj = getattr(obj, part)

                    func_name = parts[-1]
                    setattr(obj, func_name, original)
            except Exception as e:
                warnings.warn(f"Failed to restore {module_name}.{func_path}: {e}", stacklevel=2)

        self._patched_functions.clear()
        self._original_functions.clear()

    def _patch_function(self, module_name: str, func_path: str, wrapper: Callable) -> None:
        """Helper to safely patch a function or method using a dotted path."""
        try:
            module = importlib.import_module(module_name)

            # Handle dotted paths for nested attributes (e.g., "Adam.step")
            obj = module
            parts = func_path.split(".")
            for part in parts[:-1]:
                obj = getattr(obj, part)

            func_name = parts[-1]
            original = getattr(obj, func_name, None)

            if original and not hasattr(original, "_tracelet_patched"):
                self._original_functions[(module_name, func_path)] = original
                wrapped = wrapper(original)
                wrapped._tracelet_patched = True
                setattr(obj, func_name, wrapped)
                self._patched_functions[(module_name, func_path)] = wrapped
        except Exception as e:
            warnings.warn(f"Failed to patch {module_name}.{func_path}: {e}", stacklevel=2)


class PyTorchHook(FrameworkHook):
    """Hook for PyTorch automatic instrumentation."""

    def apply_hooks(self) -> None:
        """Apply PyTorch-specific hooks."""
        # Patch optimizer step for automatic metric capture
        self._patch_optimizer_step()

        # Patch model checkpointing
        self._patch_model_save()

        # Patch loss computation
        self._patch_loss_functions()

    def _patch_optimizer_step(self) -> None:
        """Patch optimizer.step() to capture learning rates and gradients."""
        optimizer_step_wrapper = self._create_optimizer_step_wrapper()
        self._apply_optimizer_patches(optimizer_step_wrapper)

    def _create_optimizer_step_wrapper(self):
        """Create the optimizer step wrapper function."""
        # Capture the hook's self to use in the wrapper
        hook = self

        def optimizer_step_wrapper(original_step):
            @functools.wraps(original_step)
            def wrapped_step(optimizer_self, *args, **kwargs):
                experiment = hook.experiment

                if experiment:
                    hook._log_learning_rates(experiment, optimizer_self.param_groups)

                result = original_step(optimizer_self, *args, **kwargs)

                if experiment and hook.config.track_model_gradients:
                    hook._log_gradient_norms(experiment, optimizer_self.param_groups)

                return result

            return wrapped_step

        return optimizer_step_wrapper

    def _log_learning_rates(self, experiment: Experiment, param_groups) -> None:
        """Log learning rates for all parameter groups."""
        for i, param_group in enumerate(param_groups):
            lr = param_group.get("lr", 0)
            experiment.log_metric(f"lr_group_{i}", lr)

    def _log_gradient_norms(self, experiment: Experiment, param_groups) -> None:
        """Log gradient norms if available."""
        try:
            total_norm = 0
            for param_group in param_groups:
                for p in param_group["params"]:
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1.0 / 2)
            experiment.log_metric("gradient_norm", total_norm)
        except (AttributeError, RuntimeError) as e:
            # AttributeError: grad might not exist on some params
            # RuntimeError: grad might not be accessible in some states
            warnings.warn(f"Could not compute gradient norm: {e}", stacklevel=2)

    def _apply_optimizer_patches(self, wrapper) -> None:
        """Apply patches to common optimizers."""
        optimizers = [
            ("torch.optim", "SGD.step"),
            ("torch.optim", "Adam.step"),
            ("torch.optim", "AdamW.step"),
            ("torch.optim", "RMSprop.step"),
        ]

        for module_name, class_path in optimizers:
            try:
                self._patch_function(module_name, class_path, wrapper)
            except (ImportError, AttributeError) as e:
                # ImportError: module not found, AttributeError: class not found
                warnings.warn(f"Could not patch {module_name}.{class_path}: {e}", stacklevel=2)

    def _patch_model_save(self) -> None:
        """Patch torch.save to automatically save model checkpoints."""

        def save_wrapper(original_save):
            @functools.wraps(original_save)
            def wrapped_save(obj, f, *args, **kwargs):
                # Call original save
                result = original_save(obj, f, *args, **kwargs)

                experiment = self.experiment
                if experiment and hasattr(obj, "state_dict"):
                    # This is likely a model checkpoint
                    filename = str(f) if hasattr(f, "__str__") else "checkpoint.pth"
                    experiment.log_artifact(filename, f"Model checkpoint saved to {filename}")

                return result

            return wrapped_save

        self._patch_function("torch", "save", save_wrapper)

    def _patch_loss_functions(self) -> None:
        """Patch common loss functions to automatically log loss values."""
        # Capture the hook's self to use in the wrapper
        hook = self

        def loss_wrapper(original_loss_class):
            class WrappedLoss(original_loss_class):
                def forward(loss_self, *args, **kwargs):
                    result = super().forward(*args, **kwargs)

                    # Log loss value if it's a scalar
                    experiment = hook.experiment
                    if experiment and hasattr(result, "item"):
                        try:
                            loss_value = result.item()
                            # Use the original class name for better metric naming
                            original_class_name = original_loss_class.__name__.lower()
                            experiment.log_metric(f"{original_class_name}_loss", loss_value)
                        except (RuntimeError, ValueError) as e:
                            # RuntimeError: tensor might not be scalar
                            # ValueError: conversion issues
                            warnings.warn(f"Could not log loss value: {e}", stacklevel=2)

                    return result

            # Preserve the original class name and qualname for better debugging
            WrappedLoss.__name__ = original_loss_class.__name__
            WrappedLoss.__qualname__ = original_loss_class.__qualname__

            return WrappedLoss

        # Common loss functions
        loss_functions = [
            "torch.nn.CrossEntropyLoss",
            "torch.nn.MSELoss",
            "torch.nn.BCELoss",
            "torch.nn.L1Loss",
        ]

        for loss_name in loss_functions:
            try:
                module_name, class_name = loss_name.rsplit(".", 1)
                self._patch_function(module_name, class_name, loss_wrapper)
            except (ImportError, AttributeError, ValueError) as e:
                # ImportError: module not found, AttributeError: class not found
                # ValueError: rsplit issue with malformed loss_name
                warnings.warn(f"Could not patch {loss_name}: {e}", stacklevel=2)


class SklearnHook(FrameworkHook):
    """Hook for scikit-learn automatic instrumentation."""

    def apply_hooks(self) -> None:
        """Apply scikit-learn specific hooks."""
        self._patch_model_fit()
        self._patch_model_predict()

    def _patch_model_fit(self) -> None:
        """Patch fit methods to capture training information."""
        # Capture the hook's self to use in the wrapper
        hook = self

        def fit_wrapper(original_fit):
            @functools.wraps(original_fit)
            def wrapped_fit(estimator_self, X, y=None, **kwargs):
                experiment = hook.experiment
                if experiment:
                    hook._log_dataset_info(experiment, X)
                    hook._log_model_hyperparameters(experiment, estimator_self)

                # Call original fit
                result = original_fit(estimator_self, X, y, **kwargs)

                if experiment:
                    # Log training completion
                    experiment.log_metric("training_completed", 1)

                return result

            return wrapped_fit

        # Patch the base estimator class to cover all scikit-learn estimators
        try:
            self._patch_function("sklearn.base", "BaseEstimator.fit", fit_wrapper)
        except Exception as e:
            warnings.warn(f"Failed to patch BaseEstimator.fit: {e}", stacklevel=2)

    def _log_dataset_info(self, experiment, X) -> None:
        """Log dataset information."""
        if hasattr(X, "shape"):
            experiment.log_hyperparameter("n_samples", X.shape[0])
            experiment.log_hyperparameter("n_features", X.shape[1])

    def _log_model_hyperparameters(self, experiment, estimator_self) -> None:
        """Log model hyperparameters."""
        if hasattr(estimator_self, "get_params"):
            params = estimator_self.get_params()
            for param_name, param_value in params.items():
                try:
                    experiment.log_hyperparameter(f"sklearn_{param_name}", param_value)
                except (TypeError, ValueError):
                    # Log string representation for non-serializable parameters
                    try:
                        experiment.log_hyperparameter(f"sklearn_{param_name}", str(param_value))
                    except Exception as e:
                        warnings.warn(f"Could not log sklearn parameter '{param_name}': {e}", stacklevel=2)

    def _patch_model_predict(self) -> None:
        """Patch predict methods to capture inference information."""
        # Capture the hook's self to use in the wrapper
        hook = self

        def predict_wrapper(original_predict):
            @functools.wraps(original_predict)
            def wrapped_predict(estimator_self, X, **kwargs):
                experiment = hook.experiment
                if experiment and hasattr(X, "shape"):
                    experiment.log_metric("inference_samples", X.shape[0])

                return original_predict(estimator_self, X, **kwargs)

            return wrapped_predict

        # Patch the base estimator class to cover all scikit-learn estimators
        try:
            self._patch_function("sklearn.base", "BaseEstimator.predict", predict_wrapper)
        except Exception as e:
            warnings.warn(f"Failed to patch BaseEstimator.predict: {e}", stacklevel=2)


class XGBoostHook(FrameworkHook):
    """Hook for XGBoost automatic instrumentation."""

    def apply_hooks(self) -> None:
        """Apply XGBoost specific hooks."""
        self._patch_xgboost_train()

    def _patch_xgboost_train(self) -> None:
        """Patch XGBoost training to capture metrics."""

        def train_wrapper(original_train):
            @functools.wraps(original_train)
            def wrapped_train(params, dtrain, *args, **kwargs):
                experiment = self.experiment
                if experiment:
                    # Log XGBoost parameters
                    for param_name, param_value in params.items():
                        experiment.log_hyperparameter(f"xgb_{param_name}", param_value)

                return original_train(params, dtrain, *args, **kwargs)

            return wrapped_train

        self._patch_function("xgboost", "train", train_wrapper)


class LightningHook(FrameworkHook):
    """Hook for PyTorch Lightning automatic instrumentation."""

    def apply_hooks(self) -> None:
        """Apply Lightning-specific hooks."""
        self._patch_lightning_log()

    def _patch_lightning_log(self) -> None:
        """Patch Lightning's self.log() to capture metrics."""

        def log_wrapper(original_log):
            @functools.wraps(original_log)
            def wrapped_log(module_self, name: str, value, *args, **kwargs):
                # Call original logging method
                result = original_log(module_self, name, value, *args, **kwargs)

                # Get any active experiment for truly automagic logging
                experiment = self._get_any_active_experiment()

                # Check if value is numeric (int, float, or tensor with item())
                numeric_value = None
                try:
                    if isinstance(value, (int, float)):
                        numeric_value = float(value)
                    elif hasattr(value, "item"):  # PyTorch tensor
                        numeric_value = float(value.item())
                    elif hasattr(value, "__float__"):  # Other numeric types
                        numeric_value = float(value)
                except (ValueError, TypeError):
                    numeric_value = None

                if experiment and numeric_value is not None:
                    # Get the current step from trainer if available
                    current_step = 0
                    try:
                        if hasattr(module_self, "trainer") and module_self.trainer:
                            current_step = getattr(module_self.trainer, "global_step", 0)
                    except (RuntimeError, AttributeError):
                        # Lightning module not attached to a trainer yet
                        current_step = 0

                    # Log to our experiment
                    experiment.log_metric(name, numeric_value, current_step)

                return result

            return wrapped_log

        # Patch LightningModule.log
        self._patch_function("pytorch_lightning.core.module", "LightningModule.log", log_wrapper)


class FrameworkHookRegistry:
    """Registry for managing framework-specific hooks."""

    def __init__(self, config):
        self.config = config
        self._active_hooks: dict[str, set[FrameworkHook]] = {}
        self._lock = threading.RLock()

        # Registry of available hooks
        self._hook_classes = {
            "pytorch": PyTorchHook,
            "sklearn": SklearnHook,
            "xgboost": XGBoostHook,
            "lightning": LightningHook,
        }

    def is_available(self, framework: str) -> bool:
        """Check if a framework is available."""
        if framework == "pytorch":
            return importlib.util.find_spec("torch") is not None
        elif framework == "sklearn":
            return importlib.util.find_spec("sklearn") is not None
        elif framework == "xgboost":
            return importlib.util.find_spec("xgboost") is not None
        elif framework == "lightning":
            return importlib.util.find_spec("pytorch_lightning") is not None
        return False

    def apply_hooks(self, framework: str, experiment: Experiment) -> None:
        """Apply hooks for a specific framework."""
        if framework not in self._hook_classes:
            return

        if not self.is_available(framework):
            return

        with self._lock:
            # Create and apply hook
            hook_class = self._hook_classes[framework]
            hook = hook_class(experiment)
            hook.apply_hooks()

            # Track active hook
            if framework not in self._active_hooks:
                self._active_hooks[framework] = set()
            self._active_hooks[framework].add(hook)

    def remove_hooks(self, framework: str) -> None:
        """Remove hooks for a specific framework."""
        with self._lock:
            if framework in self._active_hooks:
                for hook in list(self._active_hooks[framework]):
                    hook.remove_hooks()
                del self._active_hooks[framework]

    def cleanup(self) -> None:
        """Remove all hooks."""
        with self._lock:
            for framework in list(self._active_hooks.keys()):
                self.remove_hooks(framework)
