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
        for module_name, func_name in self._patched_functions:
            try:
                module = importlib.import_module(module_name)
                original = self._original_functions.get((module_name, func_name))
                if original:
                    setattr(module, func_name, original)
            except Exception as e:
                warnings.warn(f"Failed to restore {module_name}.{func_name}: {e}", stacklevel=2)

        self._patched_functions.clear()
        self._original_functions.clear()

    def _patch_function(self, module_name: str, func_name: str, wrapper: Callable) -> None:
        """Helper to safely patch a function."""
        try:
            module = importlib.import_module(module_name)
            original = getattr(module, func_name, None)
            if original and not hasattr(original, "_tracelet_patched"):
                self._original_functions[(module_name, func_name)] = original
                wrapped = wrapper(original)
                wrapped._tracelet_patched = True
                setattr(module, func_name, wrapped)
                self._patched_functions[(module_name, func_name)] = wrapped
        except Exception as e:
            warnings.warn(f"Failed to patch {module_name}.{func_name}: {e}", stacklevel=2)


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

    def _patch_optimizer_step(self) -> None:  # noqa: C901
        """Patch optimizer.step() to capture learning rates and gradients."""

        def optimizer_step_wrapper(original_step):
            @functools.wraps(original_step)
            def wrapped_step(self, *args, **kwargs):
                # Get ANY active experiment (truly automagic - no need to specify which one)
                experiment = self._get_any_active_experiment()
                if experiment:
                    # AUTOMAGIC: Log learning rate without user intervention
                    for i, param_group in enumerate(self.param_groups):
                        lr = param_group.get("lr", 0)
                        experiment.log_metric(f"lr_group_{i}", lr)

                # Call original step
                result = original_step(self, *args, **kwargs)

                if experiment:
                    # AUTOMAGIC: Log gradient norms if enabled
                    try:
                        total_norm = 0
                        for param_group in self.param_groups:
                            for p in param_group["params"]:
                                if p.grad is not None:
                                    param_norm = p.grad.data.norm(2)
                                    total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** (1.0 / 2)
                        experiment.log_metric("gradient_norm", total_norm)
                    except Exception:  # noqa: S110
                        pass  # Don't break training if gradient logging fails

                return result

            return wrapped_step

        # Patch common optimizers
        optimizers = [
            "torch.optim.SGD",
            "torch.optim.Adam",
            "torch.optim.AdamW",
            "torch.optim.RMSprop",
        ]

        for opt_name in optimizers:
            try:
                module_name, class_name = opt_name.rsplit(".", 1)
                self._patch_function(module_name, class_name + ".step", optimizer_step_wrapper)
            except Exception:  # noqa: S110
                pass  # Ignore if optimizer not available

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

        def loss_wrapper(original_loss_class):
            class WrappedLoss(original_loss_class):
                def forward(self, *args, **kwargs):
                    result = super().forward(*args, **kwargs)

                    # Log loss value if it's a scalar
                    experiment = self.experiment
                    if experiment and hasattr(result, "item"):
                        try:
                            loss_value = result.item()
                            experiment.log_metric(f"{self.__class__.__name__.lower()}_loss", loss_value)
                        except Exception:  # noqa: S110
                            pass  # Ignore if loss can't be logged

                    return result

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
            except Exception:  # noqa: S110
                pass  # Ignore if loss function not available

    def _log_gradient_norms(self, experiment: Experiment, optimizer) -> None:
        """Log gradient norms for debugging."""
        total_norm = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        experiment.log_metric("gradient_norm", total_norm)


class SklearnHook(FrameworkHook):
    """Hook for scikit-learn automatic instrumentation."""

    def apply_hooks(self) -> None:
        """Apply scikit-learn specific hooks."""
        self._patch_model_fit()
        self._patch_model_predict()

    def _patch_model_fit(self) -> None:  # noqa: C901
        """Patch fit methods to capture training information."""

        def fit_wrapper(original_fit):
            @functools.wraps(original_fit)
            def wrapped_fit(self, X, y=None, **kwargs):
                experiment = self.experiment
                if experiment:
                    # Log dataset information
                    if hasattr(X, "shape"):
                        experiment.log_hyperparameter("n_samples", X.shape[0])
                        experiment.log_hyperparameter("n_features", X.shape[1])

                    # Log model hyperparameters
                    if hasattr(self, "get_params"):
                        params = self.get_params()
                        for param_name, param_value in params.items():
                            try:  # noqa: SIM105
                                experiment.log_hyperparameter(f"sklearn_{param_name}", param_value)
                            except Exception:  # noqa: S110
                                pass  # Ignore serialization errors for complex objects

                # Call original fit
                result = original_fit(self, X, y, **kwargs)

                if experiment:
                    # Log training completion
                    experiment.log_metric("training_completed", 1)

                return result

            return wrapped_fit

        # Common sklearn estimators - this is a simplified approach
        # In practice, you'd want to patch the BaseEstimator class
        common_estimators = [
            "sklearn.ensemble.RandomForestClassifier",
            "sklearn.ensemble.RandomForestRegressor",
            "sklearn.linear_model.LogisticRegression",
            "sklearn.linear_model.LinearRegression",
            "sklearn.svm.SVC",
            "sklearn.svm.SVR",
        ]

        for estimator_name in common_estimators:
            try:
                module_name, class_name = estimator_name.rsplit(".", 1)
                self._patch_function(module_name, f"{class_name}.fit", fit_wrapper)
            except Exception:  # noqa: S110
                pass  # Ignore if module not available

    def _patch_model_predict(self) -> None:
        """Patch predict methods to capture inference information."""

        def predict_wrapper(original_predict):
            @functools.wraps(original_predict)
            def wrapped_predict(self, X, **kwargs):
                experiment = self.experiment
                if experiment and hasattr(X, "shape"):
                    experiment.log_metric("inference_samples", X.shape[0])

                return original_predict(self, X, **kwargs)

            return wrapped_predict

        # Similar patching as fit method
        # This is simplified - in practice you'd patch BaseEstimator


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
        }

    def is_available(self, framework: str) -> bool:
        """Check if a framework is available."""
        if framework == "pytorch":
            return importlib.util.find_spec("torch") is not None
        elif framework == "sklearn":
            return importlib.util.find_spec("sklearn") is not None
        elif framework == "xgboost":
            return importlib.util.find_spec("xgboost") is not None
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
