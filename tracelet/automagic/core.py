"""
Core automagic instrumentation system.
"""

import inspect
import logging
import threading
import weakref
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.experiment import Experiment

logger = logging.getLogger(__name__)


@dataclass
class AutomagicConfig:
    """Configuration for automagic instrumentation."""

    # Hyperparameter detection
    detect_function_args: bool = True
    detect_class_attributes: bool = True
    detect_argparse: bool = True
    detect_config_files: bool = True

    # Model tracking
    track_model_architecture: bool = True
    track_model_checkpoints: bool = True
    track_model_gradients: bool = False  # Can be expensive

    # Dataset tracking
    track_dataset_info: bool = True
    track_data_samples: bool = False  # Privacy-sensitive

    # Training monitoring
    monitor_training_loop: bool = True
    monitor_loss_curves: bool = True
    monitor_learning_rate: bool = True
    epoch_detection_threshold_seconds: float = 30.0  # Time between validation metrics to count as new epoch

    # Resource monitoring
    monitor_gpu_memory: bool = True
    monitor_cpu_usage: bool = True

    # Code versioning
    track_git_info: bool = True
    track_file_hashes: bool = True

    # Framework-specific
    frameworks: set[str] = field(default_factory=lambda: {"pytorch", "lightning", "sklearn", "keras", "xgboost"})


class AutomagicInstrumentor:
    """
    Main automagic instrumentation orchestrator.

    Automatically captures ML experiment information without explicit logging calls.
    Uses a combination of:
    - Frame inspection for hyperparameters
    - Monkey-patching for framework integration
    - Weak references to avoid memory leaks
    - Thread-safe operation
    """

    _instance: Optional["AutomagicInstrumentor"] = None
    _lock = threading.RLock()

    def __init__(self, config: Optional[AutomagicConfig] = None):
        if AutomagicInstrumentor._instance is not None:
            raise RuntimeError("AutomagicInstrumentor is a singleton. Use get_instance().")

        self.config = config or AutomagicConfig()
        self._active_experiments: dict[str, weakref.ref] = {}
        self._patched_modules: set[str] = set()
        self._original_functions: dict[str, Any] = {}
        self._thread_local = threading.local()

        # Initialize detectors
        from .detectors import DatasetDetector, HyperparameterDetector, ModelDetector
        from .hooks import FrameworkHookRegistry
        from .monitors import ResourceMonitor, TrainingMonitor

        self.hyperparam_detector = HyperparameterDetector(self.config)
        self.model_detector = ModelDetector(self.config)
        self.dataset_detector = DatasetDetector(self.config)
        self.training_monitor = TrainingMonitor(self.config)
        self.resource_monitor = ResourceMonitor(self.config)
        self.hook_registry = FrameworkHookRegistry(self.config)

        AutomagicInstrumentor._instance = self

    @classmethod
    def get_instance(cls, config: Optional[AutomagicConfig] = None) -> "AutomagicInstrumentor":
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
            elif config is not None:
                # Check if provided config differs from existing singleton's config
                existing_config = cls._instance.config
                if cls._configs_differ(existing_config, config):
                    import warnings

                    warnings.warn(
                        "AutomagicInstrumentor singleton already exists with different configuration. "
                        "The existing configuration will be used. To use a new configuration, "
                        "call cleanup() on the existing instance first.",
                        UserWarning,
                        stacklevel=2,
                    )
            return cls._instance

    @classmethod
    def _configs_differ(cls, config1: AutomagicConfig, config2: AutomagicConfig) -> bool:
        """Check if two AutomagicConfig instances have different settings."""
        # Compare all significant configuration fields
        fields_to_compare = [
            "detect_function_args",
            "detect_class_attributes",
            "detect_argparse",
            "detect_config_files",
            "track_model_architecture",
            "track_model_checkpoints",
            "track_model_gradients",
            "track_dataset_info",
            "track_data_samples",
            "monitor_training_loop",
            "monitor_loss_curves",
            "monitor_learning_rate",
            "epoch_detection_threshold_seconds",
            "monitor_gpu_memory",
            "monitor_cpu_usage",
            "track_git_info",
            "track_file_hashes",
            "frameworks",
        ]

        return any(
            getattr(config1, field_name, None) != getattr(config2, field_name, None) for field_name in fields_to_compare
        )

    def attach_experiment(self, experiment: Experiment) -> None:
        """Attach automagic instrumentation to an experiment."""
        with self._lock:
            # Store weak reference to avoid circular dependencies
            self._active_experiments[experiment.id] = weakref.ref(experiment)

            # TRULY AUTOMAGIC: Immediately capture hyperparameters from calling context
            self._auto_capture_hyperparameters(experiment)

            # Apply framework-specific hooks immediately
            self._apply_framework_hooks(experiment)

            # Start continuous monitoring
            if self.config.monitor_training_loop:
                self.training_monitor.start(experiment)

            if self.config.monitor_gpu_memory or self.config.monitor_cpu_usage:
                self.resource_monitor.start(experiment)

    def detach_experiment(self, experiment_id: str) -> None:
        """Detach automagic instrumentation from an experiment."""
        with self._lock:
            if experiment_id in self._active_experiments:
                del self._active_experiments[experiment_id]

            # Stop monitors
            self.training_monitor.stop(experiment_id)
            self.resource_monitor.stop(experiment_id)

            # Remove framework hooks if no more experiments
            if not self._active_experiments:
                self._remove_framework_hooks()

    def capture_hyperparameters(self, experiment: Experiment, frame_depth: int = 2) -> dict[str, Any]:
        """Capture hyperparameters from calling context."""
        frame = inspect.currentframe()
        try:
            for _ in range(frame_depth):
                frame = frame.f_back
                if frame is None:
                    return {}

            # Extract hyperparameters from frame
            hyperparams = self.hyperparam_detector.extract_from_frame(frame)

            # Log to experiment
            for name, value in hyperparams.items():
                experiment.log_hyperparameter(name, value)

            return hyperparams

        finally:
            del frame

    def capture_model_info(self, model: Any, experiment: Experiment) -> dict[str, Any]:
        """Capture model architecture and metadata."""
        model_info = self.model_detector.analyze_model(model)

        # Log model information
        experiment.log_hyperparameter("model_type", model_info.get("type"))
        experiment.log_hyperparameter("model_parameters", model_info.get("parameter_count"))

        if self.config.track_model_architecture:
            experiment.log_artifact("model_architecture.txt", str(model_info.get("architecture", "")))

        return model_info

    def capture_dataset_info(self, dataset: Any, experiment: Experiment) -> dict[str, Any]:
        """Capture dataset information."""
        dataset_info = self.dataset_detector.analyze_dataset(dataset)

        # Log dataset metadata
        for key, value in dataset_info.items():
            if key not in ["samples"]:  # Don't log actual samples
                experiment.log_hyperparameter(f"dataset_{key}", value)

        return dataset_info

    def _auto_capture_hyperparameters(self, experiment: Experiment) -> None:
        """Automatically capture hyperparameters from calling context immediately."""
        if not self.config.detect_function_args:
            return

        # Walk up the call stack to find the user's code (skip tracelet internal frames)
        frame = inspect.currentframe()
        try:
            # Skip through tracelet internal frames to find user code
            user_frame = self._find_user_frame(frame)
            if user_frame:
                # Extract hyperparameters from user's frame
                hyperparams = self.hyperparam_detector.extract_from_frame(user_frame)

                # Automatically log them
                for name, value in hyperparams.items():
                    experiment.log_hyperparameter(name, value)

                if hyperparams:
                    logger.info(f"Automagically captured hyperparameters: {list(hyperparams.keys())}")
        finally:
            del frame

    def _find_user_frame(self, start_frame):
        """Find the first frame that's not from tracelet internals.

        This method walks up the call stack to find user code, skipping all
        tracelet internal frames. It uses module names for robust detection
        that works across all installation methods and environments.
        """
        frame = start_frame

        while frame:
            # Get the module name for this frame
            module_name = frame.f_globals.get("__name__", "")
            filename = frame.f_code.co_filename

            # Primary strategy: Use module name hierarchy
            # This is the most reliable method as it doesn't depend on file paths
            is_tracelet_internal = module_name.startswith("tracelet.") or module_name == "tracelet"

            # Additional check for cases where __name__ might not be set correctly
            # or for dynamically created modules
            if not is_tracelet_internal and module_name in ("", "__main__"):
                # Fallback: Check if the filename suggests tracelet internals
                # This handles edge cases like dynamic imports or eval'd code
                is_tracelet_internal = (
                    "/tracelet/" in filename
                    or "\\tracelet\\" in filename
                    or filename.endswith("tracelet/__init__.py")
                    or filename.endswith("tracelet\\__init__.py")
                ) and (
                    # Ensure it's actually from a tracelet installation, not user code
                    "site-packages" in filename
                    or "dist-packages" in filename
                    or
                    # Development installation patterns
                    "/tracelet/tracelet/" in filename
                    or "\\tracelet\\tracelet\\" in filename
                )

            if not is_tracelet_internal:
                return frame
            frame = frame.f_back

        return None

    def _apply_framework_hooks(self, experiment: Experiment) -> None:
        """Apply framework-specific hooks."""
        for framework in self.config.frameworks:
            if self.hook_registry.is_available(framework):
                self.hook_registry.apply_hooks(framework, experiment)
                self._patched_modules.add(framework)

    def _remove_framework_hooks(self) -> None:
        """Remove all framework hooks."""
        for framework in list(self._patched_modules):
            self.hook_registry.remove_hooks(framework)
            self._patched_modules.discard(framework)

    def cleanup(self) -> None:
        """Clean up all instrumentation."""
        with self._lock:
            # Stop all monitors
            self.training_monitor.cleanup()
            self.resource_monitor.cleanup()

            # Remove hooks
            self._remove_framework_hooks()

            # Clear experiments
            self._active_experiments.clear()

            # Reset singleton
            AutomagicInstrumentor._instance = None


# Convenience functions for easy integration
def automagic(experiment: Experiment, config: Optional[AutomagicConfig] = None) -> AutomagicInstrumentor:
    """Convenience function to enable automagic instrumentation for an experiment."""
    instrumentor = AutomagicInstrumentor.get_instance(config)
    instrumentor.attach_experiment(experiment)
    return instrumentor


def capture_hyperparams(experiment: Experiment) -> dict[str, Any]:
    """Convenience function to capture hyperparameters from calling context."""
    instrumentor = AutomagicInstrumentor.get_instance()
    return instrumentor.capture_hyperparameters(experiment, frame_depth=2)
