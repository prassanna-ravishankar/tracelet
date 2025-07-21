from typing import Any, Optional

from .core.experiment import Experiment, ExperimentConfig
from .frameworks.lightning import LightningFramework
from .frameworks.pytorch import PyTorchFramework
from .settings import TraceletSettings

_active_experiment: Optional[Experiment] = None
_settings: Optional[TraceletSettings] = None


def start_logging(
    exp_name: Optional[str] = None,
    project: Optional[str] = None,
    backend: Optional[str] = None,
    api_key: Optional[str] = None,
    backend_url: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
) -> Experiment:
    """Start logging metrics and metadata for your ML experiment.

    Args:
        exp_name: Name of the experiment. If not provided, uses TRACELET_EXPERIMENT_NAME env var
        project: Project name. If not provided, uses TRACELET_PROJECT env var
        backend: Backend to use ("mlflow", "wandb", "aim"). If not provided, uses TRACELET_BACKEND env var
        api_key: API key for the backend. If not provided, uses TRACELET_API_KEY env var
        backend_url: Backend URL. If not provided, uses TRACELET_BACKEND_URL env var
        config: Additional configuration to override defaults and env vars

    Returns:
        Experiment: The active experiment instance

    Example:
        ```python
        import tracelet

        # Start logging with minimal config
        tracelet.start_logging("my_experiment")

        # Or with more configuration
        tracelet.start_logging(
            exp_name="my_experiment",
            project="my_project",
            backend="wandb",
            api_key="...",
        )
        ```
    """
    global _active_experiment, _settings

    # Stop any existing experiment first
    if _active_experiment:
        _active_experiment.stop()
        _active_experiment = None

    # Initialize settings - let TraceletSettings handle env vars
    settings_dict = {}
    if exp_name is not None:
        settings_dict["experiment_name"] = exp_name
    if project is not None:
        settings_dict["project_name"] = project
    if backend is not None:
        settings_dict["backend"] = backend
    if api_key is not None:
        settings_dict["api_key"] = api_key
    if backend_url is not None:
        settings_dict["backend_url"] = backend_url
    if config:
        settings_dict.update(config)

    _settings = TraceletSettings(**settings_dict)

    # Create experiment config
    exp_config = ExperimentConfig(
        track_metrics=True,
        track_environment=_settings.track_env,
        track_args=True,
        track_stdout=True,
        track_checkpoints=True,
        track_system_metrics=_settings.track_system_metrics,
        track_git=_settings.track_git,
    )

    # Create experiment
    _active_experiment = Experiment(
        name=_settings.experiment_name or "default_experiment",
        config=exp_config,
        backend=_settings.backend,
        tags=[f"project:{_settings.project_name}"],
    )

    # Initialize frameworks based on settings
    if _settings.track_tensorboard:
        pytorch = PyTorchFramework(patch_tensorboard=True)
        _active_experiment._framework = pytorch
        pytorch.initialize(_active_experiment)

    if _settings.track_lightning:
        lightning = LightningFramework()
        lightning.initialize(_active_experiment)

    # Start tracking
    _active_experiment.start()

    return _active_experiment


def get_active_experiment() -> Optional[Experiment]:
    """Get the currently active experiment"""
    return _active_experiment


def stop_logging():
    """Stop the active experiment and cleanup"""
    global _active_experiment
    if _active_experiment:
        _active_experiment.stop()
        _active_experiment = None
