"""Weights & Biases backend plugin for Tracelet."""

import logging
from typing import Any, Optional

try:
    import wandb

    _has_wandb = True
except ImportError:
    _has_wandb = False
    wandb = None

from tracelet.core.orchestrator import MetricData, MetricType
from tracelet.core.plugins import BackendPlugin, PluginMetadata, PluginType

logger = logging.getLogger(__name__)

# Error messages
WANDB_NOT_INSTALLED_MSG = "Weights & Biases is not installed. Install with: pip install wandb"
WANDB_NOT_AVAILABLE_MSG = "W&B not available"


class WandbBackend(BackendPlugin):
    """Weights & Biases backend plugin for experiment tracking with free tier support."""

    def __init__(self):
        super().__init__()
        self._run: Optional[wandb.sdk.wandb_run.Run] = None
        self._project_name = "tracelet-experiments"
        self._experiment_name = "experiment"
        self._entity: Optional[str] = None
        self._tags: list[str] = []
        self._job_type = "train"

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="wandb",
            version="1.0.0",
            type=PluginType.BACKEND,
            description="Weights & Biases backend for experiment tracking with free tier support",
            author="Tracelet Team",
            dependencies=[],
            capabilities={
                "metrics",
                "parameters",
                "artifacts",
                "logging",
                "model_registry",
                "hyperparameters",
                "enhanced_visualizations",
                "search",
                "comparison",
            },
        )

    def initialize(self, config: dict[str, Any]):
        """Initialize W&B backend."""
        if not _has_wandb:
            raise ImportError(WANDB_NOT_INSTALLED_MSG)

        self._config = config
        self._project_name = config.get("project_name", "tracelet-experiments")
        self._experiment_name = config.get("experiment_name", "experiment")
        self._entity = config.get("entity")
        self._tags = config.get("tags", [])
        self._job_type = config.get("job_type", "train")

        logger.info(f"Initialized W&B backend for project '{self._project_name}'")

    def start(self):
        """Start the W&B backend."""
        if not _has_wandb:
            raise RuntimeError(WANDB_NOT_AVAILABLE_MSG)

        try:
            # Initialize W&B run
            self._run = wandb.init(
                project=self._project_name,
                name=self._experiment_name,
                entity=self._entity,
                tags=self._tags,
                job_type=self._job_type,
                reinit=True,  # Allow re-initialization
                settings=wandb.Settings(
                    # Configure for clean integration
                    _disable_stats=self._config.get("disable_stats", True),
                    console="off" if self._config.get("quiet", True) else "on",
                ),
            )

            logger.info(f"Started W&B run: {self._run.id} ({self._run.name})")
        except Exception:
            logger.exception("Failed to start W&B run")
            raise

        self._active = True

    def stop(self):
        """Stop the W&B backend."""
        if self._run:
            try:
                # Finish the run
                self._run.finish()
                logger.info(f"Stopped W&B run: {self._run.id}")
            except Exception:
                logger.exception("Error stopping W&B run")

        self._run = None
        self._active = False

    def get_status(self) -> dict[str, Any]:
        """Get backend status."""
        status = {
            "active": self._active,
            "project_name": self._project_name,
            "experiment_name": self._experiment_name,
            "entity": self._entity,
            "job_type": self._job_type,
        }

        if self._run:
            try:
                status.update({
                    "run_id": self._run.id,
                    "run_name": self._run.name,
                    "run_url": self._run.url,
                    "state": self._run.state,
                    "tags": self._run.tags,
                })
            except Exception:
                logger.exception("Error getting run status")

        return status

    def receive_metric(self, metric: MetricData):  # noqa: C901
        """Receive and process a metric from the orchestrator."""
        if not self._active or not self._run:
            return

        try:
            if metric.type == MetricType.SCALAR:
                self._log_scalar_metric(metric)
            elif metric.type == MetricType.PARAMETER:
                self._log_parameter(metric)
            elif metric.type == MetricType.ARTIFACT:
                self._log_artifact(metric)
            elif metric.type == MetricType.HISTOGRAM:
                self._log_histogram_metric(metric)
            elif metric.type == MetricType.IMAGE:
                self._log_image_metric(metric)
            elif metric.type == MetricType.TEXT:
                self._log_text_metric(metric)
            elif metric.type == MetricType.FIGURE:
                self._log_figure_metric(metric)
            elif metric.type == MetricType.HPARAMS:
                self._log_hparams_metric(metric)
            else:
                # Log as custom metric
                self._log_custom_metric(metric)
        except Exception:
            logger.exception(f"Failed to log metric '{metric.name}' to W&B")

    def _log_scalar_metric(self, metric: MetricData):
        """Log a scalar metric to W&B."""
        log_data = {metric.name: float(metric.value)}

        # Add source prefix if available
        if metric.source and metric.source != "experiment":
            log_data = {f"{metric.source}_{metric.name}": float(metric.value)}

        self._run.log(log_data, step=metric.iteration)

    def _log_parameter(self, metric: MetricData):
        """Log a parameter to W&B."""
        if not self._run:
            return

        param_name = metric.name
        if metric.source and metric.source != "experiment":
            param_name = f"{metric.source}_{metric.name}"

        # Update run config with parameter
        self._run.config[param_name] = metric.value

    def _log_artifact(self, metric: MetricData):
        """Log an artifact to W&B."""
        if not self._run:
            return

        # For artifacts, metric.value should be the file path
        file_path = metric.value
        artifact_name = metric.metadata.get("artifact_name", metric.name) if metric.metadata else metric.name

        try:
            # Create artifact
            artifact = wandb.Artifact(
                name=artifact_name, type=metric.metadata.get("artifact_type", "file") if metric.metadata else "file"
            )

            # Add file to artifact
            artifact.add_file(file_path)

            # Log artifact
            self._run.log_artifact(artifact)
        except Exception:
            logger.exception(f"Failed to log artifact '{metric.name}'")

    def _log_histogram_metric(self, metric: MetricData):
        """Log a histogram metric to W&B."""
        # W&B can handle histograms using wandb.Histogram
        histogram_data = wandb.Histogram(metric.value)

        log_data = {metric.name: histogram_data}
        self._run.log(log_data, step=metric.iteration)

    def _log_image_metric(self, metric: MetricData):
        """Log an image metric to W&B."""
        # W&B can handle various image formats
        image_data = wandb.Image(metric.value, caption=metric.metadata.get("caption") if metric.metadata else None)

        log_data = {metric.name: image_data}
        self._run.log(log_data, step=metric.iteration)

    def _log_text_metric(self, metric: MetricData):
        """Log a text metric to W&B."""
        # Log text as a simple string metric
        log_data = {f"{metric.name}_text": str(metric.value)}
        self._run.log(log_data, step=metric.iteration)

    def _log_figure_metric(self, metric: MetricData):
        """Log a matplotlib figure to W&B."""
        # W&B can handle matplotlib figures directly
        image_data = wandb.Image(metric.value)

        log_data = {metric.name: image_data}
        self._run.log(log_data, step=metric.iteration)

    def _log_hparams_metric(self, metric: MetricData):
        """Log hyperparameters to W&B."""
        if not self._run:
            return

        # W&B handles hyperparameters via run.config
        if isinstance(metric.value, dict) and "hparams" in metric.value:
            hparams = metric.value["hparams"]
            for key, value in hparams.items():
                self._run.config[key] = value

    def _log_custom_metric(self, metric: MetricData):
        """Log a custom metric to W&B."""
        # Log as scalar if numeric, otherwise as text
        if isinstance(metric.value, (int, float)):
            self._log_scalar_metric(metric)
        else:
            # Log as text
            log_data = {f"{metric.name}_custom": str(metric.value)}
            self._run.log(log_data, step=metric.iteration)

    # BackendInterface implementation
    def log_metric(self, name: str, value: Any, iteration: int):
        """Log a metric to W&B."""
        if not self._active:
            return

        metric = MetricData(
            name=name,
            value=value,
            type=MetricType.SCALAR if isinstance(value, (int, float)) else MetricType.CUSTOM,
            iteration=iteration,
        )
        self.receive_metric(metric)

    def log_params(self, params: dict[str, Any]):
        """Log parameters to W&B."""
        if not self._active or not self._run:
            return

        try:
            # Update run config with all parameters
            for name, value in params.items():
                self._run.config[name] = value
        except Exception:
            logger.exception("Failed to log parameters to W&B")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Upload an artifact to W&B."""
        if not self._active or not self._run:
            return

        try:
            artifact_name = artifact_path or local_path.split("/")[-1]

            # Create artifact
            artifact = wandb.Artifact(name=artifact_name, type="file")
            artifact.add_file(local_path)

            # Log artifact
            self._run.log_artifact(artifact)
        except Exception:
            logger.exception(f"Failed to log artifact '{local_path}'")

    def save_experiment(self, experiment_data: dict[str, Any]):  # noqa: C901
        """Save experiment metadata to W&B."""
        if not self._active or not self._run:
            return

        try:
            # Add experiment metadata to config and tags
            tags_to_add = []

            # Process git information
            if "git" in experiment_data:
                git_info = experiment_data["git"]
                if isinstance(git_info, dict):
                    # Add git hash as tag if available
                    if "commit_hash" in git_info:
                        tags_to_add.append(f"git:{git_info['commit_hash'][:8]}")

                    # Add git info to config
                    for key, value in git_info.items():
                        if isinstance(value, (str, bool, int, float)):
                            self._run.config[f"git_{key}"] = value

            # Process system information
            if "system" in experiment_data:
                system_info = experiment_data["system"]
                if isinstance(system_info, dict):
                    # Add Python version as tag if available
                    if "python_version" in system_info:
                        tags_to_add.append(f"python:{system_info['python_version']}")

                    # Add system info to config
                    for key, value in system_info.items():
                        if isinstance(value, (str, bool, int, float)):
                            self._run.config[f"system_{key}"] = value

            # Process environment information
            if "environment" in experiment_data:
                env_info = experiment_data["environment"]
                if isinstance(env_info, dict):
                    # Add environment info to config
                    for key, value in env_info.items():
                        if isinstance(value, (str, bool, int, float)):
                            self._run.config[f"env_{key}"] = value

            # Store additional tags in config (W&B doesn't allow modifying tags after creation)
            if tags_to_add:
                self._run.config["additional_tags"] = tags_to_add

            # Set run description/notes
            if "description" in experiment_data:
                self._run.notes = experiment_data["description"]

        except Exception:
            logger.exception("Failed to save experiment metadata to W&B")
