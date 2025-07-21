"""ClearML backend plugin for Tracelet."""

import logging
from typing import Any, Optional

try:
    from clearml import Logger, Task

    _has_clearml = True
except ImportError:
    _has_clearml = False
    Task = Logger = None

from tracelet.core.orchestrator import MetricData, MetricType
from tracelet.core.plugins import BackendPlugin, PluginMetadata, PluginType

logger = logging.getLogger(__name__)

# Error messages
CLEARML_NOT_INSTALLED_MSG = "ClearML is not installed. Install with: pip install clearml"
CLEARML_NOT_AVAILABLE_MSG = "ClearML not available"


class ClearMLBackend(BackendPlugin):
    """ClearML backend plugin for experiment tracking with free SaaS platform integration."""

    def __init__(self):
        super().__init__()
        self._task: Optional[Task] = None
        self._logger: Optional[Logger] = None
        self._project_name = "Tracelet Experiments"
        self._task_name = "experiment"
        self._task_type = "training"
        self._tags: list[str] = []

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="clearml",
            version="1.0.0",
            type=PluginType.BACKEND,
            description="ClearML backend for experiment tracking with free SaaS platform integration",
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
            },
        )

    def initialize(self, config: dict[str, Any]):
        """Initialize ClearML backend."""
        if not _has_clearml:
            raise ImportError(CLEARML_NOT_INSTALLED_MSG)

        self._config = config
        self._project_name = config.get("project_name", "Tracelet Experiments")
        self._task_name = config.get("task_name", "experiment")
        self._task_type = config.get("task_type", "training")
        self._tags = config.get("tags", [])

        logger.info(f"Initialized ClearML backend for project '{self._project_name}'")

    def start(self):
        """Start the ClearML backend."""
        if not _has_clearml:
            raise RuntimeError(CLEARML_NOT_AVAILABLE_MSG)

        try:
            # Initialize ClearML task
            self._task = Task.init(
                project_name=self._project_name,
                task_name=self._task_name,
                task_type=self._task_type,
                tags=self._tags,
                # Don't automatically capture args/environment for cleaner integration
                auto_connect_arg_parser=self._config.get("auto_connect_arg_parser", False),
                auto_connect_frameworks=self._config.get(
                    "auto_connect_frameworks", {"matplotlib": False, "tensorboard": False}
                ),
            )

            # Get logger for metrics
            self._logger = self._task.get_logger()

            logger.info(f"Started ClearML task: {self._task.id} ({self._task.name})")
        except Exception:
            logger.exception("Failed to start ClearML task")
            raise

        self._active = True

    def stop(self):
        """Stop the ClearML backend."""
        if self._task:
            try:
                # Mark task as completed
                self._task.close()
                logger.info(f"Stopped ClearML task: {self._task.id}")
            except Exception:
                logger.exception("Error stopping ClearML task")

        self._task = None
        self._logger = None
        self._active = False

    def get_status(self) -> dict[str, Any]:
        """Get backend status."""
        status = {
            "active": self._active,
            "project_name": self._project_name,
            "task_name": self._task_name,
            "task_type": self._task_type,
        }

        if self._task:
            try:
                status.update({
                    "task_id": self._task.id,
                    "task_url": self._task.get_output_log_web_page(),
                    "status": self._task.get_status(),
                    "tags": self._task.get_tags(),
                })
            except Exception:
                logger.exception("Error getting task status")

        return status

    def receive_metric(self, metric: MetricData):  # noqa: C901
        """Receive and process a metric from the orchestrator."""
        if not self._active or not self._logger:
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
            logger.exception(f"Failed to log metric '{metric.name}' to ClearML")

    def _log_scalar_metric(self, metric: MetricData):
        """Log a scalar metric to ClearML."""
        # Extract series and title from metric name
        if "/" in metric.name:
            series, title = metric.name.split("/", 1)
        else:
            series = metric.source or "metrics"
            title = metric.name

        self._logger.report_scalar(
            title=title, series=series, value=float(metric.value), iteration=metric.iteration or 0
        )

    def _log_parameter(self, metric: MetricData):
        """Log a parameter to ClearML."""
        if self._task:
            # Parameters can be logged as hyperparameters
            params = {metric.name: metric.value}
            self._task.connect(params)

    def _log_artifact(self, metric: MetricData):
        """Log an artifact to ClearML."""
        if self._task:
            artifact_path = metric.metadata.get("artifact_path") if metric.metadata else None

            # Upload artifact
            self._task.upload_artifact(
                name=artifact_path or metric.name,
                artifact_object=metric.value,  # This should be the file path
            )

    def _log_histogram_metric(self, metric: MetricData):
        """Log a histogram metric to ClearML."""
        # Extract series and title from metric name
        if "/" in metric.name:
            series, title = metric.name.split("/", 1)
        else:
            series = metric.source or "histograms"
            title = metric.name

        # ClearML can handle numpy arrays and torch tensors for histograms
        self._logger.report_histogram(
            title=title,
            series=series,
            values=metric.value,
            iteration=metric.iteration or 0,
            xlabels=None,
            yaxis="Frequency",
            xaxis="Value",
        )

    def _log_image_metric(self, metric: MetricData):
        """Log an image metric to ClearML."""
        # Extract series and title from metric name
        if "/" in metric.name:
            series, title = metric.name.split("/", 1)
        else:
            series = metric.source or "images"
            title = metric.name

        # ClearML can handle various image formats including tensors
        self._logger.report_image(title=title, series=series, iteration=metric.iteration or 0, image=metric.value)

    def _log_text_metric(self, metric: MetricData):
        """Log a text metric to ClearML."""
        self._logger.report_text(msg=str(metric.value), level=logging.INFO, print_console=False)

    def _log_figure_metric(self, metric: MetricData):
        """Log a matplotlib figure to ClearML."""
        # Extract series and title from metric name
        if "/" in metric.name:
            series, title = metric.name.split("/", 1)
        else:
            series = metric.source or "figures"
            title = metric.name

        # ClearML can directly handle matplotlib figures
        self._logger.report_matplotlib_figure(
            title=title, series=series, iteration=metric.iteration or 0, figure=metric.value, report_interactive=False
        )

    def _log_hparams_metric(self, metric: MetricData):
        """Log hyperparameters to ClearML."""
        if not self._task:
            return

        # ClearML handles hyperparameters via task.connect()
        if isinstance(metric.value, dict) and "hparams" in metric.value:
            hparams = metric.value["hparams"]
            self._task.connect(hparams, name="hyperparameters")

    def _log_custom_metric(self, metric: MetricData):
        """Log a custom metric to ClearML."""
        # Log as scalar if numeric, otherwise as text
        if isinstance(metric.value, (int, float)):
            self._log_scalar_metric(metric)
        else:
            # Log as text
            text_msg = f"{metric.name}: {metric.value}"
            self._logger.report_text(text_msg, level=logging.INFO, print_console=False)

    # BackendInterface implementation
    def log_metric(self, name: str, value: Any, iteration: int):
        """Log a metric to ClearML."""
        metric = MetricData(
            name=name,
            value=value,
            type=MetricType.SCALAR if isinstance(value, (int, float)) else MetricType.CUSTOM,
            iteration=iteration,
        )
        self.receive_metric(metric)

    def log_params(self, params: dict[str, Any]):
        """Log parameters to ClearML."""
        if self._task:
            self._task.connect(params)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Upload an artifact to ClearML."""
        if self._task:
            self._task.upload_artifact(name=artifact_path or local_path, artifact_object=local_path)

    def save_experiment(self, experiment_data: dict[str, Any]):  # noqa: C901
        """Save experiment metadata to ClearML."""
        if not self._active or not self._task:
            return

        try:
            # Add experiment metadata as tags
            tags_to_add = []

            # Process git information
            if "git" in experiment_data:
                git_info = experiment_data["git"]
                if isinstance(git_info, dict):
                    # Add git hash as tag if available
                    if "commit_hash" in git_info:
                        tags_to_add.append(f"git:{git_info['commit_hash'][:8]}")

                    # Connect git info as configuration
                    self._task.connect(git_info, name="git_info")

            # Process system information
            if "system" in experiment_data:
                system_info = experiment_data["system"]
                if isinstance(system_info, dict):
                    # Add Python version as tag if available
                    if "python_version" in system_info:
                        tags_to_add.append(f"python:{system_info['python_version']}")

                    # Connect system info as configuration
                    self._task.connect(system_info, name="system_info")

            # Process environment information
            if "environment" in experiment_data:
                env_info = experiment_data["environment"]
                if isinstance(env_info, dict):
                    self._task.connect(env_info, name="environment")

            # Add collected tags to task
            if tags_to_add:
                current_tags = list(self._task.get_tags() or [])
                current_tags.extend(tags_to_add)
                self._task.set_tags(current_tags)

            # Set task description with experiment details
            if "description" in experiment_data:
                self._task.set_description(experiment_data["description"])

        except Exception:
            logger.exception("Failed to save experiment metadata to ClearML")
