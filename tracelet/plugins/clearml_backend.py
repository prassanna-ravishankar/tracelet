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


class ClearMLBackend(BackendPlugin):
    """ClearML backend plugin for experiment tracking."""

    def __init__(self):
        super().__init__()
        self._task: Optional[Task] = None
        self._logger: Optional[Logger] = None
        self._project_name = "Tracelet Experiments"
        self._task_name = "Default Task"

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="clearml",
            version="1.0.0",
            type=PluginType.BACKEND,
            description="ClearML backend for experiment tracking",
            author="Tracelet Team",
            dependencies=[],
            capabilities={"metrics", "parameters", "artifacts", "logging"}
        )

    def initialize(self, config: dict[str, Any]):
        """Initialize ClearML backend."""
        if not _has_clearml:
            raise ImportError("ClearML is not installed. Install with: pip install clearml")

        self._config = config
        self._project_name = config.get("project_name", "Tracelet Experiments")
        self._task_name = config.get("task_name", "Default Task")

        # Initialize ClearML Task
        self._task = Task.init(
            project_name=self._project_name,
            task_name=self._task_name,
            auto_connect_frameworks=config.get("auto_connect_frameworks", True),
            auto_connect_arg_parser=config.get("auto_connect_arg_parser", True),
        )

        # Get logger
        self._logger = self._task.get_logger()

        logger.info(f"Initialized ClearML backend for project '{self._project_name}'")

    def start(self):
        """Start the ClearML backend."""
        if self._task:
            logger.info("ClearML backend started")
        self._active = True

    def stop(self):
        """Stop the ClearML backend."""
        if self._task:
            # Mark task as completed
            self._task.close()
            logger.info("ClearML backend stopped and task completed")
        self._active = False

    def get_status(self) -> dict[str, Any]:
        """Get backend status."""
        status = {
            "active": self._active,
            "project_name": self._project_name,
            "task_name": self._task_name,
        }

        if self._task:
            status.update({
                "task_id": self._task.id,
                "task_url": self._task.get_output_log_web_page(),
                "task_status": str(self._task.get_status()) if self._task.get_status() else "unknown"
            })

        return status

    def receive_metric(self, metric: MetricData):
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
            else:
                # Log as custom metric
                self._log_custom_metric(metric)
        except Exception as e:
            logger.exception(f"Failed to log metric '{metric.name}' to ClearML: {e}")

    def _log_scalar_metric(self, metric: MetricData):
        """Log a scalar metric to ClearML."""
        series = metric.name
        if metric.source:
            series = f"{metric.source}/{metric.name}"

        self._logger.report_scalar(
            title="Metrics",
            series=series,
            value=metric.value,
            iteration=metric.iteration or 0
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
                artifact_object=metric.value  # This should be the file path
            )

    def _log_custom_metric(self, metric: MetricData):
        """Log a custom metric to ClearML."""
        # Log as text or debug info
        if self._logger:
            self._logger.report_text(
                f"Custom Metric: {metric.name} = {metric.value} (type: {metric.type.value})"
            )

    # BackendInterface implementation
    def log_metric(self, name: str, value: Any, iteration: int):
        """Log a metric to ClearML."""
        metric = MetricData(
            name=name,
            value=value,
            type=MetricType.SCALAR if isinstance(value, (int, float)) else MetricType.CUSTOM,
            iteration=iteration
        )
        self.receive_metric(metric)

    def log_params(self, params: dict[str, Any]):
        """Log parameters to ClearML."""
        if self._task:
            self._task.connect(params)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Upload an artifact to ClearML."""
        if self._task:
            self._task.upload_artifact(
                name=artifact_path or local_path,
                artifact_object=local_path
            )

    def save_experiment(self, experiment_data: dict[str, Any]):
        """Save experiment metadata to ClearML."""
        if self._task:
            # Log experiment metadata as configuration
            self._task.connect(experiment_data, name="experiment_metadata")
