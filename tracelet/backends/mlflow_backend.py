"""MLflow backend plugin for Tracelet."""

import logging
from typing import TYPE_CHECKING, Any, Optional

from tracelet.core.orchestrator import MetricData, MetricType
from tracelet.core.plugins import BackendPlugin, PluginMetadata, PluginType
from tracelet.utils.imports import require

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class MLflowBackend(BackendPlugin):
    """MLflow backend plugin for experiment tracking."""

    def __init__(self):
        super().__init__()
        self._client: Optional[MlflowClient] = None
        self._experiment_id: Optional[str] = None
        self._run_id: Optional[str] = None
        self._experiment_name = "Tracelet Experiments"
        self._run_name: Optional[str] = None

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="mlflow",
            version="1.0.0",
            type=PluginType.BACKEND,
            description="MLflow backend for experiment tracking",
            author="Tracelet Team",
            dependencies=[],
            capabilities={"metrics", "parameters", "artifacts", "logging", "model_registry"},
        )

    def initialize(self, config: dict[str, Any]):
        """Initialize MLflow backend."""
        # Use dynamic import system
        mlflow = require("mlflow", "MLflow backend")

        self._config = config
        self._experiment_name = config.get("experiment_name", "Tracelet Experiments")
        self._run_name = config.get("run_name")

        # Set tracking URI if provided
        tracking_uri = config.get("tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            self._tracking_uri = tracking_uri
        else:
            # Check if we're using Databricks
            import os

            if os.environ.get("MLFLOW_TRACKING_URI") == "databricks":
                self._tracking_uri = "databricks"
            else:
                self._tracking_uri = mlflow.get_tracking_uri()

        # Handle Databricks experiment naming
        if self._tracking_uri == "databricks" and not self._experiment_name.startswith("/"):
            # Try to get user email from environment or use a default
            user_email = os.environ.get("DATABRICKS_USER_EMAIL", "atemysemicolon@gmail.com")
            self._experiment_name = f"/Users/{user_email}/{self._experiment_name}"

        # Initialize MLflow client
        from mlflow.tracking import MlflowClient

        self._client = MlflowClient()

        # Get or create experiment
        try:
            experiment = self._client.get_experiment_by_name(self._experiment_name)
            if experiment is None:
                self._experiment_id = self._client.create_experiment(self._experiment_name)
            else:
                self._experiment_id = experiment.experiment_id
        except Exception:
            logger.exception("Failed to get/create MLflow experiment")
            raise

        logger.info(f"Initialized MLflow backend for experiment '{self._experiment_name}' (ID: {self._experiment_id})")

    def start(self):
        """Start the MLflow backend."""
        if not self._client or not self._experiment_id:
            raise RuntimeError("MLflow backend not properly initialized")

        # Start a new MLflow run
        try:
            run = self._client.create_run(experiment_id=self._experiment_id, run_name=self._run_name)
            self._run_id = run.info.run_id

            # Set active run
            mlflow = require("mlflow")
            mlflow.start_run(run_id=self._run_id, nested=True)

            logger.info(f"Started MLflow run: {self._run_id}")
        except Exception:
            logger.exception("Failed to start MLflow run")
            raise

        self._active = True

    def stop(self):
        """Stop the MLflow backend."""
        if self._run_id and self._client:
            try:
                # End the run
                mlflow = require("mlflow")
                mlflow.end_run()
                logger.info(f"Stopped MLflow run: {self._run_id}")
            except Exception:
                logger.exception("Error stopping MLflow run")

        self._active = False

    def get_status(self) -> dict[str, Any]:
        """Get backend status."""
        status = {
            "active": self._active,
            "experiment_name": self._experiment_name,
            "experiment_id": self._experiment_id,
            "run_id": self._run_id,
        }

        if self._client and self._run_id:
            try:
                run = self._client.get_run(self._run_id)
                status.update({
                    "run_name": run.info.run_name,
                    "run_status": run.info.status,
                    "start_time": run.info.start_time,
                    "artifact_uri": run.info.artifact_uri,
                })
            except Exception:
                logger.exception("Error getting run status")

        return status

    def receive_metric(self, metric: MetricData):
        """Receive and process a metric from the orchestrator."""
        if not self._active or not self._client or not self._run_id:
            return

        try:
            if metric.type == MetricType.SCALAR:
                self._log_scalar_metric(metric)
            elif metric.type == MetricType.PARAMETER:
                self._log_parameter(metric)
            elif metric.type == MetricType.ARTIFACT:
                self._log_artifact(metric)
            else:
                # Log as tag or custom metric
                self._log_custom_metric(metric)
        except Exception:
            logger.exception(f"Failed to log metric '{metric.name}' to MLflow")

    def _get_clean_metric_name(self, metric: MetricData, separator: str = ".") -> str:
        """Get clean metric name without experiment ID prefixes."""
        if not metric.source or metric.source == "experiment" or metric.source.startswith("experiment_"):
            return metric.name
        return f"{metric.source}{separator}{metric.name}"

    def _log_scalar_metric(self, metric: MetricData):
        """Log a scalar metric to MLflow."""
        metric_name = self._get_clean_metric_name(metric)

        self._client.log_metric(
            run_id=self._run_id,
            key=metric_name,
            value=float(metric.value),
            timestamp=int(metric.timestamp * 1000) if metric.timestamp else None,
            step=metric.iteration,
        )

    def _log_parameter(self, metric: MetricData):
        """Log a parameter to MLflow."""
        param_name = self._get_clean_metric_name(metric)

        # Convert value to string for MLflow
        param_value = str(metric.value)
        if len(param_value) > 500:  # MLflow has a limit on param value length
            param_value = param_value[:497] + "..."

        self._client.log_param(run_id=self._run_id, key=param_name, value=param_value)

    def _log_artifact(self, metric: MetricData):
        """Log an artifact to MLflow."""
        # For artifacts, metric.value should be the file path
        file_path = metric.value
        artifact_path = metric.metadata.get("artifact_path") if metric.metadata else None

        try:
            self._client.log_artifact(run_id=self._run_id, local_path=file_path, artifact_path=artifact_path)
        except Exception:
            logger.exception(f"Failed to log artifact '{metric.name}'")

    def _log_custom_metric(self, metric: MetricData):
        """Log a custom metric as a tag in MLflow."""
        tag_name = f"custom.{metric.name}"
        if metric.source and metric.source != "experiment":
            tag_name = f"custom.{metric.source}.{metric.name}"

        tag_value = str(metric.value)
        if len(tag_value) > 5000:  # MLflow tag value limit
            tag_value = tag_value[:4997] + "..."

        self._client.set_tag(run_id=self._run_id, key=tag_name, value=tag_value)

    # BackendInterface implementation
    def log_metric(self, name: str, value: Any, iteration: int):
        """Log a metric to MLflow."""
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
        """Log parameters to MLflow."""
        if not self._active or not self._client or not self._run_id:
            return

        for name, value in params.items():
            metric = MetricData(name=name, value=value, type=MetricType.PARAMETER)
            self.receive_metric(metric)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Upload an artifact to MLflow."""
        if not self._active or not self._client or not self._run_id:
            return

        try:
            self._client.log_artifact(run_id=self._run_id, local_path=local_path, artifact_path=artifact_path)
        except Exception:
            logger.exception(f"Failed to log artifact '{local_path}'")

    def save_experiment(self, experiment_data: dict[str, Any]):
        """Save experiment metadata to MLflow."""
        if not self._active or not self._client or not self._run_id:
            return

        # Log experiment metadata as tags and params
        for key, value in experiment_data.items():
            try:
                # Try to log as parameter first (for structured data)
                if isinstance(value, (str, int, float, bool)):
                    self._client.log_param(run_id=self._run_id, key=f"exp.{key}", value=str(value))
                else:
                    # Log complex data as tags
                    self._client.set_tag(run_id=self._run_id, key=f"exp.{key}", value=str(value))
            except Exception:
                logger.exception(f"Failed to log experiment metadata '{key}'")
