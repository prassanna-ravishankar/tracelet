"""AIM backend for Tracelet."""

import logging
from pathlib import Path
from typing import Any, Optional

try:
    import aim

    _has_aim = True
except ImportError:
    _has_aim = False
    aim = None

from tracelet.core.orchestrator import MetricData, MetricType
from tracelet.core.plugins import BackendPlugin, PluginMetadata, PluginType

logger = logging.getLogger(__name__)


class AimBackend(BackendPlugin):
    """AIM backend plugin for experiment tracking.

    Supports both local and remote AIM deployments for flexible experiment tracking.
    """

    def __init__(self):
        super().__init__()
        self._repo: Optional[Any] = None
        self._run: Optional[Any] = None
        self._experiment_name = "Tracelet Experiments"
        self._run_hash: Optional[str] = None
        self._remote_tracking_uri: Optional[str] = None

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="aim",
            version="1.0.0",
            type=PluginType.BACKEND,
            description="AIM backend for experiment tracking with local and remote support",
            author="Tracelet Team",
            dependencies=[],
            capabilities={"metrics", "parameters", "artifacts", "distributions", "images", "audio", "text", "remote"},
        )

    def initialize(self, config: dict[str, Any]):
        """Initialize AIM backend.

        Args:
            config: Configuration with optional keys:
                - repo_path: Path to local AIM repository (default: current directory)
                - remote_uri: URI for remote AIM server (e.g., "http://aim-server:53800")
                - experiment_name: Name for the experiment (default: "Tracelet Experiments")
                - run_name: Name for the specific run
                - tags: Dictionary of tags to apply to the run
        """
        if not _has_aim:
            msg = "AIM is not installed. Install with: pip install aim"
            raise ImportError(msg)

        self._config = config
        self._experiment_name = config.get("experiment_name", "Tracelet Experiments")
        self._run_name = config.get("run_name")
        self._tags = config.get("tags", {})

        # Configure repository (local or remote)
        self._remote_tracking_uri = config.get("remote_uri")
        repo_path = config.get("repo_path", ".")

        try:
            if self._remote_tracking_uri:
                # Connect to remote AIM server
                self._repo = aim.Repo.from_remote(self._remote_tracking_uri)
                logger.info(f"Connected to remote AIM server at {self._remote_tracking_uri}")
            else:
                # Use local repository
                self._repo = aim.Repo.from_path(repo_path, init=True)
                logger.info(f"Initialized local AIM repository at {repo_path}")
        except Exception:
            logger.exception("Failed to initialize AIM repository")
            raise

    def start(self):
        """Start the AIM backend."""
        if not self._repo:
            msg = "AIM backend not properly initialized"
            raise RuntimeError(msg)

        try:
            # Create a new AIM run
            self._run = aim.Run(
                repo=self._repo,
                experiment=self._experiment_name,
                run_name=self._run_name,
            )

            # Set tags
            for key, value in self._tags.items():
                self._run.set(key, value, strict=False)

            # Store run hash for reference
            self._run_hash = self._run.hash

            logger.info(f"Started AIM run: {self._run_hash} (experiment: {self._experiment_name})")
        except Exception:
            logger.exception("Failed to start AIM run")
            raise

        self._active = True

    def stop(self):
        """Stop the AIM backend."""
        if self._run:
            try:
                # Finalize the run
                self._run.close()
                logger.info(f"Stopped AIM run: {self._run_hash}")
            except Exception:
                logger.exception("Error stopping AIM run")

        self._active = False

    def get_status(self) -> dict[str, Any]:
        """Get backend status."""
        status = {
            "active": self._active,
            "experiment_name": self._experiment_name,
            "run_hash": self._run_hash,
            "remote_uri": self._remote_tracking_uri,
            "has_run": self._run is not None,
        }

        if self._run:
            try:
                status.update({
                    "run_name": self._run.name,
                    "metrics_count": len(self._run.get_metric_names()),
                    "params_count": len(list(self._run.get_params())),
                })
            except Exception:
                logger.exception("Error getting run status")

        return status

    def receive_metric(self, metric: MetricData):
        """Receive and process a metric from the orchestrator."""
        if not self._active or not self._run:
            return

        try:
            if metric.type == MetricType.SCALAR:
                self._log_scalar_metric(metric)
            elif metric.type == MetricType.PARAMETER:
                self._log_parameter(metric)
            elif metric.type == MetricType.ARTIFACT:
                self._log_artifact_metric(metric)
            elif metric.type == MetricType.HISTOGRAM:
                self._log_distribution_metric(metric)
            elif metric.type == MetricType.IMAGE:
                self._log_image_metric(metric)
            elif metric.type == MetricType.TEXT:
                self._log_text_metric(metric)
            elif metric.type == MetricType.AUDIO:
                self._log_audio_metric(metric)
            else:
                # Log as custom tracked value
                self._log_custom_metric(metric)
        except Exception:
            logger.exception(f"Failed to log metric '{metric.name}' to AIM")

    def _log_scalar_metric(self, metric: MetricData):
        """Log a scalar metric to AIM."""
        metric_name = metric.name
        if metric.source and metric.source != "experiment":
            metric_name = f"{metric.source}/{metric.name}"

        context = metric.metadata.get("context", {}) if metric.metadata else {}

        self._run.track(
            value=float(metric.value),
            name=metric_name,
            step=metric.iteration,
            context=context,
        )

    def _log_parameter(self, metric: MetricData):
        """Log a parameter to AIM."""
        param_name = metric.name
        if metric.source and metric.source != "experiment":
            param_name = f"{metric.source}.{metric.name}"

        self._run.set(param_name, metric.value, strict=False)

    def _log_artifact_metric(self, metric: MetricData):
        """Log an artifact to AIM."""
        # For artifacts, metric.value should be the file path
        file_path = Path(metric.value)

        if not file_path.exists():
            logger.warning(f"Artifact file not found: {file_path}")
            return

        artifact_name = metric.metadata.get("artifact_path", file_path.name) if metric.metadata else file_path.name

        try:
            # AIM doesn't have direct artifact logging like MLflow
            # Store file reference as a tracked parameter
            self._run.set(f"artifacts/{artifact_name}", str(file_path), strict=False)

            # If it's a small text file, we can store its content
            if file_path.suffix in [".txt", ".json", ".yaml", ".yml"] and file_path.stat().st_size < 1024 * 1024:  # 1MB
                try:
                    content = file_path.read_text()
                    self._run.track(aim.Text(content), name=f"artifact_content/{artifact_name}")
                except Exception:
                    # Failed to read text file content - skip embedding
                    logger.debug(f"Could not read text content from {file_path}")

        except Exception:
            logger.exception(f"Failed to log artifact '{metric.name}'")

    def _log_distribution_metric(self, metric: MetricData):
        """Log a distribution/histogram metric to AIM."""
        metric_name = f"distributions/{metric.name}"
        if metric.source and metric.source != "experiment":
            metric_name = f"{metric.source}/distributions/{metric.name}"

        try:
            # Convert to AIM Distribution
            self._run.track(
                aim.Distribution(metric.value),
                name=metric_name,
                step=metric.iteration,
            )
        except Exception:
            logger.exception(f"Failed to log distribution '{metric.name}'")

    def _log_image_metric(self, metric: MetricData):
        """Log an image metric to AIM."""
        metric_name = f"images/{metric.name}"
        if metric.source and metric.source != "experiment":
            metric_name = f"{metric.source}/images/{metric.name}"

        try:
            # Convert to AIM Image
            self._run.track(
                aim.Image(metric.value),
                name=metric_name,
                step=metric.iteration,
            )
        except Exception:
            logger.exception(f"Failed to log image '{metric.name}'")

    def _log_text_metric(self, metric: MetricData):
        """Log a text metric to AIM."""
        metric_name = f"text/{metric.name}"
        if metric.source and metric.source != "experiment":
            metric_name = f"{metric.source}/text/{metric.name}"

        try:
            # Convert to AIM Text
            self._run.track(
                aim.Text(str(metric.value)),
                name=metric_name,
                step=metric.iteration,
            )
        except Exception:
            logger.exception(f"Failed to log text '{metric.name}'")

    def _log_audio_metric(self, metric: MetricData):
        """Log an audio metric to AIM."""
        metric_name = f"audio/{metric.name}"
        if metric.source and metric.source != "experiment":
            metric_name = f"{metric.source}/audio/{metric.name}"

        sample_rate = metric.metadata.get("sample_rate", 44100) if metric.metadata else 44100

        try:
            # Convert to AIM Audio
            self._run.track(
                aim.Audio(metric.value, rate=sample_rate),
                name=metric_name,
                step=metric.iteration,
            )
        except Exception:
            logger.exception(f"Failed to log audio '{metric.name}'")

    def _log_custom_metric(self, metric: MetricData):
        """Log a custom metric as a tracked value in AIM."""
        metric_name = f"custom/{metric.name}"
        if metric.source and metric.source != "experiment":
            metric_name = f"custom/{metric.source}/{metric.name}"

        try:
            self._run.track(
                value=metric.value,
                name=metric_name,
                step=metric.iteration,
            )
        except Exception:
            logger.exception(f"Failed to log custom metric '{metric.name}'")

    # BackendInterface implementation
    def log_metric(self, name: str, value: Any, iteration: int):
        """Log a metric (implements BackendInterface)."""
        if self._run:
            self._run.track(value, name=name, step=iteration)

    def log_params(self, params: dict[str, Any]):
        """Log parameters (implements BackendInterface)."""
        if self._run:
            for key, value in params.items():
                self._run.set(key, value, strict=False)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact (implements BackendInterface)."""
        if self._run:
            path = Path(local_path)
            artifact_name = artifact_path or path.name
            self._run.set(f"artifacts/{artifact_name}", str(local_path), strict=False)

    def save_experiment(self, experiment_data: dict[str, Any]):
        """Save experiment metadata (implements BackendInterface)."""
        if self._run:
            for key, value in experiment_data.items():
                self._run.set(f"experiment/{key}", value, strict=False)
