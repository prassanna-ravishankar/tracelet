from typing import Any, Optional

try:
    from clearml import Logger, Task

    _HAS_CLEARML = True
except ImportError:
    Task = Logger = None
    _HAS_CLEARML = False

from ..core.interfaces import BackendInterface

# Error messages
CLEARML_NOT_INSTALLED_MSG = "ClearML is not installed. Install with: pip install clearml"
CLEARML_NOT_INITIALIZED_MSG = "ClearML backend not initialized. Call initialize() first."


class ClearMLBackend(BackendInterface):
    """ClearML backend integration for experiment tracking"""

    def __init__(
        self,
        project_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_type: str = "training",
        tags: Optional[list[str]] = None,
    ):
        if not _HAS_CLEARML:
            raise ImportError(CLEARML_NOT_INSTALLED_MSG)

        self.project_name = project_name or "TraceletExperiments"
        self.task_name = task_name or "experiment"
        self.task_type = task_type
        self.tags = tags or []
        self._task: Optional[Task] = None
        self._logger: Optional[Logger] = None

    def initialize(self):
        """Initialize ClearML task and logger"""
        # Create ClearML task
        self._task = Task.init(
            project_name=self.project_name,
            task_name=self.task_name,
            task_type=self.task_type,
            tags=self.tags,
        )

        # Get logger for metrics
        self._logger = self._task.get_logger()

    def log_metric(self, name: str, value: Any, iteration: int):
        """Log a scalar metric"""
        if not self._logger:
            raise RuntimeError(CLEARML_NOT_INITIALIZED_MSG)

        # Handle different value types
        if isinstance(value, (int, float)):
            # Extract series name and metric name from full name
            if "/" in name:
                series, title = name.split("/", 1)
            else:
                series = "metrics"
                title = name

            self._logger.report_scalar(title=title, series=series, value=float(value), iteration=iteration or 0)
        else:
            # For non-numeric values, log as text
            self._logger.report_text(f"{name}: {value}", iteration=iteration or 0)

    def log_params(self, params: dict[str, Any]):
        """Log hyperparameters"""
        if not self._task:
            raise RuntimeError(CLEARML_NOT_INITIALIZED_MSG)

        # Convert all values to strings for ClearML
        string_params = {}
        for key, value in params.items():
            if isinstance(value, (dict, list)):
                # For complex objects, convert to string representation
                string_params[key] = str(value)
            else:
                string_params[key] = value

        self._task.connect(string_params, name="hyperparameters")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Upload an artifact to ClearML"""
        if not self._task:
            raise RuntimeError(CLEARML_NOT_INITIALIZED_MSG)

        # ClearML automatically handles artifact uploads
        self._task.upload_artifact(name=artifact_path or local_path, artifact_object=local_path)

    def save_experiment(self, experiment_data: dict[str, Any]):  # noqa: C901
        """Save experiment metadata as tags and configuration"""
        if not self._task:
            raise RuntimeError(CLEARML_NOT_INITIALIZED_MSG)

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
        description_parts = []
        if "description" in experiment_data:
            description_parts.append(experiment_data["description"])

        if description_parts:
            self._task.set_description("\n".join(description_parts))

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._task:
            # Mark task as completed
            self._task.close()
            self._task = None
            self._logger = None
