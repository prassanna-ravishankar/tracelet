from typing import Any, Optional

try:
    import wandb

    _HAS_WANDB = True
except ImportError:
    wandb = None
    _HAS_WANDB = False

from ..core.interfaces import BackendInterface

# Error messages
WANDB_NOT_INSTALLED_MSG = "Weights & Biases is not installed. Install with: pip install wandb"
WANDB_NOT_INITIALIZED_MSG = "W&B backend not initialized. Call initialize() first."


class WandbBackend(BackendInterface):
    """Weights & Biases backend integration for experiment tracking"""

    def __init__(
        self,
        project_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        entity: Optional[str] = None,
        tags: Optional[list[str]] = None,
        config: Optional[dict[str, Any]] = None,
        job_type: str = "train",
    ):
        if not _HAS_WANDB:
            raise ImportError(WANDB_NOT_INSTALLED_MSG)

        self.project_name = project_name or "tracelet-experiments"
        self.experiment_name = experiment_name or "experiment"
        self.entity = entity  # W&B team/user entity
        self.tags = tags or []
        self.config = config or {}
        self.job_type = job_type
        self._run: Optional[wandb.sdk.wandb_run.Run] = None

    def initialize(self):
        """Initialize W&B run"""
        # Initialize W&B run
        self._run = wandb.init(
            project=self.project_name,
            name=self.experiment_name,
            entity=self.entity,
            tags=self.tags,
            config=self.config,
            job_type=self.job_type,
            reinit=True,  # Allow re-initialization
        )

    def log_metric(self, name: str, value: Any, iteration: int):
        """Log a scalar metric"""
        if not self._run:
            raise RuntimeError(WANDB_NOT_INITIALIZED_MSG)

        # Handle different value types
        if isinstance(value, (int, float)):
            # W&B handles step automatically, but we can specify it
            self._run.log({name: float(value)}, step=iteration)
        else:
            # For non-numeric values, log as string
            self._run.log({f"{name}_text": str(value)}, step=iteration)

    def log_params(self, params: dict[str, Any]):
        """Log hyperparameters"""
        if not self._run:
            raise RuntimeError(WANDB_NOT_INITIALIZED_MSG)

        # W&B uses config for hyperparameters
        # We can update the config during the run
        for key, value in params.items():
            self._run.config[key] = value

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Upload an artifact to W&B"""
        if not self._run:
            raise RuntimeError(WANDB_NOT_INITIALIZED_MSG)

        # W&B artifacts system
        artifact_name = artifact_path or local_path.split("/")[-1]

        # Create artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type="file",  # Can be customized based on file type
        )

        # Add file to artifact
        artifact.add_file(local_path)

        # Log artifact
        self._run.log_artifact(artifact)

    def save_experiment(self, experiment_data: dict[str, Any]):  # noqa: C901
        """Save experiment metadata to W&B"""
        if not self._run:
            raise RuntimeError(WANDB_NOT_INITIALIZED_MSG)

        # Add experiment metadata as tags and config
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

        # Add collected tags to run
        if tags_to_add:
            current_tags = list(self._run.tags or [])
            current_tags.extend(tags_to_add)
            # W&B doesn't allow modifying tags after run creation,
            # but we can add them to config
            self._run.config["additional_tags"] = tags_to_add

        # Add description/notes
        if "description" in experiment_data:
            self._run.notes = experiment_data["description"]

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._run:
            # Finish the run
            self._run.finish()
            self._run = None
