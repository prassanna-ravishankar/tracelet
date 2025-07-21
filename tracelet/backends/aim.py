"""AIM backend for Tracelet."""

import aim

from tracelet.core.plugins import BackendPlugin, PluginMetadata, PluginType


class AimBackend(BackendPlugin):
    """AIM backend for Tracelet."""

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="aim",
            version="0.1.0",
            type=PluginType.BACKEND,
            description="AIM backend for Tracelet",
        )

    def initialize(self, config: dict):
        """Initialize the backend."""
        self.repo = aim.Repo.from_path(config.get("repo", "."))
        self.run = aim.Run(repo=self.repo, experiment=config.get("experiment"))

    def log_metric(self, name: str, value: float, iteration: int):
        """Log a metric."""
        self.run.track(value, name=name, step=iteration)

    def log_param(self, name: str, value: str):
        """Log a parameter."""
        self.run.set(name, value, strict=False)

    def log_artifact(self, local_path: str, artifact_path: str | None = None):
        """Log an artifact."""
        self.run.track_file(local_path, name=artifact_path or local_path)
