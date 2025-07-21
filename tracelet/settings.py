from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TraceletSettings(BaseSettings):
    """Global settings for Tracelet, configurable via environment variables"""

    # Core settings
    project: str = Field(default="default")
    experiment_name: Optional[str] = Field(default=None)
    backend: Literal["mlflow", "wandb", "aim"] = Field(default="mlflow")

    # Backend credentials
    api_key: Optional[str] = Field(default=None)
    backend_url: Optional[str] = Field(default=None)

    # Feature flags
    track_tensorboard: bool = Field(default=True)
    track_lightning: bool = Field(default=True)
    track_system: bool = Field(default=True)
    track_git: bool = Field(default=True)
    track_env: bool = Field(default=True)

    # System metrics settings
    metrics_interval: float = Field(default=10.0)

    model_config = SettingsConfigDict(
        env_prefix="TRACELET_", case_sensitive=False, env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def project_name(self) -> str:
        """Alias for project field for backwards compatibility"""
        return self.project

    @property
    def track_system_metrics(self) -> bool:
        """Alias for track_system field for backwards compatibility"""
        return self.track_system
