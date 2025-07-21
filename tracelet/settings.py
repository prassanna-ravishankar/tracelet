from typing import Any, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import EnvSettingsSource


class CustomEnvSettings(EnvSettingsSource):
    """Custom env settings source that doesn't parse backend as JSON"""

    def prepare_field_value(self, field_name: str, field: Any, value: Any, value_is_complex: bool) -> Any:
        # Don't parse backend field as JSON
        if field_name == "backend":
            return value
        return super().prepare_field_value(field_name, field, value, value_is_complex)


class TraceletSettings(BaseSettings):
    """Global settings for Tracelet, configurable via environment variables"""

    # Core settings
    project: str = Field(default="default")
    experiment_name: Optional[str] = Field(default=None)
    backend: list[Literal["mlflow", "wandb", "aim", "clearml"]] = Field(default=["mlflow"])

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

    @field_validator("backend", mode="before")
    @classmethod
    def validate_backend(cls, v: Any) -> list:
        """Convert string backend to list if needed"""
        if isinstance(v, str):
            # Handle both comma-separated and single values
            backends = [b.strip() for b in v.split(",") if b.strip()]
            return backends if backends else ["mlflow"]
        elif isinstance(v, list):
            return v
        else:
            return ["mlflow"]

    @property
    def project_name(self) -> str:
        """Alias for project field for backwards compatibility"""
        return self.project

    @property
    def track_system_metrics(self) -> bool:
        """Alias for track_system field for backwards compatibility"""
        return self.track_system

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Customize settings sources to handle backend field properly"""
        return (
            init_settings,
            CustomEnvSettings(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )
