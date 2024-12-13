from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class TraceletSettings(BaseSettings):
    """Global settings for Tracelet, configurable via environment variables"""
    
    # Core settings
    project_name: str = Field(default="default", env="TRACELET_PROJECT")
    experiment_name: Optional[str] = Field(default=None, env="TRACELET_EXPERIMENT_NAME")
    backend: Literal["mlflow", "wandb", "aim"] = Field(default="mlflow", env="TRACELET_BACKEND")
    
    # Backend credentials
    api_key: Optional[str] = Field(default=None, env="TRACELET_API_KEY")
    backend_url: Optional[str] = Field(default=None, env="TRACELET_BACKEND_URL")
    
    # Feature flags
    track_tensorboard: bool = Field(default=True, env="TRACELET_TRACK_TENSORBOARD")
    track_lightning: bool = Field(default=True, env="TRACELET_TRACK_LIGHTNING")
    track_system_metrics: bool = Field(default=True, env="TRACELET_TRACK_SYSTEM")
    track_git: bool = Field(default=True, env="TRACELET_TRACK_GIT")
    track_env: bool = Field(default=True, env="TRACELET_TRACK_ENV")
    
    # System metrics settings
    system_metrics_interval: float = Field(default=10.0, env="TRACELET_METRICS_INTERVAL")
    
    model_config = SettingsConfigDict(
        env_prefix="TRACELET_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    ) 