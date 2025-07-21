# Settings and Configuration

::: tracelet.settings.TraceletSettings
options:
show_source: true
show_bases: true
heading_level: 2

## Configuration Fields

The `TraceletSettings` class uses Pydantic for configuration management with validation and environment variable support.

### Core Settings

```python
from tracelet.settings import TraceletSettings

# Basic configuration
settings = TraceletSettings(
    project="my_ml_project",
    experiment_name="baseline_model",
    backend=["mlflow"]  # Single backend as list
)

# Multi-backend configuration
settings = TraceletSettings(
    project="comparison_study",
    backend=["mlflow", "wandb", "clearml"]  # Multiple backends
)
```

### Backend Configuration

```python
# MLflow-specific settings
mlflow_settings = TraceletSettings(
    project="mlflow_project",
    backend=["mlflow"],
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="Deep Learning Experiments"
)

# Weights & Biases settings
wandb_settings = TraceletSettings(
    project="wandb_project",
    backend=["wandb"],
    wandb_project="ml-experiments",
    wandb_entity="my_team",
    wandb_api_key="your_api_key_here"  # Better to use env var
)

# ClearML settings
clearml_settings = TraceletSettings(
    project="clearml_project",
    backend=["clearml"],
    clearml_project_name="Research Experiments",
    clearml_task_name="Model Training"
)
```

### System Metrics Configuration

```python
# Enable system monitoring
settings = TraceletSettings(
    project="monitored_training",
    backend=["mlflow"],
    track_system=True,              # Enable system metrics
    metrics_interval=5.0,           # Collect every 5 seconds
    track_gpu=True,                 # Include GPU metrics
    track_disk=True,                # Include disk I/O
    track_network=True              # Include network I/O
)
```

## Environment Variables

All settings can be configured via environment variables using the `TRACELET_` prefix:

### Basic Environment Setup

```bash
# Core settings
export TRACELET_PROJECT="production_models"
export TRACELET_EXPERIMENT_NAME="model_v2_training"
export TRACELET_BACKEND="mlflow,wandb"  # Comma-separated for multiple

# System monitoring
export TRACELET_TRACK_SYSTEM="true"
export TRACELET_METRICS_INTERVAL="10.0"
```

### Backend-Specific Environment Variables

```bash
# MLflow
export TRACELET_MLFLOW_TRACKING_URI="http://mlflow-server:5000"
export TRACELET_MLFLOW_EXPERIMENT_NAME="Production Experiments"

# Weights & Biases
export TRACELET_WANDB_PROJECT="production-ml"
export TRACELET_WANDB_ENTITY="company-ml-team"
export WANDB_API_KEY="your_wandb_api_key"

# ClearML
export TRACELET_CLEARML_PROJECT_NAME="Production Models"
export TRACELET_CLEARML_TASK_NAME="Training Session"
export CLEARML_API_ACCESS_KEY="your_access_key"
export CLEARML_API_SECRET_KEY="your_secret_key"

# AIM
export TRACELET_AIM_REPO_PATH="./aim_logs"
export TRACELET_AIM_EXPERIMENT_NAME="Local Experiments"
```

## Configuration Files

### YAML Configuration

Create a `tracelet.yaml` file in your project root:

```yaml
# tracelet.yaml
project: "ml_pipeline"
experiment_name: "automated_training"
backend:
  - "mlflow"
  - "wandb"

# System monitoring
track_system: true
metrics_interval: 15.0
track_gpu: true

# Backend configurations
mlflow_tracking_uri: "http://localhost:5000"
mlflow_experiment_name: "Pipeline Experiments"

wandb_project: "ml-pipeline"
wandb_entity: "research-team"
```

### Loading Configuration

```python
from tracelet.settings import TraceletSettings
import tracelet

# Load from file (automatically detected)
settings = TraceletSettings()  # Loads tracelet.yaml if present

# Explicit file loading
settings = TraceletSettings(_env_file=".env.production")

# Override specific fields
settings = TraceletSettings(
    backend=["mlflow"],  # Override YAML setting
    track_system=False   # Override YAML setting
)

# Use with tracelet
tracelet.start_logging(
    exp_name="config_example",
    settings=settings
)
```

## Advanced Configuration Patterns

### Environment-Specific Configurations

```python
import os
from tracelet.settings import TraceletSettings

# Different configs for different environments
if os.getenv("ENVIRONMENT") == "production":
    settings = TraceletSettings(
        project="production_models",
        backend=["mlflow", "clearml"],  # Multiple backends for production
        track_system=True,
        mlflow_tracking_uri="http://prod-mlflow:5000"
    )
elif os.getenv("ENVIRONMENT") == "development":
    settings = TraceletSettings(
        project="dev_experiments",
        backend=["mlflow"],  # Single backend for dev
        track_system=False,  # No system tracking in dev
        mlflow_tracking_uri="http://localhost:5000"
    )
else:
    settings = TraceletSettings(project="local_tests", backend=["mlflow"])
```

### Dynamic Configuration

```python
from tracelet.settings import TraceletSettings

def create_experiment_settings(experiment_type: str) -> TraceletSettings:
    """Create settings based on experiment type."""

    base_settings = {
        "project": "adaptive_experiments",
        "track_system": True
    }

    if experiment_type == "hyperparameter_search":
        return TraceletSettings(
            **base_settings,
            backend=["wandb"],  # W&B for hyperparameter tracking
            wandb_project="hyperparameter-optimization"
        )
    elif experiment_type == "model_comparison":
        return TraceletSettings(
            **base_settings,
            backend=["mlflow", "clearml"],  # Multiple backends for comparison
            metrics_interval=5.0  # More frequent monitoring
        )
    else:
        return TraceletSettings(**base_settings, backend=["mlflow"])

# Usage
settings = create_experiment_settings("hyperparameter_search")
tracelet.start_logging(exp_name="hp_search_001", settings=settings)
```

### Validation and Error Handling

```python
from tracelet.settings import TraceletSettings
from pydantic import ValidationError

try:
    # This will raise ValidationError for invalid backend
    settings = TraceletSettings(
        project="test",
        backend=["invalid_backend"]
    )
except ValidationError as e:
    print(f"Configuration error: {e}")
    # Use default settings
    settings = TraceletSettings(project="test")

# Check available backends
valid_backends = TraceletSettings.__annotations__["backend"].__args__[0].__args__
print(f"Valid backends: {valid_backends}")
```

## Best Practices

### Security

```python
# ❌ Don't store API keys in code
settings = TraceletSettings(
    wandb_api_key="sk-1234567890"  # Bad!
)

# ✅ Use environment variables
import os
settings = TraceletSettings(
    wandb_api_key=os.getenv("WANDB_API_KEY")  # Good!
)

# ✅ Better: Let Pydantic handle it automatically
settings = TraceletSettings()  # Reads TRACELET_WANDB_API_KEY from env
```

### Configuration Hierarchy

```python
# Configuration priority (highest to lowest):
# 1. Direct parameter override
# 2. Environment variables
# 3. Configuration file
# 4. Default values

settings = TraceletSettings(
    project="direct_override",  # 1. Highest priority
    # TRACELET_BACKEND env var   # 2. Middle priority
    # tracelet.yaml file         # 3. Lower priority
    # Default: ["mlflow"]        # 4. Lowest priority
)
```
