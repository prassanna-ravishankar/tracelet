# Settings and Configuration

Tracelet provides flexible configuration options through multiple interfaces: programmatic settings, environment variables, and configuration objects. This guide covers all available configuration options and how to use them effectively.

## Configuration Methods

### 1. Environment Variables

The simplest way to configure Tracelet globally:

```bash
# Core settings
export TRACELET_PROJECT="my_ml_project"
export TRACELET_BACKEND="mlflow,wandb"

# Feature toggles
export TRACELET_ENABLE_AUTOMAGIC="true"
export TRACELET_ENABLE_ARTIFACTS="true"
export TRACELET_TRACK_SYSTEM="true"
export TRACELET_TRACK_GIT="true"

# Automagic configuration
export TRACELET_AUTOMAGIC_FRAMEWORKS="pytorch,sklearn"

# Artifact configuration
export TRACELET_AUTOMAGIC_ARTIFACTS="true"
export TRACELET_WATCH_FILESYSTEM="false"
export TRACELET_ARTIFACT_WATCH_DIRS="./checkpoints,./outputs"
```

### 2. TraceletSettings Class

Programmatic configuration for application-wide settings:

```python
from tracelet.settings import TraceletSettings

# Create settings instance
settings = TraceletSettings(
    project="my_project",
    backend=["mlflow", "wandb"],
    track_system=True,
    track_git=True,
    track_env=True,
    enable_automagic=True,
    automagic_frameworks={"pytorch", "sklearn", "xgboost"},
    enable_artifacts=True,
    automagic_artifacts=True,
    watch_filesystem=False,
    artifact_watch_dirs=["./checkpoints", "./outputs", "./models"]
)

# Apply settings globally
tracelet.configure(settings)
```

### 3. ExperimentConfig Class

Fine-grained configuration for individual experiments:

```python
from tracelet.core.experiment import ExperimentConfig

config = ExperimentConfig(
    # Core tracking
    track_metrics=True,
    track_environment=True,
    track_args=True,
    track_stdout=False,  # Disable stdout capture
    track_checkpoints=True,
    track_system_metrics=True,
    track_git=True,

    # Automagic settings
    enable_automagic=True,
    automagic_frameworks={"pytorch", "sklearn"},

    # Artifact settings
    enable_artifacts=True,
    automagic_artifacts=True,
    watch_filesystem=True,  # Enable for this experiment
    artifact_watch_dirs=["./experiment_outputs"]
)

# Use with experiment
exp = Experiment("detailed_config", config=config)
```

### 4. Direct Parameters

Pass configuration directly to Experiment constructor:

```python
exp = Experiment(
    name="quick_config",
    backend=["mlflow"],
    automagic=True,
    artifacts=True,
    automagic_artifacts=True,
    tags=["experiment", "pytorch"]
)
```

## Configuration Reference

### Core Settings

| Setting                | Type        | Default | Description                          |
| ---------------------- | ----------- | ------- | ------------------------------------ |
| `project`              | `str`       | `None`  | Default project name for experiments |
| `backend`              | `List[str]` | `[]`    | List of backend names to use         |
| `track_metrics`        | `bool`      | `True`  | Enable metric tracking               |
| `track_environment`    | `bool`      | `True`  | Capture environment information      |
| `track_args`           | `bool`      | `True`  | Track command line arguments         |
| `track_stdout`         | `bool`      | `True`  | Capture stdout output                |
| `track_checkpoints`    | `bool`      | `True`  | Monitor checkpoint saves             |
| `track_system_metrics` | `bool`      | `True`  | Monitor system resources             |
| `track_git`            | `bool`      | `True`  | Capture git repository state         |

### Automagic Settings

| Setting                | Type       | Default       | Description                               |
| ---------------------- | ---------- | ------------- | ----------------------------------------- |
| `enable_automagic`     | `bool`     | `False`       | Enable automatic hyperparameter detection |
| `automagic_frameworks` | `Set[str]` | `{"pytorch"}` | Frameworks to instrument                  |

**Supported Frameworks:**

- `pytorch` - PyTorch tensors, models, optimizers
- `sklearn` - Scikit-learn estimators and datasets
- `xgboost` - XGBoost models and parameters
- `lightgbm` - LightGBM models and parameters
- `tensorflow` - TensorFlow models (experimental)

### Artifact Settings

| Setting               | Type        | Default | Description                          |
| --------------------- | ----------- | ------- | ------------------------------------ |
| `enable_artifacts`    | `bool`      | `False` | Enable artifact tracking system      |
| `automagic_artifacts` | `bool`      | `False` | Enable automatic artifact detection  |
| `watch_filesystem`    | `bool`      | `False` | Monitor filesystem for new artifacts |
| `artifact_watch_dirs` | `List[str]` | `[]`    | Directories to watch for artifacts   |

### System Settings

| Setting            | Type    | Default | Description                                  |
| ------------------ | ------- | ------- | -------------------------------------------- |
| `metrics_interval` | `float` | `10.0`  | System metrics collection interval (seconds) |
| `max_queue_size`   | `int`   | `10000` | Maximum metric queue size                    |
| `num_workers`      | `int`   | `4`     | Number of worker threads                     |

## Environment Variable Reference

All settings can be configured via environment variables using the `TRACELET_` prefix:

### Basic Configuration

```bash
# Project and backend
TRACELET_PROJECT="my_project"
TRACELET_BACKEND="mlflow,wandb,clearml"  # Comma-separated

# Feature toggles (true/false)
TRACELET_TRACK_SYSTEM="true"
TRACELET_TRACK_GIT="true"
TRACELET_TRACK_ENV="true"
TRACELET_TRACK_ARGS="true"
TRACELET_TRACK_STDOUT="false"
TRACELET_TRACK_CHECKPOINTS="true"
```

### Automagic Configuration

```bash
# Enable automagic
TRACELET_ENABLE_AUTOMAGIC="true"

# Framework selection (comma-separated)
TRACELET_AUTOMAGIC_FRAMEWORKS="pytorch,sklearn,xgboost"
```

### Artifact Configuration

```bash
# Enable artifacts
TRACELET_ENABLE_ARTIFACTS="true"
TRACELET_AUTOMAGIC_ARTIFACTS="true"

# Filesystem watching
TRACELET_WATCH_FILESYSTEM="true"
TRACELET_ARTIFACT_WATCH_DIRS="./checkpoints,./outputs,./models"
```

### System Configuration

```bash
# Performance tuning
TRACELET_METRICS_INTERVAL="5.0"
TRACELET_MAX_QUEUE_SIZE="20000"
TRACELET_NUM_WORKERS="8"
```

### Backend-Specific Configuration

```bash
# MLflow
TRACELET_MLFLOW_TRACKING_URI="http://localhost:5000"
TRACELET_MLFLOW_EXPERIMENT_NAME="my_experiment"

# Weights & Biases
TRACELET_WANDB_PROJECT="my_project"
TRACELET_WANDB_ENTITY="my_team"
TRACELET_WANDB_API_KEY="your_api_key"

# ClearML
TRACELET_CLEARML_PROJECT="my_project"
TRACELET_CLEARML_TASK_NAME="my_task"

# AIM
TRACELET_AIM_REPO="./aim_logs"
TRACELET_AIM_EXPERIMENT="my_experiment"
```

## Configuration Precedence

Configuration is resolved in the following order (higher precedence overrides lower):

1. **Direct parameters** (highest precedence)
2. **ExperimentConfig object**
3. **TraceletSettings object**
4. **Environment variables**
5. **Default values** (lowest precedence)

### Example

```python
# Environment variable
export TRACELET_ENABLE_AUTOMAGIC="false"

# Global settings
settings = TraceletSettings(enable_automagic=True)
tracelet.configure(settings)

# Experiment config
config = ExperimentConfig(enable_automagic=False)

# Direct parameter
exp = Experiment(
    "test",
    config=config,  # enable_automagic=False
    automagic=True  # This wins - highest precedence
)
```

Result: `automagic=True` (direct parameter overrides all others)

## Backend Configuration

### MLflow

```python
# Via settings
settings = TraceletSettings(
    backend=["mlflow"],
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="my_experiment"
)

# Via environment
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_EXPERIMENT_NAME="my_experiment"
```

### Weights & Biases

```python
# Via settings
settings = TraceletSettings(
    backend=["wandb"],
    wandb_project="my_project",
    wandb_entity="my_team"
)

# Via environment
export WANDB_PROJECT="my_project"
export WANDB_ENTITY="my_team"
export WANDB_API_KEY="your_api_key"
```

### ClearML

```python
# Via settings
settings = TraceletSettings(
    backend=["clearml"],
    clearml_project="my_project",
    clearml_task_name="my_task"
)

# Via environment
export CLEARML_PROJECT="my_project"
export CLEARML_TASK_NAME="my_task"
```

### AIM

```python
# Via settings
settings = TraceletSettings(
    backend=["aim"],
    aim_repo="./aim_logs",
    aim_experiment="my_experiment"
)

# Via environment
export AIM_REPO="./aim_logs"
export AIM_EXPERIMENT="my_experiment"
```

## Configuration Validation

Tracelet validates configuration at startup and provides helpful error messages:

```python
# Invalid backend
exp = Experiment("test", backend=["invalid_backend"])
# ConfigurationError: Unknown backend 'invalid_backend'.
# Available: ['mlflow', 'wandb', 'clearml', 'aim']

# Missing required dependencies
exp = Experiment("test", backend=["mlflow"])
# ConfigurationError: MLflow backend requires 'mlflow' package.
# Install with: pip install tracelet[mlflow]

# Invalid artifact directory
config = ExperimentConfig(
    enable_artifacts=True,
    artifact_watch_dirs=["/nonexistent/path"]
)
# ConfigurationError: Artifact watch directory does not exist: /nonexistent/path
```

## Configuration Files

### YAML Configuration

Create `tracelet.yaml`:

```yaml
project: "my_ml_project"
backend:
  - mlflow
  - wandb

tracking:
  system: true
  git: true
  environment: true

automagic:
  enabled: true
  frameworks:
    - pytorch
    - sklearn

artifacts:
  enabled: true
  automagic: true
  watch_filesystem: false
  watch_dirs:
    - "./checkpoints"
    - "./outputs"
```

Load configuration:

```python
import yaml
from tracelet.settings import TraceletSettings

with open("tracelet.yaml", "r") as f:
    config_dict = yaml.safe_load(f)

settings = TraceletSettings.from_dict(config_dict)
tracelet.configure(settings)
```

### JSON Configuration

Create `tracelet.json`:

```json
{
  "project": "my_ml_project",
  "backend": ["mlflow", "wandb"],
  "tracking": {
    "system": true,
    "git": true,
    "environment": true
  },
  "automagic": {
    "enabled": true,
    "frameworks": ["pytorch", "sklearn"]
  },
  "artifacts": {
    "enabled": true,
    "automagic": true,
    "watch_filesystem": false,
    "watch_dirs": ["./checkpoints", "./outputs"]
  }
}
```

Load configuration:

```python
import json
from tracelet.settings import TraceletSettings

with open("tracelet.json", "r") as f:
    config_dict = json.load(f)

settings = TraceletSettings.from_dict(config_dict)
tracelet.configure(settings)
```

## Best Practices

### 1. Use Environment Variables for Deployment

```bash
# Production environment
export TRACELET_PROJECT="production_models"
export TRACELET_BACKEND="mlflow,clearml"
export TRACELET_MLFLOW_TRACKING_URI="https://mlflow.company.com"

# Development environment
export TRACELET_PROJECT="dev_experiments"
export TRACELET_BACKEND="mlflow"
export TRACELET_MLFLOW_TRACKING_URI="http://localhost:5000"
```

### 2. Use ExperimentConfig for Experiment-Specific Settings

```python
# Default config for quick experiments
quick_config = ExperimentConfig(
    track_stdout=False,
    track_system_metrics=False,
    enable_automagic=True
)

# Comprehensive config for important experiments
detailed_config = ExperimentConfig(
    track_stdout=True,
    track_system_metrics=True,
    enable_automagic=True,
    enable_artifacts=True,
    automagic_artifacts=True,
    watch_filesystem=True
)

# Use based on experiment importance
if is_production_run:
    exp = Experiment("prod_model", config=detailed_config)
else:
    exp = Experiment("quick_test", config=quick_config)
```

### 3. Organize Settings by Context

```python
# Base settings
base_settings = TraceletSettings(
    project="my_project",
    track_system=True,
    track_git=True
)

# Development settings
dev_settings = base_settings.copy(
    backend=["mlflow"],
    enable_automagic=True
)

# Production settings
prod_settings = base_settings.copy(
    backend=["mlflow", "clearml"],
    enable_artifacts=True,
    automagic_artifacts=True
)
```

### 4. Validate Configuration Early

```python
def validate_experiment_config(config):
    """Validate experiment configuration before use."""
    if config.enable_artifacts and not config.backend:
        raise ValueError("Artifacts require at least one backend")

    if config.watch_filesystem and not config.artifact_watch_dirs:
        raise ValueError("Filesystem watching requires watch directories")

    return config

# Use validation
config = ExperimentConfig(enable_artifacts=True, watch_filesystem=True)
config = validate_experiment_config(config)  # Raises ValueError
```

## Troubleshooting

### Common Configuration Errors

**Backend not found**:

```python
# Error: Backend 'invalid' not found
exp = Experiment("test", backend=["invalid"])

# Fix: Use valid backend names
exp = Experiment("test", backend=["mlflow"])
```

**Missing dependencies**:

```python
# Error: MLflow not installed
exp = Experiment("test", backend=["mlflow"])

# Fix: Install backend dependencies
# pip install tracelet[mlflow]
```

**Invalid directories**:

```python
# Error: Directory does not exist
config = ExperimentConfig(artifact_watch_dirs=["/invalid/path"])

# Fix: Use existing directories or create them
import os
os.makedirs("./artifacts", exist_ok=True)
config = ExperimentConfig(artifact_watch_dirs=["./artifacts"])
```

### Debug Configuration

Enable debug logging to see configuration resolution:

```python
import logging
logging.getLogger("tracelet.settings").setLevel(logging.DEBUG)

# See detailed configuration loading
exp = Experiment("debug_config", automagic=True)
# DEBUG:tracelet.settings:Loading configuration from environment
# DEBUG:tracelet.settings:TRACELET_ENABLE_AUTOMAGIC=true -> enable_automagic=True
# DEBUG:tracelet.settings:Final configuration: enable_automagic=True
```

### Configuration Inspection

Check resolved configuration:

```python
exp = Experiment("test", automagic=True, artifacts=True)
exp.start()

# Inspect final configuration
print(f"Automagic enabled: {exp._automagic_enabled}")
print(f"Artifacts enabled: {exp._artifacts_enabled}")
print(f"Backends: {exp._backends}")
print(f"Configuration: {exp.config}")
```
