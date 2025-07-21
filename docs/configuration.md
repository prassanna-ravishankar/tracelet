# Configuration

Tracelet can be configured in multiple ways to suit your workflow.

## Configuration Methods

### 1. Code Configuration

```python
import tracelet

# Basic configuration
tracelet.start_logging(
    exp_name="my_experiment",
    project="my_project",
    backend="mlflow",
    config={
        "tracking_uri": "http://localhost:5000",
        "experiment_name": "Deep Learning Experiments"
    }
)
```

### 2. Environment Variables

Set environment variables to configure defaults:

```bash
export TRACELET_BACKEND=mlflow
export TRACELET_PROJECT=my_default_project
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### 3. Configuration File

Create a `tracelet.yaml` file in your project root:

```yaml
# tracelet.yaml
backend: mlflow
project: my_project
experiment_name: my_experiment

# Backend-specific configuration
mlflow:
  tracking_uri: http://localhost:5000
  experiment_name: "ML Experiments"

clearml:
  project_name: "Tracelet Experiments"
  task_name: "Training Run"

wandb:
  project: "tracelet-experiments"
  entity: "your-username"

aim:
  repo_path: "./aim_logs"
  experiment_name: "Tracelet Experiments"
```

## Backend-Specific Configuration

### MLflow

```python
tracelet.start_logging(
    backend="mlflow",
    config={
        "tracking_uri": "http://localhost:5000",
        "experiment_name": "My Experiments",
        "run_name": "run_001",
        "tags": {"team": "ml", "version": "v1.0"}
    }
)
```

Environment variables:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_NAME="My Experiments"
```

### ClearML

```python
tracelet.start_logging(
    backend="clearml",
    config={
        "project_name": "Tracelet Experiments",
        "task_name": "Training Session",
        "tags": ["pytorch", "experiment"],
        "output_uri": "s3://my-bucket/clearml-output"
    }
)
```

Environment variables:

```bash
export CLEARML_WEB_HOST=https://app.clear.ml
export CLEARML_API_HOST=https://api.clear.ml
export CLEARML_FILES_HOST=https://files.clear.ml
export CLEARML_API_ACCESS_KEY=your_access_key
export CLEARML_API_SECRET_KEY=your_secret_key
```

### Weights & Biases

```python
tracelet.start_logging(
    backend="wandb",
    config={
        "project": "tracelet-experiments",
        "entity": "your-username",
        "name": "experiment_001",
        "tags": ["pytorch", "baseline"],
        "mode": "online"  # or "offline"
    }
)
```

Environment variables:

```bash
export WANDB_API_KEY=your_api_key
export WANDB_PROJECT=tracelet-experiments
export WANDB_ENTITY=your-username
```

### AIM

```python
tracelet.start_logging(
    backend="aim",
    config={
        "repo_path": "./aim_repo",
        "experiment_name": "Tracelet Experiments",
        "run_name": "baseline_run",
        "tags": {"model": "resnet", "dataset": "cifar10"}
    }
)
```

For remote AIM:

```python
tracelet.start_logging(
    backend="aim",
    config={
        "remote_uri": "http://aim-server:53800",
        "experiment_name": "Remote Experiments"
    }
)
```

## Advanced Configuration

### System Metrics Collection

```python
tracelet.start_logging(
    backend="mlflow",
    config={
        "collect_system_metrics": True,
        "system_metrics_interval": 30,  # seconds
        "collect_gpu_metrics": True
    }
)
```

### Git Integration

```python
tracelet.start_logging(
    backend="mlflow",
    config={
        "track_git_info": True,
        "git_repo_path": ".",
        "include_uncommitted_changes": True
    }
)
```

### Multi-Backend Configuration

```python
tracelet.start_logging(
    backend=["mlflow", "wandb"],
    config={
        "mlflow": {
            "tracking_uri": "http://localhost:5000"
        },
        "wandb": {
            "project": "multi-backend-experiment"
        }
    }
)
```

## Configuration Priority

Configuration values are resolved in this order (highest to lowest priority):

1. Direct function arguments
2. Environment variables
3. Configuration file (`tracelet.yaml`)
4. Default values

## Validation

Tracelet validates your configuration at startup:

```python
# Invalid configuration will raise an error
try:
    tracelet.start_logging(backend="nonexistent")
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Best Practices

1. **Environment-specific configs**: Use different config files for dev/staging/prod
2. **Secrets**: Store API keys and credentials in environment variables, not config files
3. **Project organization**: Use consistent project and experiment naming conventions
4. **Defaults**: Set sensible defaults in your config file to reduce boilerplate

## Configuration Reference

Complete configuration options for each backend:

[MLflow Configuration →](backends/mlflow.md#configuration)
[ClearML Configuration →](backends/clearml.md#configuration)
[W&B Configuration →](backends/wandb.md#configuration)
[AIM Configuration →](backends/aim.md#configuration)
