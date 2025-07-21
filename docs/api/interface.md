# Main Interface

::: tracelet.interface
options:
show_source: true
show_bases: true
heading_level: 2

## Main Public Functions

The main interface provides three primary functions for experiment tracking:

### start_logging

::: tracelet.interface.start_logging
options:
show_source: true
heading_level: 3

**Example Usage:**

```python
import tracelet

# Basic usage with MLflow
experiment = tracelet.start_logging(
    exp_name="image_classification",
    project="computer_vision",
    backend="mlflow"
)

# With custom configuration
experiment = tracelet.start_logging(
    exp_name="hyperparameter_tuning",
    project="optimization",
    backend="wandb",
    config={
        "entity": "my_team",
        "tags": ["pytorch", "resnet"]
    }
)

# Multi-backend tracking
experiment = tracelet.start_logging(
    exp_name="model_comparison",
    project="research",
    backend=["mlflow", "wandb", "clearml"]
)
```

### get_active_experiment

::: tracelet.interface.get_active_experiment
options:
show_source: true
heading_level: 3

**Example Usage:**

```python
# Start tracking
tracelet.start_logging(exp_name="my_exp", project="my_project", backend="mlflow")

# Get the active experiment from anywhere in your code
experiment = tracelet.get_active_experiment()

if experiment:
    experiment.log_metric("loss", 0.1, iteration=50)
    experiment.log_params({"learning_rate": 0.001})
else:
    print("No active experiment found")
```

### stop_logging

::: tracelet.interface.stop_logging
options:
show_source: true
heading_level: 3

**Example Usage:**

```python
# Stop the current experiment
tracelet.stop_logging()

# Verify no active experiment
assert tracelet.get_active_experiment() is None
```

## Integration Patterns

### Context Manager Pattern

```python
import tracelet

# Using with automatic cleanup
with tracelet.start_logging(exp_name="context_exp", project="test", backend="mlflow") as exp:
    exp.log_metric("accuracy", 0.95)
    exp.log_params({"epochs": 10})
# Automatically calls stop_logging() when exiting context
```

### Error Handling

```python
import tracelet

try:
    tracelet.start_logging(
        exp_name="my_experiment",
        project="my_project",
        backend="invalid_backend"
    )
except ValueError as e:
    print(f"Backend error: {e}")
    # Fall back to default backend
    tracelet.start_logging(
        exp_name="my_experiment",
        project="my_project",
        backend="mlflow"  # Default fallback
    )
```

### Configuration-Based Setup

```python
from tracelet.settings import TraceletSettings
import tracelet

# Load configuration
settings = TraceletSettings(
    project="ml_pipeline",
    backend=["mlflow", "wandb"],
    track_system=True,
    metrics_interval=10.0
)

# Use settings
experiment = tracelet.start_logging(
    exp_name="pipeline_run_001",
    settings=settings
)
```
