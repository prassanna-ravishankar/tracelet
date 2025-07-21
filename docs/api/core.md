# Core API Reference

This section covers the core APIs and classes that form the foundation of Tracelet.

## Main Interface

### `tracelet.start_logging()`

Start experiment tracking with the specified configuration.

**Parameters:**

- `exp_name: str` - Name of the experiment
- `project: str` - Project name (optional)
- `backend: str | List[str]` - Backend(s) to use
- `config: dict` - Additional configuration (optional)

**Returns:** `Experiment` - The created experiment instance

### `tracelet.stop_logging()`

Stop the current experiment tracking session.

### `tracelet.get_active_experiment()`

Get the currently active experiment instance.

**Returns:** `Experiment | None` - The active experiment, or None if no experiment is active

## Core Classes

### Experiment

The main experiment tracking interface.

**Key Methods:**

- `log_metric(name, value, step)` - Log a scalar metric
- `log_params(params)` - Log experiment parameters
- `log_artifact(path)` - Log an artifact file
- `log_image(name, image, step)` - Log an image
- `log_text(name, text, step)` - Log text data

_Full API documentation coming soon._

### Orchestrator

The Orchestrator class manages metric routing and backend coordination.

**Key Methods:**

- `start()` - Start the orchestrator and worker threads
- `stop()` - Stop all operations gracefully
- `route_metric(metric)` - Route a metric to all configured backends
- `add_backend(backend)` - Add a new backend to the orchestrator
- `remove_backend(backend)` - Remove a backend from the orchestrator

_Full API documentation coming soon._

## Plugin System

### Plugin Interfaces

Tracelet uses a plugin-based architecture for extensibility.

**PluginInterface** - Base interface for all plugins
**BackendPlugin** - Interface for experiment tracking backends
**FrameworkPlugin** - Interface for ML framework integrations

### Plugin Metadata

**PluginMetadata** - Contains plugin information (name, version, description)
**PluginType** - Enum defining plugin types (BACKEND, FRAMEWORK, COLLECTOR)

_Full plugin API documentation coming soon._

## Configuration

### Settings

**TraceletSettings** - Main configuration class with these key settings:

- `project: str` - Default project name
- `backend: List[str]` - Default backends to use
- `track_system: bool` - Enable system metrics tracking
- `track_git: bool` - Enable git repository tracking
- `track_env: bool` - Enable environment tracking
- `metrics_interval: float` - System metrics collection interval

## Exceptions

### Base Exceptions

**TraceletException** - Base exception for all Tracelet errors
**BackendError** - Errors related to backend operations
**ConfigurationError** - Errors in configuration or setup

_Full exception API documentation coming soon._

## Usage Examples

### Basic Usage

```python
import tracelet
from torch.utils.tensorboard import SummaryWriter

# Start experiment tracking
experiment = tracelet.start_logging(
    exp_name="my_experiment",
    project="my_project",
    backend="mlflow"
)

# Use TensorBoard normally - metrics automatically captured
writer = SummaryWriter()
writer.add_scalar('loss', 0.5, 0)
writer.add_scalar('accuracy', 0.95, 0)

# Direct API usage
experiment.log_params({
    "learning_rate": 0.01,
    "batch_size": 32
})

experiment.log_artifact("model.pth")

# Stop tracking
tracelet.stop_logging()
```

### Advanced Configuration

```python
import tracelet

# Multi-backend logging with custom configuration
experiment = tracelet.start_logging(
    exp_name="advanced_experiment",
    project="research_project",
    backend=["mlflow", "wandb"],
    config={
        "track_system": True,
        "track_git": True,
        "track_env": True,
        "metrics_interval": 10.0,
        "mlflow_tracking_uri": "http://localhost:5000",
        "wandb_project": "my-wandb-project"
    }
)

# Get current experiment for direct manipulation
current_exp = tracelet.get_active_experiment()
print(f"Experiment ID: {current_exp.experiment_id}")
print(f"Active backends: {[b.name for b in current_exp.backends]}")
```

### Context Manager Usage

```python
import tracelet

# Use as context manager for automatic cleanup
with tracelet.start_logging("context_experiment", backend="mlflow") as exp:
    # Training code here
    for epoch in range(10):
        loss = train_epoch()
        exp.log_metric("loss", loss, epoch)

    # Experiment automatically closed when exiting context
```

## Type Hints

Tracelet provides comprehensive type hints for better IDE support:

```python
from typing import Dict, List, Optional, Union
from tracelet.core.experiment import Experiment
from tracelet.core.plugins import BackendPlugin

def my_training_function(
    experiment: Experiment,
    hyperparams: Dict[str, Union[str, int, float]],
    backends: Optional[List[str]] = None
) -> None:
    """Example function with proper type hints"""
    experiment.log_params(hyperparams)
    # Training logic here
```

For more detailed information about specific modules, see the dedicated API reference pages for each component.
