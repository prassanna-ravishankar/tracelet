# API Reference

Complete API documentation for Tracelet's public interfaces.

## Quick Navigation

### Core API

- [Main Interface](interface.md) - Primary entry points (`start_logging`, `get_active_experiment`, `stop_logging`)
- [Core Components](core.md) - Core classes and data flow management
- [Settings](../settings.md) - Configuration and settings management

### Backends

- [MLflow](../backends/mlflow.md) - MLflow integration
- [ClearML](../backends/clearml.md) - ClearML integration
- [Weights & Biases](../backends/wandb.md) - W&B integration
- [AIM](../backends/aim.md) - AIM integration

### Frameworks

- [PyTorch](../integrations/pytorch.md) - PyTorch and TensorBoard integration
- [Lightning](../integrations/lightning.md) - PyTorch Lightning integration

### Data Collection

- [Git Collector](collectors/git.md) - Git repository information
- [System Metrics](collectors/system.md) - System performance metrics

### Plugin System

- [Core Plugins](core/plugins.md) - Plugin architecture and interfaces
- [Data Flow Management](core.md) - Orchestrator and data flow

## Usage Examples

### Basic Usage

```python
import tracelet

# Start experiment tracking
tracelet.start_logging(
    exp_name="my_experiment",
    project="my_project",
    backend="mlflow"
)

# Get active experiment and log metrics
exp = tracelet.get_active_experiment()
exp.log_metric("accuracy", 0.95, iteration=100)
exp.log_params({"lr": 0.001, "batch_size": 32})

# Stop tracking
tracelet.stop_logging()
```

### Multi-Backend Usage

```python
import tracelet

# Track to multiple backends simultaneously
tracelet.start_logging(
    exp_name="multi_backend_experiment",
    project="comparison_study",
    backend=["mlflow", "wandb"]  # List of backends
)
```

### Advanced Configuration

```python
from tracelet.settings import TraceletSettings

settings = TraceletSettings(
    project="advanced_project",
    backend=["mlflow"],
    track_system=True,
    metrics_interval=5.0
)

tracelet.start_logging(
    exp_name="advanced_experiment",
    settings=settings
)
```
