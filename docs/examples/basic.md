# Basic Usage Examples

This page provides practical examples of using Tracelet for common experiment tracking scenarios.

## Quick Start Example

```python
import tracelet
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Start experiment tracking
tracelet.start_logging(
    exp_name="basic_example",
    project="getting_started",
    backend="mlflow"
)

# Create a simple model and training setup
model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Use TensorBoard as normal - metrics are automatically captured!
writer = SummaryWriter()

# Training loop with automatic metric capture
for epoch in range(50):
    # Synthetic training data
    X = torch.randn(32, 10)
    y = torch.randn(32, 1)

    # Forward pass
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    # Log metrics - automatically sent to MLflow!
    writer.add_scalar('Loss/Train', loss.item(), epoch)
    writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

# Log additional experiment metadata
exp = tracelet.get_active_experiment()
exp.log_params({
    "learning_rate": 0.01,
    "batch_size": 32,
    "epochs": 50,
    "model_type": "linear"
})

# Clean up
writer.close()
tracelet.stop_logging()

print("✅ Experiment completed! Check your MLflow UI at http://localhost:5000")
```

## Configuration Examples

### Environment Variables

```bash
# Set default backend
export TRACELET_BACKEND=mlflow

# Set project name
export TRACELET_PROJECT=my_ml_project

# Backend-specific configuration
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### Programmatic Configuration

```python
from tracelet.settings import TraceletSettings

# Create custom settings
settings = TraceletSettings(
    project="advanced_project",
    backend=["mlflow"],
    track_system=True,
    metrics_interval=5.0
)

# Use settings
tracelet.start_logging(
    exp_name="configured_experiment",
    settings=settings
)
```

## Manual Metric Logging

```python
import tracelet

# Start experiment
exp = tracelet.start_logging(
    exp_name="manual_logging",
    project="examples",
    backend="mlflow"
)

# Log metrics manually
exp.log_metric("accuracy", 0.95, iteration=100)
exp.log_metric("loss", 0.05, iteration=100)

# Log parameters
exp.log_params({
    "learning_rate": 0.001,
    "batch_size": 64,
    "optimizer": "adam"
})

# Log artifacts
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [1, 4, 2])
plt.savefig("training_plot.png")
exp.log_artifact("training_plot.png", "plots/training_curve.png")

tracelet.stop_logging()
```

## Error Handling

```python
import tracelet

try:
    tracelet.start_logging(
        exp_name="robust_experiment",
        project="error_handling",
        backend="mlflow"
    )

    # Your training code here
    for epoch in range(10):
        # Simulate potential error
        if epoch == 5:
            raise ValueError("Simulated training error")

        exp = tracelet.get_active_experiment()
        exp.log_metric("epoch", epoch, iteration=epoch)

except Exception as e:
    print(f"Training failed: {e}")

    # Log error information
    exp = tracelet.get_active_experiment()
    if exp:
        exp.log_params({"error": str(e), "failed_at_epoch": epoch})

finally:
    # Always clean up
    tracelet.stop_logging()
```

## Context Manager Usage

```python
import tracelet

# Automatic cleanup using context manager
with tracelet.start_logging(
    exp_name="context_managed",
    project="examples",
    backend="mlflow"
) as exp:
    # Training code here
    exp.log_metric("start_time", time.time())

    for epoch in range(10):
        exp.log_metric("epoch_loss", 1.0 / (epoch + 1), iteration=epoch)

    exp.log_params({"total_epochs": 10})

# Automatic cleanup when exiting context
```

## Multi-Metric Logging

```python
import tracelet
from torch.utils.tensorboard import SummaryWriter

tracelet.start_logging(
    exp_name="multi_metric",
    project="examples",
    backend="mlflow"
)

writer = SummaryWriter()

for epoch in range(20):
    # Log multiple related metrics
    train_loss = 1.0 / (epoch + 1)
    val_loss = train_loss * 1.1
    accuracy = min(0.95, epoch * 0.05)

    # Batch logging with TensorBoard
    writer.add_scalars('Loss', {
        'Train': train_loss,
        'Validation': val_loss
    }, epoch)

    writer.add_scalar('Metrics/Accuracy', accuracy, epoch)
    writer.add_scalar('Metrics/LearningRate', 0.01 * (0.9 ** epoch), epoch)

writer.close()
tracelet.stop_logging()
```

## Reproducibility Example

```python
import tracelet
import torch
import numpy as np
import random

# Set seeds for reproducibility
def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seeds(42)

# Start experiment with reproducibility info
exp = tracelet.start_logging(
    exp_name="reproducible_experiment",
    project="reproducibility",
    backend="mlflow"
)

# Log reproducibility parameters
exp.log_params({
    "random_seed": 42,
    "pytorch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
})

# Your training code here...
model = torch.nn.Linear(5, 1)
data = torch.randn(100, 5)
target = torch.randn(100, 1)

output = model(data)
loss = torch.nn.functional.mse_loss(output, target)

exp.log_metric("initial_loss", loss.item())

tracelet.stop_logging()
```

## System Metrics Monitoring

```python
import tracelet
import time

# Enable system metrics collection
from tracelet.settings import TraceletSettings

settings = TraceletSettings(
    project="system_monitoring",
    backend=["mlflow"],
    track_system=True,
    metrics_interval=5.0  # Collect every 5 seconds
)

tracelet.start_logging(
    exp_name="monitored_training",
    settings=settings
)

# Simulate training workload
for i in range(10):
    # Simulate some work
    time.sleep(2)

    # Log training progress
    exp = tracelet.get_active_experiment()
    exp.log_metric("training_step", i, iteration=i)

tracelet.stop_logging()
print("Check your MLflow UI to see system metrics alongside training metrics!")
```

## Next Steps

- Try the [Multi-Backend Example](multi-backend.md) to track to multiple platforms
- Explore [Backend-Specific Guides](../backends/index.md) for advanced features
- Check out the [Interactive Notebooks](notebooks.md) for hands-on tutorials
- Review [Best Practices](../guides/best-practices.md) for production usage
