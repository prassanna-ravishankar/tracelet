# AIM Backend

AIM is a lightweight, open-source experiment tracking system optimized for speed and simplicity.

## Overview

AIM provides fast experiment tracking with a focus on performance and ease of use. It's perfect for:

- Local development and experimentation
- High-frequency metric logging
- Simple deployment scenarios
- Teams that prefer open-source solutions

## Installation

=== "pip"
`bash
    pip install tracelet aim
    `

=== "uv"
`bash
    uv add tracelet aim
    `

## Quick Start

### Local Repository

```python
import tracelet
from torch.utils.tensorboard import SummaryWriter

# Start with local AIM repository
tracelet.start_logging(
    exp_name="aim_experiment",
    project="my_project",
    backend="aim"
)

# Use TensorBoard as normal - metrics go to AIM
writer = SummaryWriter()
writer.add_scalar("loss", 0.5, 1)
writer.add_scalar("accuracy", 0.9, 1)

tracelet.stop_logging()
```

### View Results

Start the AIM UI:

```bash
aim up
```

Visit `http://localhost:43800` to view your experiments.

!!! note "AIM Ports" - UI server (`aim up`): Default port `43800` - API server: Default port `53800`

## Configuration

!!! warning "Current Limitation"
Backend-specific configuration is currently not supported. The AIM backend uses default settings: - Repository path: Current directory (`.`) - Experiment name: "Tracelet Experiments" - Remote server: Not supported

    Advanced configuration will be added in a future release.

### Current Usage

```python
# This works with default settings
tracelet.start_logging(
    backend="aim",
    exp_name="my_experiment",     # Sets experiment run name
    project="my_project"          # Sets project context
)
```

### Planned Configuration (Future Release)

```python
# This will be supported in future versions
tracelet.start_logging(
    backend="aim",
    config={
        "repo_path": "./aim_repo",           # Custom repository path
        "experiment_name": "My Experiments", # Custom experiment name
        "run_name": "baseline_run",         # Custom run name
        "tags": {                           # Run tags
            "model": "resnet",
            "dataset": "cifar10"
        },
        "remote_uri": "http://aim-server:53800"  # Remote server
    }
)
```

## Features

### Metrics Logging

AIM automatically captures all TensorBoard metrics:

```python
writer = SummaryWriter()

# Scalars
writer.add_scalar("train/loss", loss, epoch)
writer.add_scalar("val/accuracy", acc, epoch)

# Histograms (converted to AIM distributions)
writer.add_histogram("model/weights", model.parameters(), epoch)

# Images
writer.add_image("predictions", image_tensor, epoch)

# Text
writer.add_text("notes", "Training progressing well", epoch)
```

### Parameter Logging

```python
exp = tracelet.get_active_experiment()
exp.log_params({
    "learning_rate": 0.001,
    "batch_size": 32,
    "optimizer": "adam",
    "model_architecture": "resnet50"
})
```

### Artifact Storage

```python
# Log model artifacts
exp.log_artifact("model.pth", artifact_path="models/")

# Log dataset info
exp.log_artifact("data_stats.json", artifact_path="data/")
```

!!! note "AIM Artifact Limitations"
AIM doesn't have full artifact storage like MLflow. Files are referenced by path rather than uploaded to a central store.

## Advanced Features

### Context-based Metrics

AIM supports rich metric contexts for better organization:

```python
# Metrics are automatically organized by source and name
writer.add_scalar("train/loss", loss, step)      # Context: train
writer.add_scalar("val/loss", val_loss, step)    # Context: val
```

### High-Frequency Logging

AIM is optimized for high-frequency metric logging:

```python
# Log every batch without performance concerns
for batch_idx, (data, target) in enumerate(dataloader):
    # ... training code ...
    writer.add_scalar("batch/loss", batch_loss, batch_idx)
    writer.add_scalar("batch/lr", current_lr, batch_idx)
```

### Multi-Run Comparison

AIM's UI excels at comparing multiple runs:

```python
for lr in [0.001, 0.01, 0.1]:
    tracelet.start_logging(
        backend="aim",
        exp_name=f"lr_sweep_{lr}",
        config={"tags": {"learning_rate": lr}}
    )
    # ... training with this LR ...
    tracelet.stop_logging()
```

## Deployment Options

### Local Development

```bash
# Initialize repository
aim init

# Start tracking server
aim up --host 0.0.0.0 --port 43800
```

### Docker Deployment

```yaml
# docker-compose.yml
version: "3.8"
services:
  aim:
    image: aimhubio/aim:latest
    ports:
      - "43800:43800" # UI port
      - "53800:53800" # API port
    volumes:
      - ./aim_data:/opt/aim
    command: aim up --host 0.0.0.0 --port 43800
```

### Production Server

```bash
# Install AIM server
pip install aim

# Run UI server
aim up --host 0.0.0.0 --port 43800

# Or run API server for remote connections
aim server --host 0.0.0.0 --port 53800
```

## Best Practices

### Repository Organization

```python
# Current approach - organize by experiment name
tracelet.start_logging(
    backend="aim",
    exp_name="hyperparameter_tuning_resnet",
    project="computer_vision"
)

# Future: Custom repo paths (not currently supported)
# tracelet.start_logging(
#     backend="aim",
#     config={"repo_path": "./experiments/my_project"}
# )
```

### Parameter Logging Strategy

```python
# Use structured parameter names for organization
exp = tracelet.get_active_experiment()
exp.log_params({
    "model.architecture": "resnet50",
    "model.layers": 50,
    "data.dataset": "cifar10",
    "training.stage": "development",
    "training.version": "v1.2",
    "optimizer.name": "adam",
    "optimizer.lr": 0.001
})
```

### Metric Naming

```python
# Use hierarchical naming
writer.add_scalar("train/loss/total", loss, step)
writer.add_scalar("train/loss/classification", cls_loss, step)
writer.add_scalar("train/metrics/accuracy", acc, step)
writer.add_scalar("val/metrics/f1_score", f1, step)
```

## Troubleshooting

### Common Issues

**Repository not found:**

```bash
# Initialize AIM repository
aim init
```

**Port already in use:**

```bash
# Use different port
aim up --port 43801
```

**Remote connection failed:**
Currently not supported. Remote AIM server connections will be available in a future release when backend configuration is implemented.

### Performance Tuning

```python
# Current: AIM backend uses optimal defaults for performance
# Repository is created in current directory on fast local storage

# Future: Custom configuration will support
# config = {
#     "repo_path": "/fast/ssd/aim_repo",  # Use SSD storage
#     "buffer_size": 1000,               # Batch metrics
# }
```

## Comparison with Other Backends

| Feature               | AIM    | MLflow | ClearML | W&B    |
| --------------------- | ------ | ------ | ------- | ------ |
| Setup complexity      | ⭐⭐⭐ | ⭐⭐   | ⭐      | ⭐⭐⭐ |
| Logging performance   | ⭐⭐⭐ | ⭐⭐   | ⭐⭐    | ⭐⭐   |
| Visualization quality | ⭐⭐⭐ | ⭐⭐   | ⭐⭐⭐  | ⭐⭐⭐ |
| Query capabilities    | ⭐⭐⭐ | ⭐⭐   | ⭐⭐⭐  | ⭐⭐⭐ |
| Resource usage        | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐    | ⭐⭐   |

## Migration

### From TensorBoard

```python
# Before: Pure TensorBoard
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("./runs")

# After: TensorBoard + AIM via Tracelet
import tracelet
tracelet.start_logging(backend="aim")
writer = SummaryWriter()  # Same code!
```

### To Other Backends

```python
# Easy switch to different backend
# tracelet.start_logging(backend="aim")      # Old
tracelet.start_logging(backend="wandb")     # New
# All TensorBoard code remains unchanged
```

## Complete Example

```python
import tracelet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

# Start AIM tracking
tracelet.start_logging(
    exp_name="aim_pytorch_example",
    project="tutorials",
    backend="aim"
)

# Model setup
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Data
X = torch.randn(1000, 784)
y = torch.randint(0, 10, (1000,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32)

# Training with automatic AIM logging
writer = SummaryWriter()

for epoch in range(10):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Metrics automatically sent to AIM
        writer.add_scalar("batch/loss", loss.item(),
                         epoch * len(dataloader) + batch_idx)
        total_loss += loss.item()

    # Epoch metrics
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar("epoch/loss", avg_loss, epoch)

    # Model weights histogram
    for name, param in model.named_parameters():
        writer.add_histogram(f"weights/{name}", param, epoch)

# Log final parameters
exp = tracelet.get_active_experiment()
exp.log_params({
    "model.type": "simple_mlp",
    "model.input_size": 784,
    "model.hidden_size": 128,
    "model.output_size": 10,
    "optimizer.type": "adam",
    "optimizer.learning_rate": 0.001,
    "training.batch_size": 32,
    "training.epochs": 10
})

# Cleanup
writer.close()
tracelet.stop_logging()

print("✅ Training completed! View results with: aim up")
```

## Next Steps

- [View the multi-backend comparison example](../examples/multi-backend.md)
- [Learn about AIM's advanced querying features](https://aimstack.readthedocs.io/)
- [Set up remote AIM deployment](https://aimstack.readthedocs.io/en/latest/using/remote_tracking.html)
