# MLflow Backend

MLflow is an open-source platform for managing the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry.

## Overview

MLflow provides comprehensive experiment tracking with:

- Local and remote tracking servers
- Model registry for versioning
- Artifact storage and management
- Experiment comparison and analysis
- Production deployment capabilities

Perfect for traditional ML workflows and production environments.

## Installation

=== "pip"
`bash
    pip install tracelet[mlflow]
    # or
    pip install tracelet mlflow
    `

=== "uv"
`bash
    uv add tracelet[mlflow]
    # or
    uv add tracelet mlflow
    `

**Minimum Requirements:**

- MLflow >= 3.1.1
- Python >= 3.9

## Quick Start

### Local Tracking

```python
import tracelet
from torch.utils.tensorboard import SummaryWriter

# Start with local MLflow tracking
tracelet.start_logging(
    exp_name="mlflow_experiment",
    project="my_project",
    backend="mlflow"
)

# Use TensorBoard as normal - metrics go to MLflow
writer = SummaryWriter()
writer.add_scalar("loss", 0.5, 1)
writer.add_scalar("accuracy", 0.9, 1)

tracelet.stop_logging()
```

### View Results

Start the MLflow UI:

```bash
mlflow ui
```

Visit `http://localhost:5000` to view your experiments.

## Configuration

### Local Setup (Default)

```python
# Uses local mlruns directory
tracelet.start_logging(
    backend="mlflow",
    exp_name="local_experiment",
    project="my_project"
)
```

### Remote Tracking Server

```python
import os

# Set tracking URI for remote server
os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow-server:5000"

tracelet.start_logging(
    backend="mlflow",
    exp_name="remote_experiment",
    project="distributed_training"
)
```

### Environment Variables

```bash
# MLflow server configuration
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_NAME="Default Experiment"

# Optional: Authentication for hosted MLflow
export MLFLOW_TRACKING_USERNAME=username
export MLFLOW_TRACKING_PASSWORD=password
export MLFLOW_TRACKING_TOKEN=your_token
```

### Backend-Specific Settings

```python
# Available via TraceletSettings
import tracelet

tracelet.start_logging(
    backend="mlflow",
    exp_name="configured_experiment",
    config={
        "backend_url": "http://localhost:5000",  # Custom tracking URI
        "experiment_name": "Custom Experiment", # MLflow experiment name
        "run_name": "baseline_run",             # Optional run name
        "tags": {"team": "ml", "version": "v1.0"}
    }
)
```

## Features

### Automatic Metric Logging

All TensorBoard calls are automatically captured:

```python
writer = SummaryWriter()

# Scalars
writer.add_scalar("train/loss", loss, epoch)
writer.add_scalar("val/accuracy", acc, epoch)

# Histograms
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

### Artifact Management

```python
# Log model artifacts
exp.log_artifact("model.pth", artifact_path="models/")

# Log entire directories
exp.log_artifact("checkpoints/", artifact_path="training/")

# Log config files
exp.log_artifact("config.yaml")
```

### Model Registry Integration

```python
import mlflow.pytorch

# Log model to registry
mlflow.pytorch.log_model(
    model,
    "model",
    registered_model_name="MyModel"
)
```

## Advanced Features

### Experiment Organization

```python
# Organize experiments by project
tracelet.start_logging(
    backend="mlflow",
    project="computer_vision",        # Sets experiment name
    exp_name="resnet_baseline",       # Sets run name
    config={
        "tags": {
            "model": "resnet50",
            "dataset": "cifar10"
        }
    }
)
```

### Multi-Run Experiments

```python
# Parameter sweep
for lr in [0.001, 0.01, 0.1]:
    tracelet.start_logging(
        backend="mlflow",
        exp_name=f"lr_sweep_{lr}",
        config={"tags": {"learning_rate": lr}}
    )
    # ... training with this LR ...
    tracelet.stop_logging()
```

### Nested Runs

```python
# Parent run for hyperparameter sweep
with mlflow.start_run(run_name="hyperparameter_sweep"):
    for params in param_grid:
        with mlflow.start_run(run_name=f"child_{params}", nested=True):
            tracelet.start_logging(backend="mlflow")
            # ... train with params ...
            tracelet.stop_logging()
```

## Deployment Options

### Local Development

```bash
# Basic UI
mlflow ui

# Custom port and host
mlflow ui --host 0.0.0.0 --port 5001
```

### Docker Deployment

```yaml
# docker-compose.yml
version: "3.8"
services:
  mlflow:
    image: python:3.11-slim
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlflow/mlruns
      - ./artifacts:/mlflow/artifacts
    working_dir: /mlflow
    command: |
      sh -c "
        pip install mlflow==3.1.1 &&
        mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root /mlflow/artifacts
      "
```

### Production Server

```bash
# Install MLflow server
pip install mlflow

# Start tracking server with artifact storage
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --default-artifact-root s3://my-mlflow-bucket/artifacts \
  --backend-store-uri postgresql://user:password@postgres:5432/mlflow
```

## Best Practices

### Experiment Naming

```python
# Use hierarchical naming
tracelet.start_logging(
    backend="mlflow",
    project="image_classification",  # Experiment name
    exp_name="resnet50_baseline_v1", # Run name
)
```

### Parameter Organization

```python
# Structure parameters by category
exp.log_params({
    "model.architecture": "resnet50",
    "model.layers": 50,
    "data.dataset": "cifar10",
    "data.augmentation": True,
    "training.optimizer": "adam",
    "training.learning_rate": 0.001,
    "training.batch_size": 32
})
```

### Metric Naming

```python
# Use consistent metric naming
writer.add_scalar("train/loss/total", total_loss, step)
writer.add_scalar("train/loss/classification", cls_loss, step)
writer.add_scalar("train/metrics/accuracy", acc, step)
writer.add_scalar("val/metrics/f1_score", f1, step)
```

### Artifact Organization

```python
# Organize artifacts by type
exp.log_artifact("model.pth", artifact_path="models/")
exp.log_artifact("config.yaml", artifact_path="configs/")
exp.log_artifact("results.json", artifact_path="results/")
```

## Troubleshooting

### Common Issues

**Connection refused:**

```bash
# Start MLflow server
mlflow ui

# Or check if server is running
curl http://localhost:5000/health
```

**Permission denied on mlruns:**

```bash
# Fix permissions
chmod -R 755 mlruns
```

**Experiment already exists:**

```python
# MLflow will reuse existing experiments by name
# No action needed - this is expected behavior
```

### Performance Tuning

```bash
# For high-throughput logging
export MLFLOW_ENABLE_ASYNC_LOGGING=true

# Optimize database backend
mlflow server --backend-store-uri sqlite:///mlflow.db
```

## Comparison with Other Backends

| Feature          | MLflow | ClearML | W&B    | AIM    |
| ---------------- | ------ | ------- | ------ | ------ |
| Setup complexity | ⭐⭐⭐ | ⭐⭐    | ⭐⭐⭐ | ⭐⭐⭐ |
| Model registry   | ⭐⭐⭐ | ⭐⭐⭐  | ⭐⭐⭐ | ⭐     |
| Artifact storage | ⭐⭐⭐ | ⭐⭐⭐  | ⭐⭐⭐ | ⭐⭐   |
| Visualization    | ⭐⭐   | ⭐⭐⭐  | ⭐⭐⭐ | ⭐⭐⭐ |
| Production ready | ⭐⭐⭐ | ⭐⭐⭐  | ⭐⭐⭐ | ⭐⭐   |

## Migration

### From TensorBoard

```python
# Before: Pure TensorBoard
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("./runs")

# After: TensorBoard + MLflow via Tracelet
import tracelet
tracelet.start_logging(backend="mlflow")
writer = SummaryWriter()  # Same code!
```

### To Other Backends

```python
# Easy switch to different backend
# tracelet.start_logging(backend="mlflow")      # Old
tracelet.start_logging(backend="wandb")        # New
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

# Start MLflow tracking
tracelet.start_logging(
    exp_name="mlflow_pytorch_example",
    project="tutorials",
    backend="mlflow"
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

# Training with automatic MLflow logging
writer = SummaryWriter()

for epoch in range(10):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Metrics automatically sent to MLflow
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

# Save model artifact
torch.save(model.state_dict(), "model.pth")
exp.log_artifact("model.pth", artifact_path="models/")

# Cleanup
writer.close()
tracelet.stop_logging()

print("✅ Training completed! View results with: mlflow ui")
```

## Next Steps

- [Compare with other backends](../examples/multi-backend.md)
- [Learn about MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Set up MLflow in production](https://mlflow.org/docs/latest/tracking.html#mlflow-tracking-servers)
