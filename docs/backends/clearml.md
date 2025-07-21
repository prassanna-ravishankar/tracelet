# ClearML Backend

ClearML is an enterprise-grade MLOps platform that provides comprehensive experiment tracking, pipeline orchestration, and model management.

## Overview

ClearML offers powerful features for enterprise teams:

- Automatic experiment tracking and logging
- Rich visualization and comparison tools
- Pipeline orchestration and automation
- Model registry with versioning
- Resource management and scaling
- Comprehensive audit trails

Perfect for enterprise MLOps and automated ML pipelines.

## Installation

=== "pip"
`bash
    pip install tracelet[clearml]
    # or
    pip install tracelet clearml
    `

=== "uv"
`bash
    uv add tracelet[clearml]
    # or
    uv add tracelet clearml
    `

**Minimum Requirements:**

- ClearML >= 1.15.0
- Python >= 3.9

## Quick Start

### SaaS Platform (Recommended)

```python
import tracelet
from torch.utils.tensorboard import SummaryWriter

# Start with ClearML SaaS (app.clear.ml)
tracelet.start_logging(
    exp_name="clearml_experiment",
    project="my_project",
    backend="clearml"
)

# Use TensorBoard as normal - metrics go to ClearML
writer = SummaryWriter()
writer.add_scalar("loss", 0.5, 1)
writer.add_scalar("accuracy", 0.9, 1)

tracelet.stop_logging()
```

### View Results

Visit [app.clear.ml](https://app.clear.ml) to view your experiments in the web UI.

## Setup and Authentication

### Initial Setup

1. **Create account** at [app.clear.ml](https://app.clear.ml)
2. **Get credentials** from Settings → Workspace → Create new credentials
3. **Configure ClearML**:

```bash
# Interactive setup (recommended)
clearml-init

# Manual configuration
mkdir -p ~/.clearml
cat > ~/.clearml/clearml.conf << EOF
api {
    web_server: https://app.clear.ml
    api_server: https://api.clear.ml
    files_server: https://files.clear.ml
    credentials {
        "access_key" = "YOUR_ACCESS_KEY"
        "secret_key" = "YOUR_SECRET_KEY"
    }
}
EOF
```

### Environment Variables

```bash
# ClearML SaaS configuration
export CLEARML_WEB_HOST=https://app.clear.ml
export CLEARML_API_HOST=https://api.clear.ml
export CLEARML_FILES_HOST=https://files.clear.ml
export CLEARML_API_ACCESS_KEY=your_access_key
export CLEARML_API_SECRET_KEY=your_secret_key

# Optional: Project defaults
export CLEARML_PROJECT_NAME="Default Project"
export CLEARML_TASK_NAME="Default Task"
```

### Offline Mode

```python
import os

# Enable offline mode for development
os.environ["CLEARML_OFFLINE_MODE"] = "1"

tracelet.start_logging(
    backend="clearml",
    exp_name="offline_experiment",
    project="development"
)
```

## Configuration

### Basic Configuration

```python
tracelet.start_logging(
    backend="clearml",
    exp_name="my_experiment",
    project="computer_vision",  # ClearML project name
    config={
        "task_name": "resnet_training",     # Custom task name
        "tags": ["pytorch", "baseline"],    # Task tags
        "task_type": "training",            # Task type
        "output_uri": "s3://my-bucket/"     # Artifact storage
    }
)
```

### Advanced Configuration

```python
# Comprehensive setup
tracelet.start_logging(
    backend="clearml",
    exp_name="advanced_experiment",
    project="ml_research",
    config={
        "task_name": "hyperparameter_sweep",
        "task_type": "training",
        "tags": ["experiment", "sweep", "v2.0"],
        "output_uri": "s3://clearml-artifacts/",
        "auto_connect_frameworks": True,     # Auto-detect frameworks
        "auto_connect_arg_parser": True,     # Capture argparse
        "continue_last_task": False,         # Create new task
        "reuse_last_task_id": False
    }
)
```

## Features

### Automatic Framework Detection

ClearML automatically detects and logs from popular frameworks:

```python
# These are automatically captured when detected:
# - PyTorch models and hyperparameters
# - TensorBoard metrics and plots
# - Matplotlib figures
# - Pandas DataFrames
# - scikit-learn models
# - Command-line arguments
```

### Manual Logging

```python
exp = tracelet.get_active_experiment()

# Parameters
exp.log_params({
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "optimizer": "adam"
})

# Artifacts
exp.log_artifact("model.pth", artifact_path="models/")
exp.log_artifact("config.yaml")

# Tables and reports
import pandas as pd
df = pd.DataFrame({"metric": [1, 2, 3], "value": [0.8, 0.9, 0.95]})
exp.log_artifact(df, artifact_path="results.csv")
```

### Rich Visualizations

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# All these are automatically captured and enhanced in ClearML:
writer.add_scalar("train/loss", loss, step)
writer.add_histogram("model/weights", weights, step)
writer.add_image("predictions", image_grid, step)
writer.add_text("notes", training_notes, step)

# ClearML adds automatic comparison and analysis tools
```

## Advanced Features

### Pipeline Integration

```python
# ClearML can orchestrate multi-step pipelines
from clearml import Task

# Parent pipeline task
pipeline_task = Task.init(
    project_name="ML Pipeline",
    task_name="Data Processing Pipeline"
)

# Child tasks for each step
with tracelet.logging("clearml", exp_name="data_preprocessing"):
    # ... preprocessing code ...
    pass

with tracelet.logging("clearml", exp_name="model_training"):
    # ... training code ...
    pass
```

### Hyperparameter Optimization

```python
# ClearML HyperParameter Optimization
from clearml.automation import HyperParameterOptimizer

# Define search space
optimizer = HyperParameterOptimizer(
    base_task_id="task_template_id",
    hyper_parameters={
        "learning_rate": [0.001, 0.01, 0.1],
        "batch_size": [16, 32, 64]
    }
)

# Launch optimization
optimizer.start()
```

### Model Registry

```python
# Register models automatically
exp = tracelet.get_active_experiment()

# Save model - automatically registered
torch.save(model.state_dict(), "best_model.pth")
exp.log_artifact("best_model.pth", artifact_path="models/")

# Add model metadata
exp.log_params({
    "model.accuracy": 0.95,
    "model.f1_score": 0.93,
    "model.version": "v1.2"
})
```

## Deployment Options

### SaaS Platform (Recommended)

```python
# No setup required - just configure credentials
tracelet.start_logging(backend="clearml")
```

### Self-Hosted Server

```yaml
# docker-compose.yml for ClearML Server
version: "3.6"
services:
  clearml-server:
    image: allegroai/clearml:latest
    ports:
      - "8080:8080" # Web UI
      - "8008:8008" # API server
      - "8081:8081" # File server
    volumes:
      - ./clearml-data:/opt/clearml/data
    environment:
      CLEARML_ELASTIC_SERVICE_HOST: elasticsearch
      CLEARML_ELASTIC_SERVICE_PORT: 9200
      CLEARML_REDIS_SERVICE_HOST: redis
      CLEARML_REDIS_SERVICE_PORT: 6379
      CLEARML_MONGODB_SERVICE_HOST: mongo
      CLEARML_MONGODB_SERVICE_PORT: 27017

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.6.2
    environment:
      - discovery.type=single-node

  redis:
    image: redis:5.0-alpine

  mongo:
    image: mongo:3.6.5
```

## Best Practices

### Project Organization

```python
# Use hierarchical project names
tracelet.start_logging(
    backend="clearml",
    project="Computer Vision/Image Classification",  # Nested projects
    exp_name="resnet50_cifar10_baseline"
)
```

### Task Naming

```python
# Descriptive task names with versioning
tracelet.start_logging(
    backend="clearml",
    project="ML Research",
    exp_name="bert_fine_tuning_v2.1",
    config={
        "tags": ["bert", "nlp", "fine-tuning", "v2.1"]
    }
)
```

### Artifact Management

```python
# Organize artifacts by purpose
exp = tracelet.get_active_experiment()

# Models
exp.log_artifact("model.pth", artifact_path="models/final/")
exp.log_artifact("checkpoint.pth", artifact_path="models/checkpoints/")

# Data
exp.log_artifact("train_stats.json", artifact_path="data/statistics/")
exp.log_artifact("predictions.csv", artifact_path="results/predictions/")

# Configs
exp.log_artifact("config.yaml", artifact_path="configs/")
```

## Troubleshooting

### Common Issues

**Authentication failure:**

```bash
# Re-run setup
clearml-init

# Or check credentials manually
cat ~/.clearml/clearml.conf
```

**Task not appearing in UI:**

```python
# Ensure task is properly closed
tracelet.stop_logging()

# Check project name spelling
# ClearML is case-sensitive
```

**Large artifact upload fails:**

```python
# Configure timeout for large files
import os
os.environ["CLEARML_FILES_SERVER_TIMEOUT"] = "300"  # 5 minutes
```

### Performance Optimization

```python
# Optimize for high-frequency logging
import os
os.environ["CLEARML_OFFLINE_MODE"] = "1"  # For development
os.environ["CLEARML_LOG_LEVEL"] = "WARNING"  # Reduce verbosity
```

## Comparison with Other Backends

| Feature                | ClearML | MLflow | W&B    | AIM    |
| ---------------------- | ------- | ------ | ------ | ------ |
| Auto-logging           | ⭐⭐⭐  | ⭐⭐   | ⭐⭐⭐ | ⭐     |
| Pipeline orchestration | ⭐⭐⭐  | ⭐     | ⭐⭐   | ⭐     |
| Enterprise features    | ⭐⭐⭐  | ⭐⭐   | ⭐⭐⭐ | ⭐     |
| Visualization          | ⭐⭐⭐  | ⭐⭐   | ⭐⭐⭐ | ⭐⭐⭐ |
| Setup complexity       | ⭐⭐    | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

## Migration

### From TensorBoard

```python
# Before: Pure TensorBoard
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("./runs")

# After: TensorBoard + ClearML via Tracelet
import tracelet
tracelet.start_logging(backend="clearml")
writer = SummaryWriter()  # Same code, enhanced tracking!
```

### From MLflow

```python
# Change backend while keeping same code
# tracelet.start_logging(backend="mlflow")     # Old
tracelet.start_logging(backend="clearml")     # New
# All TensorBoard code unchanged, gains ClearML features
```

## Complete Example

```python
import tracelet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import argparse

# Command line arguments (automatically captured by ClearML)
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=10)
args = parser.parse_args()

# Start ClearML tracking
tracelet.start_logging(
    exp_name="clearml_pytorch_example",
    project="tutorials",
    backend="clearml",
    config={
        "tags": ["pytorch", "tutorial", "automated"],
        "task_type": "training"
    }
)

# Model setup (automatically detected and logged)
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

# Data
X = torch.randn(1000, 784)
y = torch.randint(0, 10, (1000,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=args.batch_size)

# Training with automatic ClearML logging
writer = SummaryWriter()

for epoch in range(args.epochs):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Metrics automatically enhanced in ClearML
        writer.add_scalar("batch/loss", loss.item(),
                         epoch * len(dataloader) + batch_idx)
        total_loss += loss.item()

    # Epoch metrics with automatic comparison
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar("epoch/loss", avg_loss, epoch)

    # Model analysis (automatic histograms in ClearML)
    for name, param in model.named_parameters():
        writer.add_histogram(f"weights/{name}", param, epoch)

# Manual parameter logging (supplements automatic detection)
exp = tracelet.get_active_experiment()
exp.log_params({
    "model.type": "simple_mlp",
    "model.architecture": "784->128->10",
    "training.final_loss": avg_loss,
    "training.total_batches": len(dataloader) * args.epochs
})

# Save model (automatically tracked in model registry)
torch.save(model.state_dict(), "model.pth")
exp.log_artifact("model.pth", artifact_path="models/")

# Save training metadata
import json
metadata = {
    "training_completed": True,
    "final_epoch": args.epochs,
    "total_parameters": sum(p.numel() for p in model.parameters())
}
with open("training_metadata.json", "w") as f:
    json.dump(metadata, f)
exp.log_artifact("training_metadata.json", artifact_path="metadata/")

# Cleanup
writer.close()
tracelet.stop_logging()

print("✅ Training completed! View results at: https://app.clear.ml")
```

## Next Steps

- [Explore ClearML's pipeline features](https://clear.ml/docs/latest/docs/pipelines/pipelines_overview)
- [Set up hyperparameter optimization](https://clear.ml/docs/latest/docs/hyperdatasets/hyperdatasets)
- [Compare with other backends](../examples/multi-backend.md)
