# Weights & Biases Backend

Weights & Biases (wandb) is a collaborative platform for experiment tracking, visualization, and model management, particularly popular in deep learning research.

## Overview

W&B provides best-in-class features for ML teams:

- Beautiful, interactive visualizations
- Collaborative experiment sharing
- Automatic hyperparameter tracking
- Model registry and artifact versioning
- Real-time collaboration and reports
- Integration with popular ML frameworks

Perfect for deep learning research and team collaboration.

## Installation

=== "pip"
`bash
    pip install tracelet[wandb]
    # Note: wandb is already included as core dependency
    `

=== "uv"
`bash
    uv add tracelet[wandb]
    # Note: wandb is already included as core dependency
    `

**Minimum Requirements:**

- wandb >= 0.16.0 (included in tracelet core dependencies)
- Python >= 3.9

## Quick Start

### Online Mode (Default)

```python
import tracelet
from torch.utils.tensorboard import SummaryWriter

# Start with W&B online tracking
tracelet.start_logging(
    exp_name="wandb_experiment",
    project="my_project",
    backend="wandb"
)

# Use TensorBoard as normal - metrics go to W&B
writer = SummaryWriter()
writer.add_scalar("loss", 0.5, 1)
writer.add_scalar("accuracy", 0.9, 1)

tracelet.stop_logging()
```

### View Results

Visit [wandb.ai](https://wandb.ai) to view your experiments in the web dashboard.

## Setup and Authentication

### Initial Setup

1. **Create account** at [wandb.ai](https://wandb.ai)
2. **Get API key** from Settings → API keys
3. **Login via CLI**:

```bash
# Interactive login (recommended)
wandb login

# Or set API key directly
export WANDB_API_KEY=your_api_key_here
```

### Environment Variables

```bash
# Authentication
export WANDB_API_KEY=your_api_key_here

# Project defaults
export WANDB_PROJECT=my_default_project
export WANDB_ENTITY=your_username_or_team

# Mode configuration
export WANDB_MODE=online    # online, offline, disabled
export WANDB_DIR=/path/to/wandb/logs
```

### Offline Mode

```python
import os

# Enable offline mode (automatic fallback in examples)
os.environ["WANDB_MODE"] = "offline"

tracelet.start_logging(
    backend="wandb",
    exp_name="offline_experiment",
    project="development"
)

# Sync later when online
# wandb sync wandb/offline-run-xxx
```

## Configuration

### Basic Configuration

```python
tracelet.start_logging(
    backend="wandb",
    exp_name="my_experiment",
    project="computer_vision",    # W&B project name
    config={
        "entity": "your_username",        # W&B entity (user/team)
        "name": "resnet_baseline",        # Run name
        "tags": ["pytorch", "baseline"],  # Run tags
        "notes": "Initial baseline run"   # Run description
    }
)
```

### Advanced Configuration

```python
# Comprehensive setup
tracelet.start_logging(
    backend="wandb",
    exp_name="advanced_experiment",
    project="ml_research",
    config={
        "entity": "research_team",
        "name": "hyperparameter_sweep_v2",
        "tags": ["experiment", "sweep", "optimized"],
        "notes": "Systematic hyperparameter optimization",
        "group": "resnet_experiments",    # Group related runs
        "job_type": "training",           # Job type for organization
        "mode": "online",                 # online, offline, disabled
        "save_code": True,                # Save code snapshot
        "resume": "auto"                  # Resume previous run if exists
    }
)
```

### Run Groups and Jobs

```python
# Organize experiments with groups and job types
for lr in [0.001, 0.01, 0.1]:
    tracelet.start_logging(
        backend="wandb",
        project="hyperparameter_sweep",
        exp_name=f"lr_{lr}",
        config={
            "group": "learning_rate_sweep",    # Groups related runs
            "job_type": "hp_search",           # Job type
            "tags": [f"lr_{lr}"]
        }
    )
    # ... training code ...
    tracelet.stop_logging()
```

## Features

### Automatic Logging

W&B captures extensive information automatically:

```python
# These metrics are automatically enhanced in W&B:
writer = SummaryWriter()
writer.add_scalar("train/loss", loss, step)
writer.add_histogram("model/gradients", gradients, step)
writer.add_image("predictions", image_grid, step)

# W&B adds:
# - Interactive plots and zooming
# - Automatic metric correlation analysis
# - Real-time streaming
# - Mobile notifications
```

### Hyperparameter Tracking

```python
exp = tracelet.get_active_experiment()

# Structured hyperparameter logging
exp.log_params({
    "model": {
        "architecture": "resnet50",
        "layers": 50,
        "dropout": 0.1
    },
    "training": {
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "adam"
    },
    "data": {
        "dataset": "cifar10",
        "augmentation": True
    }
})
```

### Rich Artifacts

```python
# Log models and datasets
exp.log_artifact("model.pth", artifact_path="models/")
exp.log_artifact("dataset.tar.gz", artifact_path="data/")

# Log tables for structured data
import pandas as pd
predictions_df = pd.DataFrame({
    "image_id": range(100),
    "predicted_class": predictions,
    "confidence": confidences
})
exp.log_artifact(predictions_df, artifact_path="predictions.csv")
```

### Interactive Visualizations

```python
# Custom plots and charts
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(epochs, losses)
ax.set_title("Training Loss")

# Log as W&B artifact with interactive viewing
exp.log_artifact(fig, artifact_path="plots/training_loss.png")
```

## Advanced Features

### Hyperparameter Sweeps

```yaml
# sweep_config.yaml
program: train.py
method: random
metric:
  goal: maximize
  name: val_accuracy
parameters:
  learning_rate:
    values: [0.001, 0.01, 0.1]
  batch_size:
    values: [16, 32, 64]
  dropout:
    min: 0.1
    max: 0.5
```

```python
# Run sweep
import wandb

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="hyperparameter_optimization")

# Run sweep agents
wandb.agent(sweep_id, function=train_function, count=20)
```

### Model Registry

```python
# Log model to registry with versioning
import wandb

# Create artifact
model_artifact = wandb.Artifact(
    name="resnet_model",
    type="model",
    description="ResNet50 trained on CIFAR-10"
)

# Add model files
model_artifact.add_file("model.pth")
model_artifact.add_file("model_config.json")

# Log to registry
exp.log_artifact(model_artifact)

# Use model in another run
artifact = wandb.use_artifact("resnet_model:latest")
artifact.download()
```

### Reports and Sharing

```python
# Create shareable reports
import wandb

# Generate report from runs
report = wandb.Report(
    project="my_project",
    title="Model Comparison Report",
    description="Comparing different architectures"
)

# Add plots and analyses
report.blocks = [
    wandb.report.PanelGrid(
        panels=[
            wandb.report.ScalarChart(
                title="Training Loss Comparison",
                config={"metrics": ["train/loss"]}
            )
        ]
    )
]

# Save and share
report.save()
```

## Integration Features

### Framework Auto-logging

```python
# Automatic integration with popular frameworks
import pytorch_lightning as pl

class LightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # W&B automatically logs:
        # - Hyperparameters
        # - Metrics
        # - Model topology
        # - System metrics

# Just start tracelet - everything else is automatic
tracelet.start_logging(backend="wandb")
trainer = pl.Trainer()
trainer.fit(model)
```

### Jupyter Integration

```python
# Enhanced Jupyter notebook support
import wandb

# Initialize in notebook
wandb.init(project="notebook_experiments")

# Automatic cell tracking
%wandb notebook  # Magic command for cell tracking
```

## Best Practices

### Project Organization

```python
# Use consistent naming conventions
tracelet.start_logging(
    backend="wandb",
    project="computer-vision-research",     # Descriptive project name
    exp_name="resnet50_cifar10_baseline",   # Structured run name
    config={
        "entity": "ml_team",
        "tags": ["resnet", "cifar10", "baseline", "v1.0"]
    }
)
```

### Experiment Tracking

```python
# Log comprehensive context
exp = tracelet.get_active_experiment()

# Code version
exp.log_params({
    "git.commit": "abc123def",
    "git.branch": "feature/new_model",
    "code.version": "v1.2.0"
})

# Environment
exp.log_params({
    "env.python_version": "3.11.0",
    "env.cuda_version": "11.8",
    "env.gpu_type": "RTX 4090"
})

# Dataset info
exp.log_params({
    "data.name": "CIFAR-10",
    "data.size": 50000,
    "data.preprocessing": "normalize+augment"
})
```

### Collaboration

```python
# Share experiments with team
tracelet.start_logging(
    backend="wandb",
    project="team_experiments",
    config={
        "entity": "research_team",          # Team workspace
        "tags": ["shared", "review"],
        "notes": "Ready for team review"
    }
)
```

## Troubleshooting

### Common Issues

**Authentication failure:**

```bash
# Re-login
wandb login

# Or check API key
echo $WANDB_API_KEY
```

**Offline mode not working:**

```python
# Explicitly set offline mode
import os
os.environ["WANDB_MODE"] = "offline"

# Verify mode
import wandb
print(wandb.env.get_mode())  # Should print "offline"
```

**Large file upload timeouts:**

```python
# Increase timeout for large artifacts
import os
os.environ["WANDB_HTTP_TIMEOUT"] = "300"  # 5 minutes
```

### Performance Optimization

```python
# Optimize for high-frequency logging
import os
os.environ["WANDB_LOG_INTERVAL_SECONDS"] = "5"  # Batch logs every 5 seconds
os.environ["WANDB_DISABLE_GIT"] = "true"        # Disable git tracking if not needed
```

## Comparison with Other Backends

| Feature                | W&B    | MLflow | ClearML | AIM    |
| ---------------------- | ------ | ------ | ------- | ------ |
| Visualization quality  | ⭐⭐⭐ | ⭐⭐   | ⭐⭐⭐  | ⭐⭐⭐ |
| Collaboration features | ⭐⭐⭐ | ⭐⭐   | ⭐⭐⭐  | ⭐⭐   |
| Hyperparameter sweeps  | ⭐⭐⭐ | ⭐⭐   | ⭐⭐⭐  | ⭐⭐   |
| Framework integration  | ⭐⭐⭐ | ⭐⭐   | ⭐⭐⭐  | ⭐⭐   |
| Setup simplicity       | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐    | ⭐⭐⭐ |

## Migration

### From TensorBoard

```python
# Before: Pure TensorBoard
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("./runs")

# After: TensorBoard + W&B via Tracelet
import tracelet
tracelet.start_logging(backend="wandb")
writer = SummaryWriter()  # Same code, enhanced with W&B!
```

### From MLflow

```python
# Easy backend switch
# tracelet.start_logging(backend="mlflow")     # Old
tracelet.start_logging(backend="wandb")       # New
# All existing TensorBoard code works unchanged
```

## Complete Example

```python
import tracelet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Start W&B tracking with comprehensive config
tracelet.start_logging(
    exp_name="wandb_pytorch_example",
    project="tutorials",
    backend="wandb",
    config={
        "entity": "your_username",
        "tags": ["pytorch", "tutorial", "example"],
        "notes": "Complete example showing W&B integration",
        "group": "tutorial_runs",
        "job_type": "training"
    }
)

# Model setup
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Log model architecture
exp = tracelet.get_active_experiment()
exp.log_params({
    "model.architecture": "mlp",
    "model.layers": [784, 128, 64, 10],
    "model.dropout": 0.2,
    "model.total_params": sum(p.numel() for p in model.parameters()),
    "optimizer.type": "adam",
    "optimizer.lr": 0.001
})

# Data
X = torch.randn(1000, 784)
y = torch.randint(0, 10, (1000,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32)

# Training with automatic W&B logging
writer = SummaryWriter()
losses = []

for epoch in range(20):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Metrics automatically enhanced in W&B
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar("batch/loss", loss.item(), global_step)
        total_loss += loss.item()

    # Epoch metrics
    avg_loss = total_loss / len(dataloader)
    losses.append(avg_loss)
    writer.add_scalar("epoch/loss", avg_loss, epoch)
    writer.add_scalar("epoch/learning_rate", optimizer.param_groups[0]['lr'], epoch)

    # Model analysis
    for name, param in model.named_parameters():
        writer.add_histogram(f"weights/{name}", param, epoch)
        if param.grad is not None:
            writer.add_histogram(f"gradients/{name}", param.grad, epoch)

# Create and log training plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(losses)
ax.set_title("Training Loss Over Time")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.grid(True)

# Save and log plot
plt.savefig("training_loss.png")
exp.log_artifact("training_loss.png", artifact_path="plots/")
plt.close()

# Log final metrics
exp.log_params({
    "training.final_loss": losses[-1],
    "training.best_loss": min(losses),
    "training.epochs_completed": len(losses),
    "training.total_batches": len(dataloader) * len(losses)
})

# Save and log model
torch.save(model.state_dict(), "final_model.pth")
exp.log_artifact("final_model.pth", artifact_path="models/")

# Create model summary
model_info = {
    "architecture": "3-layer MLP",
    "input_size": 784,
    "hidden_sizes": [128, 64],
    "output_size": 10,
    "total_parameters": sum(p.numel() for p in model.parameters()),
    "final_loss": losses[-1]
}

import json
with open("model_summary.json", "w") as f:
    json.dump(model_info, f, indent=2)
exp.log_artifact("model_summary.json", artifact_path="metadata/")

# Cleanup
writer.close()
tracelet.stop_logging()

print("✅ Training completed! View results at: https://wandb.ai")
print("📊 Check your dashboard for interactive visualizations and model analysis")
```

## Next Steps

- [Set up hyperparameter sweeps](https://docs.wandb.ai/guides/sweeps)
- [Explore W&B Reports for sharing results](https://docs.wandb.ai/guides/reports)
- [Compare with other backends](../examples/multi-backend.md)
