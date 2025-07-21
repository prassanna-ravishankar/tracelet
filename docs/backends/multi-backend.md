# Multi-Backend Usage

Tracelet supports logging to multiple experiment tracking backends simultaneously, allowing you to leverage the strengths of different platforms or maintain backups across systems.

## Overview

Multi-backend logging enables you to:

- **Compare platforms** side-by-side with identical experiments
- **Maintain backups** across different tracking systems
- **Support team preferences** where different members prefer different tools
- **Gradual migration** from one platform to another
- **Leverage unique features** of each platform simultaneously

## Basic Multi-Backend Setup

### Simultaneous Logging

```python
import tracelet
from torch.utils.tensorboard import SummaryWriter

# Log to multiple backends at once
tracelet.start_logging(
    exp_name="multi_backend_experiment",
    project="platform_comparison",
    backend=["mlflow", "wandb"]  # List of backends
)

# All metrics automatically go to both platforms
writer = SummaryWriter()
writer.add_scalar("loss", 0.5, 1)
writer.add_scalar("accuracy", 0.9, 1)

tracelet.stop_logging()
```

### Three-Way Comparison

```python
# Log to three backends simultaneously
tracelet.start_logging(
    exp_name="comprehensive_comparison",
    project="backend_evaluation",
    backend=["mlflow", "clearml", "wandb"]
)

# Single codebase, three tracking platforms
writer = SummaryWriter()
for epoch in range(10):
    loss = train_one_epoch()
    writer.add_scalar("train/loss", loss, epoch)
    # → Goes to MLflow, ClearML, and W&B simultaneously
```

## Backend-Specific Configuration

### Individual Backend Settings

```python
# Configure each backend independently
tracelet.start_logging(
    exp_name="configured_multi_backend",
    project="custom_setup",
    backend=["mlflow", "wandb", "clearml"],
    config={
        "mlflow": {
            "backend_url": "http://mlflow-server:5000",
            "experiment_name": "Production Experiments"
        },
        "wandb": {
            "entity": "research_team",
            "tags": ["production", "comparison"],
            "group": "multi_backend_runs"
        },
        "clearml": {
            "task_name": "Multi-Backend Training",
            "tags": ["mlflow", "wandb", "comparison"],
            "output_uri": "s3://artifacts-bucket/"
        }
    }
)
```

### Environment-Based Configuration

```python
import os

# Set different configurations via environment
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
os.environ["WANDB_PROJECT"] = "multi_backend_project"
os.environ["CLEARML_PROJECT_NAME"] = "Multi Platform Testing"

# Backends will use their respective environment configs
tracelet.start_logging(
    backend=["mlflow", "wandb", "clearml"],
    exp_name="env_configured_experiment"
)
```

## Platform Comparison Example

### Complete Multi-Backend Training

```python
import tracelet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import time

def run_multi_backend_experiment():
    """Run the same experiment across multiple backends for comparison."""

    # Start logging to all available backends
    exp = tracelet.start_logging(
        exp_name="multi_backend_comparison",
        project="platform_evaluation",
        backend=["mlflow", "wandb", "clearml", "aim"],
        config={
            "wandb": {
                "tags": ["comparison", "multi-backend"],
                "notes": "Comparing tracking platforms"
            },
            "clearml": {
                "tags": ["comparison", "evaluation"],
                "task_type": "training"
            }
        }
    )

    # Model and data setup
    model = nn.Sequential(
        nn.Linear(100, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Synthetic regression data
    X = torch.randn(1000, 100)
    y = torch.randn(1000, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32)

    # Log hyperparameters to all backends
    exp.log_params({
        "model.architecture": "3-layer MLP",
        "model.input_size": 100,
        "model.hidden_sizes": [64, 32],
        "model.dropout": 0.1,
        "optimizer.type": "adam",
        "optimizer.lr": 0.001,
        "data.batch_size": 32,
        "data.total_samples": 1000
    })

    # Training loop with comprehensive logging
    writer = SummaryWriter()
    start_time = time.time()

    for epoch in range(25):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Batch-level metrics
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("batch/loss", loss.item(), global_step)
            epoch_loss += loss.item()

        # Epoch-level metrics
        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar("epoch/loss", avg_loss, epoch)
        writer.add_scalar("epoch/learning_rate", optimizer.param_groups[0]['lr'], epoch)

        # Model analysis every 5 epochs
        if epoch % 5 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f"weights/{name}", param, epoch)
                if param.grad is not None:
                    writer.add_histogram(f"gradients/{name}", param.grad, epoch)

        print(f"Epoch {epoch:2d}/24: Loss = {avg_loss:.4f}")

    training_time = time.time() - start_time

    # Log final metrics
    exp.log_params({
        "training.final_loss": avg_loss,
        "training.total_time": training_time,
        "training.epochs": 25,
        "training.batches_per_epoch": len(dataloader)
    })

    # Save artifacts
    torch.save(model.state_dict(), "multi_backend_model.pth")
    exp.log_artifact("multi_backend_model.pth", artifact_path="models/")

    # Create training summary
    summary = {
        "experiment": "multi_backend_comparison",
        "backends": ["mlflow", "wandb", "clearml", "aim"],
        "final_loss": avg_loss,
        "training_time_seconds": training_time,
        "model_parameters": sum(p.numel() for p in model.parameters())
    }

    import json
    with open("experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    exp.log_artifact("experiment_summary.json")

    writer.close()
    tracelet.stop_logging()

    return {
        "final_loss": avg_loss,
        "training_time": training_time,
        "backends_used": ["mlflow", "wandb", "clearml", "aim"]
    }

# Run the experiment
if __name__ == "__main__":
    results = run_multi_backend_experiment()
    print(f"✅ Multi-backend experiment completed!")
    print(f"📊 Final loss: {results['final_loss']:.4f}")
    print(f"⏱️  Training time: {results['training_time']:.2f}s")
    print(f"🔄 Backends: {', '.join(results['backends_used'])}")
```

## Backend-Specific Features

### Leveraging Unique Capabilities

```python
# Start multi-backend logging
tracelet.start_logging(
    backend=["mlflow", "wandb", "clearml"],
    exp_name="feature_showcase"
)

# Use each platform's strengths:

# 1. MLflow: Model registry and serving
import mlflow.pytorch
mlflow.pytorch.log_model(model, "model", registered_model_name="ProductionModel")

# 2. W&B: Interactive plots and sweeps
import wandb
wandb.log({"custom_plot": wandb.plot.line_series(
    xs=epochs,
    ys=[train_losses, val_losses],
    keys=["train", "val"],
    title="Loss Comparison"
)})

# 3. ClearML: Automatic framework detection
# (Automatically captures more context and metadata)

# All platforms still get the same core metrics via Tracelet
writer.add_scalar("shared_metric", value, step)
```

## Performance Considerations

### Overhead Analysis

```python
import time

def measure_logging_overhead():
    """Compare single vs multi-backend performance."""

    metrics = []

    # Single backend
    start = time.time()
    tracelet.start_logging(backend="mlflow", exp_name="single_backend_test")

    writer = SummaryWriter()
    for i in range(1000):
        writer.add_scalar("test_metric", i * 0.001, i)

    tracelet.stop_logging()
    single_time = time.time() - start

    # Multi-backend
    start = time.time()
    tracelet.start_logging(
        backend=["mlflow", "wandb", "clearml"],
        exp_name="multi_backend_test"
    )

    writer = SummaryWriter()
    for i in range(1000):
        writer.add_scalar("test_metric", i * 0.001, i)

    tracelet.stop_logging()
    multi_time = time.time() - start

    print(f"Single backend: {single_time:.2f}s")
    print(f"Multi backend:  {multi_time:.2f}s")
    print(f"Overhead:       {multi_time - single_time:.2f}s ({((multi_time/single_time - 1) * 100):.1f}%)")
```

### Optimization Strategies

```python
# Optimize multi-backend performance
import os

# Reduce verbosity
os.environ["WANDB_SILENT"] = "true"
os.environ["CLEARML_LOG_LEVEL"] = "WARNING"

# Use offline modes for development
os.environ["WANDB_MODE"] = "offline"
os.environ["CLEARML_OFFLINE_MODE"] = "1"

# Batch metrics for efficiency
os.environ["WANDB_LOG_INTERVAL_SECONDS"] = "10"
```

## Selective Backend Usage

### Conditional Backend Selection

```python
import os

# Choose backends based on environment
def get_backends_for_env():
    env = os.environ.get("ENVIRONMENT", "development")

    if env == "development":
        return ["mlflow"]  # Local only
    elif env == "staging":
        return ["mlflow", "wandb"]  # Add visualization
    elif env == "production":
        return ["mlflow", "wandb", "clearml"]  # Full tracking
    else:
        return ["mlflow"]  # Default fallback

backends = get_backends_for_env()
tracelet.start_logging(backend=backends, exp_name="env_aware_experiment")
```

### Feature-Based Selection

```python
def select_backends_by_features(need_visualization=True, need_collaboration=True, need_artifacts=True):
    """Select backends based on required features."""
    backends = ["mlflow"]  # Always include MLflow as base

    if need_visualization:
        backends.append("wandb")  # Best visualizations

    if need_collaboration:
        backends.append("clearml")  # Team features

    if need_artifacts and "wandb" not in backends:
        backends.append("wandb")  # Good artifact management

    return backends

# Use based on experiment needs
experiment_backends = select_backends_by_features(
    need_visualization=True,
    need_collaboration=False,
    need_artifacts=True
)

tracelet.start_logging(backend=experiment_backends, exp_name="feature_selected_experiment")
```

## Migration Workflows

### Gradual Platform Migration

```python
# Phase 1: Run both old and new platforms
tracelet.start_logging(
    backend=["mlflow", "wandb"],  # Old + New
    exp_name="migration_phase_1"
)

# Phase 2: Compare results and validate new platform
# (Run experiments on both, verify data consistency)

# Phase 3: Switch to new platform only
tracelet.start_logging(
    backend=["wandb"],  # New only
    exp_name="migration_complete"
)
```

### A/B Testing Platforms

```python
import random

# Randomly assign experiments to different backends for comparison
backend_choice = random.choice([
    ["mlflow"],
    ["wandb"],
    ["clearml"],
    ["mlflow", "wandb"]  # Multi-backend group
])

tracelet.start_logging(
    backend=backend_choice,
    exp_name=f"platform_ab_test_{hash(str(backend_choice)) % 1000}",
    config={"tags": [f"backend_test_{len(backend_choice)}_platforms"]}
)
```

## Best Practices

### Naming Conventions

```python
# Use consistent naming across all backends
tracelet.start_logging(
    backend=["mlflow", "wandb", "clearml"],
    exp_name="resnet50_cifar10_v2.1",  # Version in name
    project="computer_vision_research", # Consistent project
    config={
        "shared_tags": ["resnet", "cifar10", "v2.1", "multi_backend"],
        "mlflow": {"experiment_name": "CV Research"},
        "wandb": {"group": "resnet_experiments"},
        "clearml": {"task_type": "training"}
    }
)
```

### Configuration Management

```python
# Centralized configuration for multi-backend setups
BACKEND_CONFIGS = {
    "development": {
        "backends": ["mlflow"],
        "config": {
            "mlflow": {"backend_url": "sqlite:///dev.db"}
        }
    },
    "staging": {
        "backends": ["mlflow", "wandb"],
        "config": {
            "mlflow": {"backend_url": "http://staging-mlflow:5000"},
            "wandb": {"mode": "offline"}
        }
    },
    "production": {
        "backends": ["mlflow", "wandb", "clearml"],
        "config": {
            "mlflow": {"backend_url": "http://prod-mlflow:5000"},
            "wandb": {"entity": "production_team"},
            "clearml": {"output_uri": "s3://prod-artifacts/"}
        }
    }
}

# Use configuration
env = os.environ.get("ENVIRONMENT", "development")
config = BACKEND_CONFIGS[env]

tracelet.start_logging(
    backend=config["backends"],
    exp_name="configured_experiment",
    config=config["config"]
)
```

## Troubleshooting Multi-Backend Issues

### Common Problems

```python
# Handle backend-specific failures gracefully
try:
    tracelet.start_logging(
        backend=["mlflow", "wandb", "clearml", "aim"],
        exp_name="robust_experiment"
    )
except Exception as e:
    print(f"Some backends failed to initialize: {e}")
    # Fallback to working backends
    tracelet.start_logging(
        backend=["mlflow"],  # Reliable fallback
        exp_name="fallback_experiment"
    )
```

### Debugging Backend Issues

```python
# Enable detailed logging for debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check which backends are active
exp = tracelet.get_active_experiment()
if hasattr(exp, 'get_active_backends'):
    active_backends = exp.get_active_backends()
    print(f"Active backends: {active_backends}")
```

## Results Comparison

After running multi-backend experiments, compare results across platforms:

1. **MLflow**: Check `http://localhost:5000` for model registry and basic metrics
2. **W&B**: Visit [wandb.ai](https://wandb.ai) for interactive visualizations and collaboration
3. **ClearML**: Check [app.clear.ml](https://app.clear.ml) for comprehensive experiment analysis
4. **AIM**: Run `aim up` and visit `http://localhost:43800` for fast querying

Each platform will show the same core metrics but with their unique visualization and analysis capabilities!

## Next Steps

- [Try the complete multi-backend example](../examples/multi-backend.md)
- [Learn about platform-specific features](index.md)
- [Set up production multi-backend workflows](../guides/best-practices.md)
