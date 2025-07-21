# Multi-Backend Comparison

This page demonstrates how to use Tracelet with multiple backends simultaneously and provides comparison examples to help you choose the right backend for your needs.

## Using Multiple Backends Simultaneously

Tracelet can log to multiple experiment tracking platforms at once, allowing you to compare their features or migrate between systems.

### Basic Multi-Backend Setup

```python
import tracelet
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Start experiment tracking with multiple backends
tracelet.start_logging(
    exp_name="multi_backend_demo",
    project="backend_comparison",
    backend=["mlflow", "clearml", "wandb"]  # List multiple backends
)

# Create a simple training setup
model = nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Use TensorBoard - metrics automatically sent to ALL backends
writer = SummaryWriter()

for epoch in range(30):
    # Generate synthetic data
    X = torch.randn(64, 10)
    y = torch.randn(64, 1)

    # Training step
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    # Log metrics - sent to MLflow, ClearML, AND W&B
    writer.add_scalar('Loss/Train', loss.item(), epoch)
    writer.add_scalar('Metrics/LearningRate', optimizer.param_groups[0]['lr'], epoch)

# Log experiment parameters to all backends
exp = tracelet.get_active_experiment()
exp.log_params({
    "learning_rate": 0.01,
    "batch_size": 64,
    "epochs": 30,
    "optimizer": "adam"
})

writer.close()
tracelet.stop_logging()

print("✅ Experiment logged to MLflow, ClearML, and W&B simultaneously!")
```

### Backend-Specific Configuration

```python
import tracelet
from tracelet.settings import TraceletSettings

# Configure settings for multiple backends
settings = TraceletSettings(
    project="backend_specific_demo",
    backend=["mlflow", "wandb"],
    track_system=True,

    # Backend-specific settings
    mlflow_tracking_uri="http://localhost:5000",
    wandb_entity="your_team_name",
    wandb_project="comparison_study"
)

tracelet.start_logging(
    exp_name="configured_multi_backend",
    settings=settings
)

# Your training code here...
exp = tracelet.get_active_experiment()
exp.log_metric("test_metric", 0.95)

tracelet.stop_logging()
```

## Backend Feature Comparison

### Quick Comparison Table

| Feature                         | MLflow | ClearML | W&B | AIM |
| ------------------------------- | ------ | ------- | --- | --- |
| **Local Deployment**            | ✅     | ✅      | ❌  | ✅  |
| **Cloud Hosting**               | ✅     | ✅      | ✅  | ❌  |
| **Real-time Metrics**           | ✅     | ✅      | ✅  | ✅  |
| **Model Registry**              | ✅     | ✅      | ✅  | ❌  |
| **Artifact Storage**            | ✅     | ✅      | ✅  | ✅  |
| **Hyperparameter Optimization** | ❌     | ✅      | ✅  | ❌  |
| **Dataset Versioning**          | ❌     | ✅      | ✅  | ❌  |
| **Collaboration Tools**         | ❌     | ✅      | ✅  | ❌  |
| **Open Source**                 | ✅     | ✅      | ❌  | ✅  |

### MLflow Example

```python
import tracelet

# MLflow - Great for model registry and local deployment
tracelet.start_logging(
    exp_name="mlflow_demo",
    project="backend_comparison",
    backend="mlflow"
)

exp = tracelet.get_active_experiment()

# Log comprehensive experiment data
exp.log_params({
    "model_type": "ResNet50",
    "dataset": "CIFAR-10",
    "augmentation": True
})

for epoch in range(5):
    exp.log_metric("train_loss", 0.5 - epoch * 0.1, iteration=epoch)
    exp.log_metric("val_accuracy", 0.8 + epoch * 0.03, iteration=epoch)

# MLflow excels at artifact logging
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4, 5], [0.5, 0.4, 0.3, 0.2, 0.1])
plt.title("Training Loss")
plt.savefig("loss_curve.png")
exp.log_artifact("loss_curve.png", "plots/")

tracelet.stop_logging()
print("✅ Check MLflow UI at http://localhost:5000")
```

### ClearML Example

```python
import tracelet

# ClearML - Enterprise features and data management
tracelet.start_logging(
    exp_name="clearml_demo",
    project="backend_comparison",
    backend="clearml"
)

exp = tracelet.get_active_experiment()

# ClearML automatically captures environment info
exp.log_params({
    "framework": "PyTorch",
    "clearml_auto_capture": True
})

# Simulate hyperparameter optimization
for lr in [0.001, 0.01, 0.1]:
    for batch_size in [32, 64, 128]:
        exp.log_metric(f"accuracy_lr{lr}_bs{batch_size}",
                      0.8 + lr * 0.5 + batch_size * 0.001)

tracelet.stop_logging()
print("✅ Check ClearML at https://app.clear.ml")
```

### Weights & Biases Example

```python
import tracelet

# W&B - Great for collaboration and sweeps
tracelet.start_logging(
    exp_name="wandb_demo",
    project="backend_comparison",
    backend="wandb"
)

exp = tracelet.get_active_experiment()

# W&B excels at rich media logging
exp.log_params({
    "architecture": "transformer",
    "attention_heads": 8,
    "hidden_size": 512
})

# Log metrics with custom charts
for step in range(20):
    exp.log_metric("training/loss", 2.0 * (0.9 ** step), iteration=step)
    exp.log_metric("training/perplexity", 50 * (0.95 ** step), iteration=step)
    exp.log_metric("validation/bleu_score", min(0.9, step * 0.05), iteration=step)

tracelet.stop_logging()
print("✅ Check W&B at https://wandb.ai")
```

### AIM Example

```python
import tracelet

# AIM - Lightweight and self-hosted
tracelet.start_logging(
    exp_name="aim_demo",
    project="backend_comparison",
    backend="aim"
)

exp = tracelet.get_active_experiment()

# AIM is great for fast iteration and visualization
exp.log_params({
    "model": "lightweight_cnn",
    "optimization": "fast_iteration"
})

# Log multiple runs for comparison
for run_id in range(3):
    for epoch in range(10):
        # Simulate different random seeds
        noise = run_id * 0.1
        exp.log_metric(f"run_{run_id}/accuracy",
                      0.7 + epoch * 0.02 + noise, iteration=epoch)

tracelet.stop_logging()
print("✅ Check AIM UI at http://localhost:43800")
```

## Migration Between Backends

### Step-by-Step Migration

```python
import tracelet

# Step 1: Export from current backend (example with MLflow)
def export_mlflow_experiment(experiment_id):
    """Export experiment data for migration"""
    import mlflow

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment(experiment_id)
    runs = client.search_runs([experiment_id])

    export_data = []
    for run in runs:
        run_data = {
            "name": run.info.run_name,
            "params": run.data.params,
            "metrics": run.data.metrics,
            "tags": run.data.tags
        }
        export_data.append(run_data)

    return export_data

# Step 2: Import to new backend
def migrate_to_new_backend(export_data, new_backend="wandb"):
    """Migrate experiment data to new backend"""

    for i, run_data in enumerate(export_data):
        tracelet.start_logging(
            exp_name=f"migrated_{run_data['name']}",
            project="migration_project",
            backend=new_backend
        )

        exp = tracelet.get_active_experiment()

        # Migrate parameters
        exp.log_params(run_data["params"])

        # Migrate metrics (simplified - you may need to handle iterations)
        for metric_name, metric_value in run_data["metrics"].items():
            exp.log_metric(metric_name, metric_value)

        tracelet.stop_logging()
        print(f"✅ Migrated run {i+1}/{len(export_data)}")

# Example usage (uncomment to use):
# exported = export_mlflow_experiment("your_experiment_id")
# migrate_to_new_backend(exported, "wandb")
```

### Parallel Logging for Safe Migration

```python
import tracelet

# Log to both old and new backends during transition
tracelet.start_logging(
    exp_name="safe_migration",
    project="migration_project",
    backend=["mlflow", "wandb"]  # Old and new backend
)

# Your existing training code works unchanged
exp = tracelet.get_active_experiment()

for epoch in range(10):
    loss = 1.0 / (epoch + 1)
    accuracy = min(0.95, epoch * 0.1)

    # Metrics go to BOTH backends automatically
    exp.log_metric("loss", loss, iteration=epoch)
    exp.log_metric("accuracy", accuracy, iteration=epoch)

exp.log_params({"migration_phase": "parallel_logging"})

tracelet.stop_logging()
print("✅ Data safely logged to both MLflow and W&B")
```

## Choosing the Right Backend

### Use MLflow When:

- You need local deployment and control
- Model registry is important
- You're building ML platforms or tools
- Cost is a primary concern (open source)

### Use ClearML When:

- You need enterprise features
- Data versioning is critical
- You want comprehensive experiment tracking
- Team collaboration is important

### Use Weights & Biases When:

- You're doing research or need rich visualizations
- Hyperparameter optimization is key
- You want the best user experience
- Collaboration and sharing are priorities

### Use AIM When:

- You need lightweight, fast iteration
- You want self-hosted with minimal setup
- Visualization performance is critical
- You're doing metric-heavy experiments

## Best Practices for Multi-Backend Usage

### 1. Start Simple

```python
# Begin with one backend
tracelet.start_logging(backend="mlflow")

# Add more as needed
tracelet.start_logging(backend=["mlflow", "wandb"])
```

### 2. Use Environment Variables for Flexibility

```bash
# .env file
TRACELET_BACKEND=mlflow,wandb
TRACELET_PROJECT=my_project
```

```python
import os
import tracelet

# Automatically uses backends from environment
backends = os.getenv("TRACELET_BACKEND", "mlflow").split(",")
tracelet.start_logging(
    exp_name="flexible_backend",
    backend=backends
)
```

### 3. Handle Backend-Specific Features

```python
import tracelet

exp = tracelet.get_active_experiment()

# Check which backends are active
if "wandb" in exp.active_backends:
    # W&B-specific logging
    exp.log_metric("wandb/special_metric", 0.95)

if "clearml" in exp.active_backends:
    # ClearML-specific features
    exp.log_params({"clearml_task_id": "auto_captured"})
```

## Next Steps

- Explore [Backend-Specific Guides](../backends/index.md) for detailed configuration
- Try the [Interactive Notebooks](notebooks.md) for hands-on multi-backend experience
- Check [Best Practices](../guides/best-practices.md) for production multi-backend usage
- Review [Migration Guide](../guides/migration.md) for detailed migration strategies
