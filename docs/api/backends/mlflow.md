# MLflow Backend

::: tracelet.backends.mlflow_backend.MLflowBackend
options:
show_source: true
show_bases: true
merge_init_into_class: true
heading_level: 2

## Overview

The MLflow backend provides integration with MLflow tracking server for experiment logging and management.

## Basic Usage

```python
import tracelet

# Basic MLflow usage
tracelet.start_logging(
    exp_name="mlflow_experiment",
    project="my_project",
    backend="mlflow"
)

# Custom MLflow configuration
tracelet.start_logging(
    exp_name="custom_mlflow",
    project="my_project",
    backend="mlflow",
    config={
        "tracking_uri": "http://mlflow-server:5000",
        "experiment_name": "Deep Learning Experiments"
    }
)
```

## Configuration Options

### Via Settings

```python
from tracelet.settings import TraceletSettings

# Configure via settings
settings = TraceletSettings(
    project="mlflow_project",
    backend=["mlflow"],
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="ML Experiments"
)

tracelet.start_logging(exp_name="configured_exp", settings=settings)
```

### Via Environment Variables

```bash
# MLflow-specific environment variables
export MLFLOW_TRACKING_URI="http://mlflow-server:5000"
export MLFLOW_EXPERIMENT_NAME="Production Experiments"

# Tracelet MLflow settings
export TRACELET_MLFLOW_TRACKING_URI="http://localhost:5000"
export TRACELET_MLFLOW_EXPERIMENT_NAME="Development Experiments"
```

### Via Configuration Object

```python
# Direct configuration
mlflow_config = {
    "tracking_uri": "http://localhost:5000",
    "experiment_name": "My Experiments",
    "run_name": "baseline_run_001",
    "tags": {
        "team": "ml_research",
        "version": "v1.0",
        "model_type": "transformer"
    }
}

tracelet.start_logging(
    exp_name="tagged_experiment",
    project="research",
    backend="mlflow",
    config=mlflow_config
)
```

## MLflow Server Setup

### Local MLflow Server

```bash
# Start local MLflow server
mlflow server --host 0.0.0.0 --port 5000

# With backend store (SQLite)
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
```

### Remote MLflow Server

```python
# Connect to remote MLflow server
tracelet.start_logging(
    exp_name="remote_experiment",
    project="distributed_training",
    backend="mlflow",
    config={
        "tracking_uri": "http://mlflow.company.com:5000",
        "experiment_name": "Production Models"
    }
)
```

### Docker MLflow Setup

```yaml
# docker-compose.yml
version: "3.8"
services:
  mlflow:
    image: mlflow/mlflow
    ports:
      - "5000:5000"
    command: >
      mlflow server
      --backend-store-uri sqlite:///mlflow.db
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0
      --port 5000
    volumes:
      - mlflow_data:/mlflow

volumes:
  mlflow_data:
```

## Advanced Usage

### Experiment Organization

```python
# Organize experiments by project/team
tracelet.start_logging(
    exp_name="resnet_baseline",
    project="computer_vision",
    backend="mlflow",
    config={
        "experiment_name": "CV Team - Image Classification",
        "tags": {
            "project": "computer_vision",
            "team": "cv_team",
            "model_family": "resnet",
            "dataset": "imagenet"
        }
    }
)
```

### Artifact Management

```python
import tracelet
import torch

# Start experiment
exp = tracelet.start_logging(
    exp_name="artifact_demo",
    project="model_artifacts",
    backend="mlflow"
)

# Log model artifacts
torch.save(model.state_dict(), "model.pth")
exp.log_artifact("model.pth", "models/trained_model.pth")

# Log configuration files
exp.log_artifact("config.yaml", "configs/training_config.yaml")

# Log processed datasets
exp.log_artifact("train_data.csv", "datasets/processed_train.csv")
```

### Model Registry Integration

```python
import mlflow
import tracelet

# Register model after training
exp = tracelet.start_logging(
    exp_name="model_registry_demo",
    project="production_models",
    backend="mlflow"
)

# Train model and log metrics
# ... training code ...

# Register model in MLflow Model Registry
mlflow.pytorch.log_model(
    pytorch_model=model,
    artifact_path="model",
    registered_model_name="ProductionImageClassifier"
)
```

### Auto-logging Integration

```python
import mlflow
import mlflow.pytorch
import tracelet

# Enable MLflow auto-logging
mlflow.pytorch.autolog()

# Start Tracelet (will work alongside auto-logging)
tracelet.start_logging(
    exp_name="autolog_experiment",
    project="auto_tracking",
    backend="mlflow"
)

# Training code - metrics logged automatically by both systems
model = torch.nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters())
# ... training loop ...
```

## Error Handling

### Connection Issues

```python
import tracelet

try:
    tracelet.start_logging(
        exp_name="connection_test",
        project="reliability",
        backend="mlflow",
        config={"tracking_uri": "http://unreachable-server:5000"}
    )
except Exception as e:
    print(f"MLflow connection failed: {e}")

    # Fallback to local tracking
    tracelet.start_logging(
        exp_name="connection_test_local",
        project="reliability",
        backend="mlflow"  # Uses default local tracking
    )
```

### Experiment Creation

```python
# Handle experiment name conflicts
import mlflow
import tracelet

experiment_name = "Shared Experiment"

try:
    tracelet.start_logging(
        exp_name="shared_run",
        project="collaborative",
        backend="mlflow",
        config={"experiment_name": experiment_name}
    )
except mlflow.exceptions.MlflowException as e:
    if "already exists" in str(e):
        print(f"Experiment '{experiment_name}' already exists, joining it")
        # MLflow will automatically use existing experiment
    else:
        raise e
```

## Integration Examples

### PyTorch Training Loop

```python
import torch
import torch.nn as nn
import tracelet

# Start experiment
exp = tracelet.start_logging(
    exp_name="pytorch_training",
    project="deep_learning",
    backend="mlflow"
)

# Log hyperparameters
exp.log_params({
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "model": "resnet18"
})

# Training loop
model = torch.nn.Linear(784, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Log epoch metrics
    avg_loss = total_loss / len(dataloader)
    exp.log_metric("train_loss", avg_loss, iteration=epoch)

    # Save checkpoint every 10 epochs
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")
        exp.log_artifact(
            f"checkpoint_epoch_{epoch}.pth",
            f"checkpoints/epoch_{epoch}.pth"
        )

tracelet.stop_logging()
```

### Hyperparameter Optimization

```python
import optuna
import tracelet

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Start experiment for this trial
    exp = tracelet.start_logging(
        exp_name=f"optuna_trial_{trial.number}",
        project="hyperparameter_optimization",
        backend="mlflow",
        config={
            "tags": {
                "optimization": "optuna",
                "trial_number": str(trial.number)
            }
        }
    )

    # Log trial parameters
    exp.log_params({
        "lr": lr,
        "batch_size": batch_size,
        "trial_number": trial.number
    })

    # Train model with suggested parameters
    accuracy = train_model(lr=lr, batch_size=batch_size, experiment=exp)

    # Log final result
    exp.log_metric("final_accuracy", accuracy)
    tracelet.stop_logging()

    return accuracy

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

## Best Practices

### Experiment Naming

```python
# Use descriptive experiment names
tracelet.start_logging(
    exp_name=f"resnet50_imagenet_{datetime.now().strftime('%Y%m%d_%H%M')}",
    project="computer_vision",
    backend="mlflow",
    config={
        "experiment_name": "CV Team - ImageNet Classification",
        "tags": {
            "architecture": "resnet50",
            "dataset": "imagenet",
            "date": datetime.now().isoformat()
        }
    }
)
```

### Metric Organization

```python
# Use hierarchical metric names
exp.log_metric("train/loss", train_loss, iteration=epoch)
exp.log_metric("train/accuracy", train_acc, iteration=epoch)
exp.log_metric("val/loss", val_loss, iteration=epoch)
exp.log_metric("val/accuracy", val_acc, iteration=epoch)
exp.log_metric("lr_schedule/learning_rate", current_lr, iteration=epoch)
```

### Resource Management

```python
# Ensure proper cleanup
import atexit
import tracelet

exp = tracelet.start_logging(
    exp_name="robust_experiment",
    project="production",
    backend="mlflow"
)

# Register cleanup function
atexit.register(tracelet.stop_logging)

try:
    # Training code here
    pass
finally:
    # Always stop logging
    tracelet.stop_logging()
```
