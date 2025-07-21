# Best Practices

This guide covers best practices for using Tracelet effectively in your machine learning projects.

## Experiment Organization

### Naming Conventions

Use consistent, descriptive naming for experiments and metrics:

```python
# Good: Descriptive experiment names
exp_name = f"resnet50_imagenet_{datetime.now().strftime('%Y%m%d_%H%M')}"
project = "image_classification"

# Good: Hierarchical metric names
writer.add_scalar('Loss/Train/CrossEntropy', ce_loss, step)
writer.add_scalar('Loss/Train/L2Regularization', l2_loss, step)
writer.add_scalar('Metrics/Accuracy/Train', train_acc, step)
writer.add_scalar('Metrics/Accuracy/Validation', val_acc, step)

# Avoid: Vague or flat naming
exp_name = "experiment_1"
writer.add_scalar('loss', loss, step)
writer.add_scalar('acc', acc, step)
```

### Project Structure

Organize your experiments by project and phase:

```python
# Development phase
tracelet.start_logging(
    exp_name="model_v1_dev",
    project="sentiment_analysis_dev",
    backend="mlflow"
)

# Production experiments
tracelet.start_logging(
    exp_name="model_v2_prod",
    project="sentiment_analysis_prod",
    backend=["mlflow", "wandb"]  # Multi-backend for production
)
```

## Metric Logging Strategy

### What to Log

**Essential Metrics:**

- Training and validation loss
- Key performance metrics (accuracy, F1, etc.)
- Learning rate and other hyperparameters
- System metrics (GPU/CPU usage, memory)

**Useful Metrics:**

- Gradient norms and parameter distributions
- Intermediate layer activations
- Sample predictions and visualizations
- Timing and performance benchmarks

**Avoid Over-logging:**

- Don't log every single parameter
- Limit high-resolution images and large tensors
- Reduce frequency for expensive operations

### Logging Frequency

Balance detail with performance:

```python
# High-frequency: Essential metrics every step
writer.add_scalar('Loss/Train', loss, step)

# Medium-frequency: Performance metrics every epoch
if step % steps_per_epoch == 0:
    writer.add_scalar('Accuracy/Validation', val_acc, epoch)
    writer.add_scalar('Learning_Rate', scheduler.get_lr()[0], epoch)

# Low-frequency: Expensive operations occasionally
if step % 1000 == 0:
    # Histograms of model parameters
    for name, param in model.named_parameters():
        writer.add_histogram(f'Parameters/{name}', param, step)

if step % 5000 == 0:
    # Sample predictions and visualizations
    writer.add_image('Predictions', prediction_grid, step)
```

## Multi-Backend Strategy

### When to Use Multiple Backends

**Development**: Single backend (usually MLflow for local development)

```python
tracelet.start_logging("dev_experiment", backend="mlflow")
```

**Team Collaboration**: Shared backend (W&B or ClearML)

```python
tracelet.start_logging("team_experiment", backend="wandb")
```

**Production**: Multiple backends for redundancy

```python
tracelet.start_logging(
    "prod_experiment",
    backend=["mlflow", "wandb"]  # Local + cloud backup
)
```

### Backend Selection Guide

| Use Case              | Recommended Backend | Reason                                  |
| --------------------- | ------------------- | --------------------------------------- |
| Local development     | MLflow              | Lightweight, no external dependencies   |
| Team collaboration    | Weights & Biases    | Excellent sharing and visualization     |
| Enterprise deployment | ClearML             | Enterprise features, self-hosted option |
| Research projects     | AIM                 | Open source, powerful analysis tools    |
| Production systems    | MLflow + W&B        | Local reliability + cloud visualization |

## Performance Optimization

### Memory Management

```python
# Configure memory-efficient logging
tracelet.start_logging(
    "memory_efficient_exp",
    backend="mlflow",
    config={
        "track_system": True,
        "metrics_interval": 30.0,  # Reduce system monitoring frequency
        "max_image_size": "512KB", # Limit image sizes
        "track_tensorboard": True,
        "track_lightning": True,
    }
)

# In your training loop
if step % 100 == 0:  # Reduce logging frequency
    # Resize large images before logging
    small_image = F.interpolate(large_image, size=(224, 224))
    writer.add_image('Sample', small_image, step)
```

### Network Efficiency

For cloud backends, batch operations when possible:

```python
# Good: Batch related metrics
metrics = {
    'Loss/Train': train_loss,
    'Loss/Validation': val_loss,
    'Accuracy/Train': train_acc,
    'Accuracy/Validation': val_acc
}
for name, value in metrics.items():
    writer.add_scalar(name, value, step)
```

## Reproducibility

### Environment Tracking

Enable comprehensive environment tracking:

```python
tracelet.start_logging(
    "reproducible_experiment",
    backend="mlflow",
    config={
        "track_git": True,      # Git commit and status
        "track_env": True,      # Python packages and versions
        "track_system": True,   # Hardware information
    }
)

# Log important hyperparameters
experiment = tracelet.get_active_experiment()
experiment.log_params({
    'model_architecture': 'ResNet50',
    'optimizer': 'AdamW',
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'data_augmentation': True,
    'random_seed': 42
})
```

### Random Seed Management

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed before training
set_seed(42)

# Log the seed
experiment.log_params({'random_seed': 42})
```

## Error Handling and Recovery

### Robust Experiment Setup

```python
import tracelet
from contextlib import contextmanager

@contextmanager
def experiment_context(exp_name, project, backend, config=None):
    """Context manager for safe experiment handling"""
    try:
        experiment = tracelet.start_logging(exp_name, project, backend, config)
        yield experiment
    except Exception as e:
        print(f"Experiment setup failed: {e}")
        # Fallback to basic logging
        experiment = tracelet.start_logging(f"{exp_name}_fallback", project, "mlflow")
        yield experiment
    finally:
        tracelet.stop_logging()

# Usage
with experiment_context("my_experiment", "my_project", "wandb") as exp:
    # Your training code here
    pass
```

### Graceful Degradation

```python
def safe_log_metric(writer, name, value, step):
    """Safely log metrics with fallback"""
    try:
        writer.add_scalar(name, value, step)
    except Exception as e:
        print(f"Failed to log {name}: {e}")
        # Continue training even if logging fails

def safe_log_image(writer, name, image, step):
    """Safely log images with size limits"""
    try:
        # Limit image size to prevent memory issues
        if image.numel() > 1000000:  # 1M pixels
            image = F.interpolate(image, size=(512, 512))
        writer.add_image(name, image, step)
    except Exception as e:
        print(f"Failed to log image {name}: {e}")
```

## Team Collaboration

### Shared Experiments

Use consistent configuration across team members:

```python
# shared_config.py
TRACELET_CONFIG = {
    "project": "team_project_2024",
    "backend": "wandb",
    "config": {
        "track_git": True,
        "track_env": True,
        "track_system": True,
        "metrics_interval": 60.0,
    }
}

# In individual scripts
from shared_config import TRACELET_CONFIG

tracelet.start_logging(
    exp_name=f"member_{os.getenv('USER')}_experiment",
    **TRACELET_CONFIG
)
```

### Experiment Handoffs

Document experiments thoroughly:

```python
# Log experiment metadata
experiment.log_params({
    'researcher': 'alice@company.com',
    'experiment_purpose': 'hyperparameter_tuning',
    'baseline_experiment': 'exp_v1_baseline',
    'notes': 'Testing different learning rate schedules',
    'next_steps': 'Try cosine annealing scheduler'
})

# Log important files and configurations
experiment.log_artifact('config.yaml')
experiment.log_artifact('model_architecture.py')
```

## Security and Privacy

### Sensitive Data Handling

```python
# Don't log sensitive information
safe_config = config.copy()
safe_config.pop('api_key', None)
safe_config.pop('password', None)
experiment.log_params(safe_config)

# Use environment variables for credentials
import os
wandb_key = os.getenv('WANDB_API_KEY')
```

### Data Filtering

```python
def filter_sensitive_metrics(metrics_dict):
    """Remove sensitive information from metrics"""
    filtered = {}
    for key, value in metrics_dict.items():
        if 'password' not in key.lower() and 'secret' not in key.lower():
            filtered[key] = value
    return filtered

# Apply filtering before logging
safe_metrics = filter_sensitive_metrics(all_metrics)
for name, value in safe_metrics.items():
    writer.add_scalar(name, value, step)
```
