# Experiment

::: tracelet.core.experiment.Experiment
options:
show_source: true
show_bases: true
merge_init_into_class: true
heading_level: 2

## Core Methods

### Metric Logging

#### log_metric

::: tracelet.core.experiment.Experiment.log_metric
options:
show_source: true
heading_level: 4

**Example Usage:**

```python
import tracelet

# Start experiment
exp = tracelet.start_logging(exp_name="metrics_demo", project="examples", backend="mlflow")

# Log scalar metrics
exp.log_metric("loss", 0.1, iteration=100)
exp.log_metric("accuracy", 0.95, iteration=100)
exp.log_metric("learning_rate", 0.001, iteration=100)

# Log metrics over time
for epoch in range(10):
    train_loss = 1.0 / (epoch + 1)  # Decreasing loss
    exp.log_metric("train_loss", train_loss, iteration=epoch)

    if epoch % 2 == 0:  # Log validation every 2 epochs
        val_loss = train_loss * 1.1
        exp.log_metric("val_loss", val_loss, iteration=epoch)
```

### Parameter Logging

#### log_params

::: tracelet.core.experiment.Experiment.log_params
options:
show_source: true
heading_level: 4

**Example Usage:**

```python
# Log hyperparameters
exp.log_params({
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "optimizer": "adam",
    "model_type": "resnet50"
})

# Log model architecture details
exp.log_params({
    "num_layers": 18,
    "hidden_dim": 512,
    "dropout": 0.2,
    "activation": "relu"
})

# Log data preprocessing parameters
exp.log_params({
    "data_augmentation": True,
    "normalization": "imagenet",
    "train_split": 0.8,
    "random_seed": 42
})
```

### Artifact Management

#### log_artifact

::: tracelet.core.experiment.Experiment.log_artifact
options:
show_source: true
heading_level: 4

**Example Usage:**

```python
import torch
import matplotlib.pyplot as plt

# Save and log model checkpoint
torch.save(model.state_dict(), "model_checkpoint.pth")
exp.log_artifact("model_checkpoint.pth", "models/checkpoint_epoch_10.pth")

# Log training plots
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.savefig("training_curves.png")
exp.log_artifact("training_curves.png", "plots/training_curves.png")

# Log configuration files
exp.log_artifact("config.yaml", "configs/experiment_config.yaml")

# Log processed datasets
exp.log_artifact("processed_data.csv", "data/processed/final_dataset.csv")
```

### Experiment Control

#### set_iteration

::: tracelet.core.experiment.Experiment.set_iteration
options:
show_source: true
heading_level: 4

**Example Usage:**

```python
# Manual iteration tracking
for epoch in range(100):
    exp.set_iteration(epoch)

    # All subsequent metrics logged without iteration will use current iteration
    exp.log_metric("loss", train_loss)  # Uses iteration=epoch
    exp.log_metric("accuracy", train_acc)  # Uses iteration=epoch

    # Override with specific iteration if needed
    exp.log_metric("val_loss", val_loss, iteration=epoch*2)  # Custom iteration
```

#### start and stop

::: tracelet.core.experiment.Experiment.start
options:
show_source: true
heading_level: 4

::: tracelet.core.experiment.Experiment.stop
options:
show_source: true
heading_level: 4

**Example Usage:**

```python
# Manual experiment lifecycle management
exp = tracelet.start_logging(exp_name="manual_control", project="test", backend="mlflow")

# Start tracking (usually called automatically)
exp.start()

# Log metrics during experiment
exp.log_metric("initial_metric", 1.0)

# Stop tracking (usually called by tracelet.stop_logging())
exp.stop()
```

## Configuration

### ExperimentConfig

::: tracelet.core.experiment.ExperimentConfig
options:
show_source: true
show_bases: true
heading_level: 3

**Example Usage:**

```python
from tracelet.core.experiment import ExperimentConfig

# Create custom configuration
config = ExperimentConfig(
    name="custom_experiment",
    project="research_project",
    backend_name="mlflow",
    tags={"team": "ml", "version": "v2.0"},
    tracking_uri="http://mlflow.company.com:5000"
)

# Use with experiment
exp = Experiment(name="test", config=config, backend=mlflow_backend)
```

## Advanced Usage Patterns

### Metric Batching

```python
# Efficient metric logging for large datasets
metrics_batch = {}
for batch_idx, (data, target) in enumerate(dataloader):
    # ... training code ...

    # Collect metrics
    metrics_batch[f"batch_{batch_idx}_loss"] = loss.item()

    # Log in batches every 100 iterations
    if batch_idx % 100 == 0:
        exp.log_params(metrics_batch)
        metrics_batch.clear()
```

### Hierarchical Parameter Organization

```python
# Organize parameters hierarchically
exp.log_params({
    # Model parameters
    "model.architecture": "transformer",
    "model.num_layers": 12,
    "model.hidden_size": 768,

    # Training parameters
    "training.learning_rate": 0.0001,
    "training.batch_size": 16,
    "training.gradient_clip": 1.0,

    # Data parameters
    "data.max_seq_length": 512,
    "data.vocab_size": 30000,
})
```

### Error Handling and Validation

```python
try:
    exp.log_metric("accuracy", accuracy, iteration=epoch)
except Exception as e:
    print(f"Failed to log metric: {e}")
    # Continue training without failing

# Validate parameters before logging
params = {"lr": learning_rate, "batch_size": batch_size}
valid_params = {k: v for k, v in params.items() if v is not None}
exp.log_params(valid_params)
```
