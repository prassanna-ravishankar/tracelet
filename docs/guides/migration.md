# Migration Guide

This guide helps you migrate from other experiment tracking solutions to Tracelet.

## Migrating from Pure TensorBoard

### Before (Pure TensorBoard)

```python
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

writer = SummaryWriter(log_dir='runs/experiment_1')

for epoch in range(100):
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = F.cross_entropy(output, target)

        # Manual TensorBoard logging
        writer.add_scalar('Loss/Train', loss.item(),
                         epoch * len(train_loader) + batch_idx)

        if batch_idx % 100 == 0:
            writer.add_histogram('Weights/Conv1', model.conv1.weight,
                               epoch * len(train_loader) + batch_idx)

writer.close()
```

### After (With Tracelet)

```python
import tracelet  # Add this import
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# Add this line - everything else stays the same!
tracelet.start_logging(
    exp_name="experiment_1",
    project="my_project",
    backend="mlflow"  # Choose your backend
)

writer = SummaryWriter(log_dir='runs/experiment_1')

# Same training loop - no changes needed!
for epoch in range(100):
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = F.cross_entropy(output, target)

        # Same TensorBoard calls - now automatically routed to MLflow!
        writer.add_scalar('Loss/Train', loss.item(),
                         epoch * len(train_loader) + batch_idx)

        if batch_idx % 100 == 0:
            writer.add_histogram('Weights/Conv1', model.conv1.weight,
                               epoch * len(train_loader) + batch_idx)

writer.close()
tracelet.stop_logging()  # Add this line
```

**Migration effort**: Add 2 lines, zero other changes! ✨

## Migrating from MLflow

### Before (Direct MLflow)

```python
import mlflow
import mlflow.pytorch

# Manual MLflow setup
mlflow.set_experiment("my_experiment")
with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 100
    })

    for epoch in range(100):
        loss = train_epoch()
        accuracy = validate_epoch()

        # Manual MLflow logging
        mlflow.log_metric("train_loss", loss, step=epoch)
        mlflow.log_metric("val_accuracy", accuracy, step=epoch)

    # Log model
    mlflow.pytorch.log_model(model, "model")
```

### After (With Tracelet)

```python
import tracelet
from torch.utils.tensorboard import SummaryWriter

# Simpler setup
tracelet.start_logging(
    exp_name="my_experiment",
    project="my_project",
    backend="mlflow"
)

# Log parameters once
experiment = tracelet.get_active_experiment()
experiment.log_params({
    "learning_rate": 0.01,
    "batch_size": 32,
    "epochs": 100
})

# Use TensorBoard for metrics (cleaner API)
writer = SummaryWriter()
for epoch in range(100):
    loss = train_epoch()
    accuracy = validate_epoch()

    # Simpler metric logging
    writer.add_scalar("train_loss", loss, epoch)
    writer.add_scalar("val_accuracy", accuracy, epoch)

# Still can log model directly if needed
experiment.log_artifact("model.pth")
tracelet.stop_logging()
```

**Benefits**: Simpler API, automatic TensorBoard integration, multi-backend support.

## Migrating from Weights & Biases

### Before (Direct W&B)

```python
import wandb

# Manual W&B setup
wandb.init(
    project="my_project",
    name="experiment_1",
    config={
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 100
    }
)

for epoch in range(100):
    loss = train_epoch()
    accuracy = validate_epoch()

    # Manual W&B logging
    wandb.log({
        "train/loss": loss,
        "val/accuracy": accuracy,
        "epoch": epoch
    })

    # Log images
    if epoch % 10 == 0:
        wandb.log({"predictions": wandb.Image(prediction_image)})

wandb.finish()
```

### After (With Tracelet)

```python
import tracelet
from torch.utils.tensorboard import SummaryWriter

# Simpler setup with same W&B backend
tracelet.start_logging(
    exp_name="experiment_1",
    project="my_project",
    backend="wandb",  # Still uses W&B!
    config={
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 100
    }
)

# Use standard TensorBoard API
writer = SummaryWriter()
for epoch in range(100):
    loss = train_epoch()
    accuracy = validate_epoch()

    # Standard TensorBoard calls - automatically sent to W&B
    writer.add_scalar("train/loss", loss, epoch)
    writer.add_scalar("val/accuracy", accuracy, epoch)

    # Images work the same way
    if epoch % 10 == 0:
        writer.add_image("predictions", prediction_image, epoch)

tracelet.stop_logging()
```

**Benefits**: Standard TensorBoard API, easier to switch backends later, automatic metric capture.

## Migrating from ClearML

### Before (Direct ClearML)

```python
from clearml import Task

# Manual ClearML setup
task = Task.init(
    project_name="my_project",
    task_name="experiment_1"
)

# Get logger
logger = task.get_logger()

# Connect hyperparameters
task.connect({
    "learning_rate": 0.01,
    "batch_size": 32,
    "epochs": 100
})

for epoch in range(100):
    loss = train_epoch()
    accuracy = validate_epoch()

    # Manual ClearML logging
    logger.report_scalar("train", "loss", value=loss, iteration=epoch)
    logger.report_scalar("validation", "accuracy", value=accuracy, iteration=epoch)

task.close()
```

### After (With Tracelet)

```python
import tracelet
from torch.utils.tensorboard import SummaryWriter

# Simpler setup
tracelet.start_logging(
    exp_name="experiment_1",
    project="my_project",
    backend="clearml"
)

# Log hyperparameters
experiment = tracelet.get_active_experiment()
experiment.log_params({
    "learning_rate": 0.01,
    "batch_size": 32,
    "epochs": 100
})

# Standard TensorBoard API
writer = SummaryWriter()
for epoch in range(100):
    loss = train_epoch()
    accuracy = validate_epoch()

    # Clean metric logging
    writer.add_scalar("train/loss", loss, epoch)
    writer.add_scalar("validation/accuracy", accuracy, epoch)

tracelet.stop_logging()
```

## Migrating from PyTorch Lightning Loggers

### Before (Lightning + Multiple Loggers)

```python
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger

# Multiple logger setup
mlf_logger = MLFlowLogger(
    experiment_name="my_experiment",
    tracking_uri="http://localhost:5000"
)
wandb_logger = WandbLogger(
    project="my_project",
    name="experiment_1"
)

class MyModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        # Automatic logging to both loggers
        self.log('train/loss', loss)
        return loss

trainer = pl.Trainer(
    logger=[mlf_logger, wandb_logger],  # Multiple loggers
    max_epochs=100
)
trainer.fit(model)
```

### After (Lightning + Tracelet)

```python
import tracelet
import pytorch_lightning as pl

# Single Tracelet setup for multiple backends
tracelet.start_logging(
    exp_name="experiment_1",
    project="my_project",
    backend=["mlflow", "wandb"]  # Multi-backend with one call!
)

class MyModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        # Same Lightning logging - automatically captured
        self.log('train/loss', loss)
        return loss

# No logger needed - Tracelet handles everything
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model)

tracelet.stop_logging()
```

**Benefits**: Simpler configuration, unified multi-backend logging, automatic capture.

## Migration Strategies

### Gradual Migration

1. **Phase 1**: Add Tracelet alongside existing logging

```python
# Keep your existing logging
mlflow.log_metric("loss", loss, step)

# Add Tracelet in parallel
import tracelet
tracelet.start_logging("parallel_test", backend="mlflow")
writer = SummaryWriter()
writer.add_scalar("loss_tracelet", loss, step)
```

2. **Phase 2**: Switch to TensorBoard API with Tracelet

```python
# Remove direct backend calls
# mlflow.log_metric("loss", loss, step)  # Remove this

# Use only TensorBoard + Tracelet
writer.add_scalar("loss", loss, step)  # Automatically goes to MLflow
```

3. **Phase 3**: Leverage Tracelet's advanced features

```python
# Multi-backend logging
tracelet.start_logging("production_exp", backend=["mlflow", "wandb"])

# Automatic system monitoring
config = {"track_system": True, "track_git": True}
```

### Testing Migration

Create a validation script to ensure metrics match:

```python
def validate_migration():
    """Test that Tracelet produces same metrics as direct backend calls"""

    # Test data
    test_metrics = {"loss": 0.5, "accuracy": 0.95}

    # Method 1: Direct MLflow
    import mlflow
    with mlflow.start_run():
        for name, value in test_metrics.items():
            mlflow.log_metric(name, value, step=0)
        direct_run_id = mlflow.active_run().info.run_id

    # Method 2: Tracelet
    import tracelet
    from torch.utils.tensorboard import SummaryWriter

    tracelet.start_logging("validation_test", backend="mlflow")
    writer = SummaryWriter()
    for name, value in test_metrics.items():
        writer.add_scalar(name, value, 0)
    tracelet.stop_logging()

    print("✅ Migration validation complete - check MLflow UI for matching experiments")

validate_migration()
```

## Best Practices for Migration

### 1. **Start Small**

- Begin with a single experiment
- Use your current backend initially
- Gradually add Tracelet features

### 2. **Maintain Compatibility**

```python
# Keep existing parameter names and structure
# Before: wandb.log({"train_loss": loss})
# After:  writer.add_scalar("train_loss", loss, step)
```

### 3. **Test Thoroughly**

- Compare metric values between old and new systems
- Verify all important metrics are captured
- Test with different experiment configurations

### 4. **Document Changes**

```python
# Document migration in your code
"""
Migration Notes:
- Switched from direct MLflow to Tracelet on 2024-01-15
- TensorBoard metrics now automatically routed to MLflow
- Added multi-backend support for production experiments
"""
```

### 5. **Training Team**

- Update documentation and examples
- Provide migration scripts for common patterns
- Share best practices for new unified workflow

## Common Migration Issues

### Metric Name Differences

Different backends may expect different metric formats:

```python
# Solution: Use consistent hierarchical naming
writer.add_scalar("Loss/Train", loss, step)      # Works everywhere
writer.add_scalar("Metrics/Accuracy", acc, step) # Clear hierarchy
```

### Configuration Migration

Map your existing configuration to Tracelet:

```python
# Old W&B config
wandb_config = {
    "learning_rate": 0.01,
    "architecture": "resnet50"
}

# New Tracelet config (same data, better organization)
experiment = tracelet.get_active_experiment()
experiment.log_params(wandb_config)  # Same parameters
```

### Timing Differences

Some backends may show slight timing differences:

```python
# Use consistent step counting
global_step = epoch * len(dataloader) + batch_idx
writer.add_scalar("loss", loss, global_step)
```

Need help with your specific migration? [Contact us](mailto:support@tracelet.io) or [open an issue](https://github.com/prassanna-ravishankar/tracelet/issues).
