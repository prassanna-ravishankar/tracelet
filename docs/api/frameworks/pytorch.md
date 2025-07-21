# PyTorch Framework Integration

::: tracelet.frameworks.pytorch.PyTorchFramework
options:
show_source: true
show_bases: true
merge_init_into_class: true
heading_level: 2

## Overview

The PyTorch Framework integration provides seamless experiment tracking for PyTorch models with automatic TensorBoard interception.

## Key Features

- **Zero-Code Integration**: Automatically captures `SummaryWriter.add_scalar()` calls
- **TensorBoard Compatibility**: Works with existing TensorBoard logging code
- **Enhanced Metrics**: Supports metadata and metric type classification
- **Manual Logging**: Direct metric logging without TensorBoard

## Basic Usage

### Automatic TensorBoard Interception

```python
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tracelet

# Start experiment with PyTorch framework
tracelet.start_logging(
    exp_name="pytorch_auto_demo",
    project="pytorch_examples",
    backend="mlflow"
)

# Use TensorBoard as normal - metrics automatically captured!
writer = SummaryWriter()

# Training loop
model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(100):
    # Synthetic training step
    data = torch.randn(32, 10)
    target = torch.randn(32, 1)

    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # This gets automatically sent to MLflow!
    writer.add_scalar('Loss/Train', loss.item(), epoch)
    writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

writer.close()
tracelet.stop_logging()
```

### Manual Framework Initialization

```python
from tracelet.frameworks.pytorch import PyTorchFramework
from tracelet.core.experiment import Experiment
import tracelet

# Manual framework setup
exp = tracelet.start_logging(
    exp_name="manual_pytorch",
    project="framework_demo",
    backend="mlflow"
)

# Get the PyTorch framework instance
pytorch_framework = exp._frameworks.get("pytorch")  # Internal access

# Or initialize separately
framework = PyTorchFramework(patch_tensorboard=True)
framework.initialize(exp)
framework.start_tracking()
```

## Advanced Features

### Enhanced Metric Logging

::: tracelet.frameworks.pytorch.PyTorchFramework.log_enhanced_metric
options:
show_source: true
heading_level: 3

```python
import tracelet

# Start experiment
exp = tracelet.start_logging(
    exp_name="enhanced_metrics",
    project="advanced_pytorch",
    backend="mlflow"
)

# Get framework for enhanced logging
pytorch_framework = exp._frameworks["pytorch"]

# Log enhanced metrics with metadata
pytorch_framework.log_enhanced_metric(
    name="validation_accuracy",
    value=0.95,
    metric_type="accuracy",
    iteration=100,
    metadata={
        "dataset": "validation",
        "model_checkpoint": "epoch_100",
        "data_split": "val"
    }
)

pytorch_framework.log_enhanced_metric(
    name="training_loss",
    value=0.1,
    metric_type="loss",
    iteration=100,
    metadata={
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 32
    }
)
```

### Multi-Writer Support

```python
import torch
from torch.utils.tensorboard import SummaryWriter
import tracelet

# Start experiment
tracelet.start_logging(
    exp_name="multi_writer",
    project="tensorboard_demo",
    backend="mlflow"
)

# Multiple writers - all captured automatically
train_writer = SummaryWriter(log_dir="runs/train")
val_writer = SummaryWriter(log_dir="runs/validation")

for epoch in range(50):
    # Training metrics
    train_loss = 1.0 / (epoch + 1)  # Decreasing loss
    train_writer.add_scalar("Loss", train_loss, epoch)
    train_writer.add_scalar("Accuracy", min(0.9, epoch * 0.02), epoch)

    # Validation metrics (every 5 epochs)
    if epoch % 5 == 0:
        val_loss = train_loss * 1.1
        val_writer.add_scalar("Loss", val_loss, epoch)
        val_writer.add_scalar("Accuracy", min(0.85, epoch * 0.018), epoch)

train_writer.close()
val_writer.close()
tracelet.stop_logging()
```

### Custom Metric Processing

```python
import tracelet
from torch.utils.tensorboard import SummaryWriter

class CustomMetricProcessor:
    def __init__(self):
        self.metric_history = {}

    def process_metric(self, name, value, iteration):
        """Custom processing for specific metrics."""
        if name not in self.metric_history:
            self.metric_history[name] = []

        self.metric_history[name].append((iteration, value))

        # Log smoothed version for loss metrics
        if "loss" in name.lower():
            if len(self.metric_history[name]) >= 5:
                recent_values = [v for _, v in self.metric_history[name][-5:]]
                smoothed_value = sum(recent_values) / len(recent_values)

                # Get active experiment and log smoothed metric
                exp = tracelet.get_active_experiment()
                if exp:
                    exp.log_metric(f"{name}_smoothed", smoothed_value, iteration)

# Usage
processor = CustomMetricProcessor()

tracelet.start_logging(
    exp_name="custom_processing",
    project="advanced_features",
    backend="mlflow"
)

writer = SummaryWriter()

for epoch in range(100):
    # Noisy loss simulation
    import random
    base_loss = 1.0 / (epoch + 1)
    noisy_loss = base_loss + random.uniform(-0.1, 0.1)

    # Log original (gets processed by custom processor)
    writer.add_scalar("train_loss", noisy_loss, epoch)
    processor.process_metric("train_loss", noisy_loss, epoch)

writer.close()
tracelet.stop_logging()
```

## Integration Patterns

### PyTorch Lightning Integration

```python
import pytorch_lightning as pl
import tracelet
from torch.utils.tensorboard import SummaryWriter

class LightningModelWithTracelet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(28*28, 10)

        # Start Tracelet experiment
        self.experiment = tracelet.start_logging(
            exp_name="lightning_integration",
            project="pytorch_lightning",
            backend="mlflow"
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.layer(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)

        # Log with TensorBoard (automatically captured by Tracelet)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.layer(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)

        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def on_train_end(self):
        # Clean up Tracelet
        tracelet.stop_logging()
```

### Distributed Training

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import tracelet

def train_worker(rank, world_size):
    # Initialize distributed training
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Only rank 0 handles experiment tracking
    if rank == 0:
        tracelet.start_logging(
            exp_name="distributed_training",
            project="multi_gpu",
            backend="mlflow"
        )
        writer = SummaryWriter()

    # Create model and move to GPU
    model = torch.nn.Linear(1000, 10).cuda(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    for epoch in range(100):
        # Training code here...
        loss = torch.randn(1).cuda(rank)  # Simulated loss

        # Gather losses from all ranks
        gathered_losses = [torch.zeros_like(loss) for _ in range(world_size)]
        dist.all_gather(gathered_losses, loss)

        # Only rank 0 logs metrics
        if rank == 0:
            avg_loss = torch.stack(gathered_losses).mean().item()
            writer.add_scalar("train_loss", avg_loss, epoch)

    if rank == 0:
        writer.close()
        tracelet.stop_logging()

# Launch distributed training
def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size)

if __name__ == "__main__":
    main()
```

## Configuration Options

### Framework Settings

```python
from tracelet.settings import TraceletSettings

# Configure PyTorch framework behavior
settings = TraceletSettings(
    project="pytorch_config",
    backend=["mlflow"],
    # Framework-specific settings would go here if supported
)

tracelet.start_logging(
    exp_name="configured_pytorch",
    settings=settings
)
```

### TensorBoard Patch Control

```python
# Disable TensorBoard patching
from tracelet.frameworks.pytorch import PyTorchFramework

framework = PyTorchFramework(patch_tensorboard=False)

# Manual metric logging only
exp = tracelet.start_logging(
    exp_name="manual_only",
    project="no_tensorboard",
    backend="mlflow"
)

# Log metrics directly through framework
framework.initialize(exp)
framework.log_metric("manual_metric", 0.5, iteration=1)
```

## Error Handling

### TensorBoard Import Issues

```python
try:
    import tracelet

    tracelet.start_logging(
        exp_name="safe_tensorboard",
        project="error_handling",
        backend="mlflow"
    )

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

except ImportError as e:
    print(f"TensorBoard not available: {e}")
    # Fall back to manual logging
    exp = tracelet.get_active_experiment()
    exp.log_metric("fallback_metric", 1.0)
```

### Framework Initialization Errors

```python
import tracelet

try:
    exp = tracelet.start_logging(
        exp_name="framework_error_test",
        project="error_handling",
        backend="mlflow"
    )

    # Framework should initialize automatically
    assert "pytorch" in exp._frameworks

except Exception as e:
    print(f"Framework initialization failed: {e}")
    # Continue without framework features
    exp.log_metric("basic_metric", 1.0)
```

## Best Practices

### Metric Naming Conventions

```python
# Use consistent naming patterns
writer.add_scalar("train/loss", train_loss, epoch)
writer.add_scalar("train/accuracy", train_acc, epoch)
writer.add_scalar("val/loss", val_loss, epoch)
writer.add_scalar("val/accuracy", val_acc, epoch)
writer.add_scalar("lr/learning_rate", current_lr, epoch)
writer.add_scalar("gpu/memory_usage", gpu_memory, epoch)
```

### Resource Management

```python
import atexit
import tracelet
from torch.utils.tensorboard import SummaryWriter

# Ensure cleanup on exit
exp = tracelet.start_logging(
    exp_name="resource_managed",
    project="best_practices",
    backend="mlflow"
)

writer = SummaryWriter()

def cleanup():
    writer.close()
    tracelet.stop_logging()

atexit.register(cleanup)

try:
    # Training code here
    pass
finally:
    cleanup()
```

### Performance Optimization

```python
# Batch metric logging for better performance
import tracelet
from torch.utils.tensorboard import SummaryWriter

tracelet.start_logging(
    exp_name="optimized_logging",
    project="performance",
    backend="mlflow"
)

writer = SummaryWriter()

# Log metrics less frequently for long training runs
log_interval = 10  # Log every 10 epochs

for epoch in range(1000):
    # Training code...
    train_loss = 1.0 / (epoch + 1)

    if epoch % log_interval == 0:
        writer.add_scalar("train_loss", train_loss, epoch)
        # Also log accumulated metrics
        writer.add_scalar("avg_loss_10_epochs", train_loss, epoch)

writer.close()
tracelet.stop_logging()
```
