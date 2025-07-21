# PyTorch Lightning Integration

Tracelet automatically captures PyTorch Lightning training metrics without any code modifications.

## Overview

The Lightning integration hooks into the Lightning framework's logging system to capture all metrics logged via `self.log()` calls in your LightningModule.

## Supported Features

- **Training Metrics** - Loss, accuracy, custom metrics
- **Validation Metrics** - Validation loss, metrics from validation_step
- **Test Metrics** - Test phase metrics
- **Hyperparameters** - Model and trainer configuration
- **System Metrics** - GPU utilization, memory usage during training

## Basic Usage

```python
import tracelet
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# Start Tracelet before creating trainer
tracelet.start_logging(
    exp_name="lightning_experiment",
    project="my_project",
    backend="mlflow"
)

# Define your LightningModule as usual
class MyModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        # All these metrics are automatically captured by Tracelet
        self.log('train/loss', loss)
        self.log('train/accuracy', self.compute_accuracy(batch))

        return loss

    def validation_step(self, batch, batch_idx):
        val_loss = self.compute_loss(batch)
        val_acc = self.compute_accuracy(batch)

        # Validation metrics are also captured
        self.log('val/loss', val_loss)
        self.log('val/accuracy', val_acc)

# Train your model - metrics automatically tracked
model = MyModel()
trainer = Trainer(max_epochs=10)
trainer.fit(model, train_dataloader, val_dataloader)

# Stop tracking
tracelet.stop_logging()
```

## Advanced Configuration

```python
# Configure Lightning-specific tracking
tracelet.start_logging(
    exp_name="advanced_lightning",
    backend=["mlflow", "wandb"],  # Multi-backend logging
    config={
        "track_lightning": True,        # Enable Lightning integration
        "track_system": True,           # Monitor system resources
        "track_git": True,              # Track git information
        "metrics_interval": 10.0,       # System metrics every 10 seconds
    }
)
```

## Best Practices

### Metric Naming

Use consistent, hierarchical naming:

```python
def training_step(self, batch, batch_idx):
    # Good: Hierarchical naming
    self.log('train/loss', loss)
    self.log('train/accuracy', accuracy)
    self.log('train/f1_score', f1)

    # Also good: Phase-specific metrics
    self.log('metrics/train_loss', loss)
    self.log('metrics/train_acc', accuracy)
```

### Logging Frequency

Control when metrics are logged:

```python
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)

    # Log every step
    self.log('train/loss', loss, on_step=True, on_epoch=False)

    # Log epoch averages
    self.log('train/epoch_loss', loss, on_step=False, on_epoch=True)

    # Log both
    self.log('train/loss_detailed', loss, on_step=True, on_epoch=True)
```

### Custom Metrics

Log custom metrics and hyperparameters:

```python
def on_train_start(self):
    # Log hyperparameters
    self.logger.log_hyperparams({
        'learning_rate': self.learning_rate,
        'batch_size': self.batch_size,
        'model_name': self.__class__.__name__
    })

def training_step(self, batch, batch_idx):
    # Custom metrics
    predictions = self.forward(batch)
    custom_metric = self.compute_custom_metric(predictions, batch)

    self.log('custom/my_metric', custom_metric)
```

## Multi-GPU Support

Tracelet works seamlessly with Lightning's distributed training:

```python
# Works with DDP, DDP2, etc.
trainer = Trainer(
    accelerator='gpu',
    devices=4,
    strategy='ddp'
)

# Metrics from all processes are automatically aggregated
trainer.fit(model)
```

## Integration with Callbacks

Use with Lightning callbacks:

```python
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

tracelet.start_logging("lightning_with_callbacks", backend="clearml")

trainer = Trainer(
    callbacks=[
        ModelCheckpoint(monitor='val/loss'),
        EarlyStopping(monitor='val/loss', patience=3)
    ],
    max_epochs=100
)

trainer.fit(model)
```

## Troubleshooting

### Common Issues

**Metrics not appearing**: Ensure `tracelet.start_logging()` is called before creating the Trainer.

**Duplicate metrics**: If using multiple loggers, you may see duplicate entries. Use Tracelet as the primary logger.

**Memory issues with large models**: Enable gradient checkpointing and reduce logging frequency for memory-intensive operations.
