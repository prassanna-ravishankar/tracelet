# PyTorch Integration

Tracelet provides seamless integration with PyTorch through automatic TensorBoard metric capture.

## Overview

The PyTorch integration works by automatically patching TensorBoard's `SummaryWriter` to capture all logged metrics and route them to your configured backends.

## Supported Features

- **Scalar Metrics** - Training loss, validation accuracy, learning rates
- **Histograms** - Weight distributions, gradient histograms
- **Images** - Sample predictions, model visualizations
- **Text** - Training logs, model summaries
- **Audio** - Audio samples and generated content

## Basic Usage

```python
import tracelet
import torch
from torch.utils.tensorboard import SummaryWriter

# Start tracking
tracelet.start_logging(
    exp_name="pytorch_experiment",
    project="my_project",
    backend="mlflow"
)

# Use TensorBoard normally
writer = SummaryWriter()

# Your training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Training code...
        loss = train_step(model, data, target)

        # Log metrics - automatically captured by Tracelet
        writer.add_scalar('Loss/Train', loss.item(), epoch * len(train_loader) + batch_idx)

        if batch_idx % 100 == 0:
            # Log histograms
            for name, param in model.named_parameters():
                writer.add_histogram(f'Parameters/{name}', param, epoch)

# Stop tracking
tracelet.stop_logging()
```

## Advanced Configuration

```python
# Configure specific PyTorch tracking options
tracelet.start_logging(
    exp_name="advanced_pytorch",
    backend="wandb",
    config={
        "track_tensorboard": True,      # Enable TensorBoard capture
        "track_system": True,           # Monitor system metrics
        "metrics_interval": 5.0,        # System metrics every 5 seconds
    }
)
```

## Best Practices

### Metric Organization

Organize your metrics with clear hierarchies:

```python
# Good: Hierarchical naming
writer.add_scalar('Loss/Train', train_loss, step)
writer.add_scalar('Loss/Validation', val_loss, step)
writer.add_scalar('Accuracy/Train', train_acc, step)
writer.add_scalar('Accuracy/Validation', val_acc, step)

# Avoid: Flat naming
writer.add_scalar('train_loss', train_loss, step)
writer.add_scalar('val_loss', val_loss, step)
```

### Performance Optimization

For high-frequency logging:

```python
# Log less frequently for expensive operations
if step % 100 == 0:
    writer.add_histogram('gradients', model.parameters(), step)

if step % 1000 == 0:
    writer.add_image('predictions', sample_predictions, step)
```

## Troubleshooting

### Common Issues

**TensorBoard metrics not appearing**: Ensure you're using `SummaryWriter` after calling `tracelet.start_logging()`.

**Memory issues**: Reduce logging frequency for large tensors like histograms and images.

**Performance impact**: Use `metrics_interval` to control system metrics collection frequency.
