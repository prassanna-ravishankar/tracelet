# TensorBoard Integration

Tracelet's TensorBoard integration automatically captures all metrics logged to TensorBoard and routes them to your configured experiment tracking backends.

## Overview

The integration works by transparently patching TensorBoard's `SummaryWriter` class to intercept all logging calls. Your existing TensorBoard code works unchanged while metrics are automatically sent to backends like MLflow, W&B, or ClearML.

## Supported Operations

Tracelet captures all TensorBoard logging operations:

- `add_scalar()` - Scalar metrics (loss, accuracy, etc.)
- `add_histogram()` - Weight distributions, gradients
- `add_image()` - Images, plots, visualizations
- `add_text()` - Text logs, summaries
- `add_audio()` - Audio samples
- `add_figure()` - Matplotlib figures
- `add_graph()` - Model computational graphs

## Basic Usage

```python
import tracelet
from torch.utils.tensorboard import SummaryWriter

# Start Tracelet
tracelet.start_logging(
    exp_name="tensorboard_experiment",
    project="my_project",
    backend="mlflow"
)

# Use TensorBoard exactly as before
writer = SummaryWriter(log_dir='./runs/experiment_1')

# All these operations are automatically captured
for step in range(100):
    # Scalars
    writer.add_scalar('Loss/Train', loss_value, step)
    writer.add_scalar('Accuracy/Train', acc_value, step)

    # Histograms
    writer.add_histogram('Weights/Layer1', model.layer1.weight, step)

    # Images (every 10 steps)
    if step % 10 == 0:
        writer.add_image('Predictions', pred_image, step)

writer.close()
tracelet.stop_logging()
```

## Advanced Features

### Multiple Writers

Tracelet supports multiple SummaryWriter instances:

```python
# Multiple writers for different aspects
train_writer = SummaryWriter('runs/train')
val_writer = SummaryWriter('runs/validation')

# Both are automatically captured
train_writer.add_scalar('loss', train_loss, step)
val_writer.add_scalar('loss', val_loss, step)
```

### Hierarchical Metrics

Organize metrics with forward slashes:

```python
# Creates nested structure in backends
writer.add_scalar('Loss/Train/CrossEntropy', ce_loss, step)
writer.add_scalar('Loss/Train/Regularization', reg_loss, step)
writer.add_scalar('Loss/Validation/Total', val_loss, step)

writer.add_scalar('Metrics/Accuracy/Train', train_acc, step)
writer.add_scalar('Metrics/Accuracy/Validation', val_acc, step)
writer.add_scalar('Metrics/F1/Macro', f1_macro, step)
```

### Custom Tags and Metadata

Add additional context to your metrics:

```python
# Scalars with custom metadata
writer.add_scalar('learning_rate', lr, step)
writer.add_scalar('batch_size', batch_size, step)

# Text logs for additional context
writer.add_text('Config', f"Model: {model_name}, LR: {lr}", step)
writer.add_text('Notes', 'Changed optimizer to AdamW', step)
```

## Configuration Options

Control TensorBoard integration behavior:

```python
tracelet.start_logging(
    exp_name="custom_tensorboard",
    backend="wandb",
    config={
        "track_tensorboard": True,       # Enable TensorBoard capture (default: True)
        "tensorboard_log_dir": "./runs", # TensorBoard log directory
        "capture_images": True,          # Capture add_image() calls
        "capture_histograms": True,      # Capture add_histogram() calls
        "capture_audio": False,          # Skip audio (can be large)
        "max_image_size": "1MB",         # Limit image sizes
    }
)
```

## Performance Considerations

### High-Frequency Logging

For high-frequency metrics, consider batching:

```python
# Good: Batch similar metrics
if step % 10 == 0:  # Log every 10 steps
    writer.add_scalar('Loss/Train', loss, step)

if step % 100 == 0:  # Log expensive operations less frequently
    writer.add_histogram('Gradients', model.gradients, step)

if step % 1000 == 0:  # Log very expensive operations rarely
    writer.add_image('Samples', sample_images, step)
```

### Memory Management

For large tensors and images:

```python
# Limit image resolution
resized_image = F.interpolate(large_image, size=(224, 224))
writer.add_image('Prediction', resized_image, step)

# Log histograms selectively
if step % 500 == 0:  # Reduce frequency for memory-intensive ops
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only log weights, not biases
            writer.add_histogram(f'Weights/{name}', param, step)
```

## Migration from Pure TensorBoard

Migrating existing TensorBoard code is trivial:

### Before (Pure TensorBoard)

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
# ... training loop with writer.add_scalar() calls ...
writer.close()
```

### After (With Tracelet)

```python
import tracelet  # Add this import
from torch.utils.tensorboard import SummaryWriter

tracelet.start_logging("my_experiment", backend="mlflow")  # Add this line
writer = SummaryWriter()
# ... same training loop, no changes needed ...
writer.close()
tracelet.stop_logging()  # Add this line
```

## Troubleshooting

### Common Issues

**Metrics appear in TensorBoard but not backend**: Ensure `tracelet.start_logging()` is called before creating `SummaryWriter`.

**Some metrics missing**: Check if you're using multiple writers - all are captured automatically.

**Performance degradation**: Reduce logging frequency for expensive operations like histograms and images.

**Large file sizes**: Configure limits for images and audio, or reduce logging frequency.

### Debugging

Enable debug logging to see what's being captured:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

tracelet.start_logging("debug_experiment", backend="mlflow")
# ... your code ...
```

This will show all TensorBoard operations being intercepted and routed to backends.
