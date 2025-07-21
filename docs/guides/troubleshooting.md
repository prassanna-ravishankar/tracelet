# Troubleshooting

This guide helps you diagnose and resolve common issues when using Tracelet.

## Installation Issues

### ImportError: No module named 'tracelet'

**Problem**: Tracelet is not installed or not in the Python path.

**Solution**:

```bash
# Install Tracelet
pip install tracelet

# Or with specific backends
pip install tracelet[mlflow,wandb]

# For development
pip install -e ".[dev]"
```

### Backend Import Errors

**Problem**: `ImportError: MLflow is not installed` or similar for other backends.

**Solution**: Install the specific backend extras:

```bash
pip install tracelet[mlflow]     # For MLflow
pip install tracelet[clearml]    # For ClearML
pip install tracelet[aim]        # For AIM
pip install tracelet[all]        # For all backends
```

## Connection Issues

### MLflow Server Connection Failed

**Problem**: Cannot connect to MLflow tracking server.

**Diagnosis**:

```python
import mlflow
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
```

**Solutions**:

```bash
# Start local MLflow server
mlflow server --host 127.0.0.1 --port 5000

# Or set tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000
```

```python
# In code
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
```

### W&B Authentication Issues

**Problem**: `wandb.errors.UsageError: api_key not configured`

**Solution**:

```bash
# Login to W&B
wandb login

# Or set API key
export WANDB_API_KEY=your_api_key_here
```

### ClearML Offline Mode

**Problem**: ClearML requires internet connection.

**Solution** (for testing/CI):

```python
import os
os.environ["CLEARML_WEB_HOST"] = ""
os.environ["CLEARML_API_HOST"] = ""
os.environ["CLEARML_FILES_HOST"] = ""
```

## Metric Logging Issues

### Metrics Not Appearing in Backend

**Problem**: TensorBoard metrics logged but not showing in MLflow/W&B.

**Diagnosis**:

```python
# Check if Tracelet is active
import tracelet
print(f"Active experiment: {tracelet.get_active_experiment()}")
print(f"Active backends: {tracelet.get_active_backends()}")
```

**Solutions**:

1. Ensure `tracelet.start_logging()` is called before creating `SummaryWriter`
2. Check backend configuration:

```python
tracelet.start_logging(
    exp_name="test",
    backend="mlflow",
    config={"track_tensorboard": True}  # Ensure this is True
)
```

### Duplicate Metrics

**Problem**: Same metrics appearing multiple times.

**Cause**: Multiple experiment tracking tools running simultaneously.

**Solution**: Use only Tracelet for experiment tracking:

```python
# Don't use multiple loggers simultaneously
# writer = SummaryWriter()  # Tracelet handles this
# mlflow.log_metric()       # Avoid direct backend calls
# wandb.log()              # Let Tracelet route metrics
```

### Missing Lightning Metrics

**Problem**: PyTorch Lightning metrics not captured.

**Solutions**:

1. Start Tracelet before creating Trainer:

```python
tracelet.start_logging("experiment", backend="mlflow")
trainer = pl.Trainer()  # Create after start_logging
```

2. Enable Lightning tracking:

```python
tracelet.start_logging(
    "experiment",
    backend="mlflow",
    config={"track_lightning": True}
)
```

## Performance Issues

### High Memory Usage

**Problem**: Tracelet consuming too much memory.

**Solutions**:

```python
# Reduce system monitoring frequency
tracelet.start_logging(
    "experiment",
    backend="mlflow",
    config={
        "track_system": False,          # Disable system monitoring
        "metrics_interval": 60.0,       # Reduce frequency
        "max_image_size": "512KB",      # Limit image sizes
    }
)

# In training loop - reduce logging frequency
if step % 100 == 0:  # Log less frequently
    writer.add_histogram('weights', model.weights, step)
```

### Slow Training

**Problem**: Training significantly slower with Tracelet.

**Diagnosis**:

```python
import time

# Test with and without Tracelet
start = time.time()
# Your training step
elapsed = time.time() - start
print(f"Training step took {elapsed:.3f}s")
```

**Solutions**:

1. Reduce logging frequency:

```python
# Log metrics less frequently
if step % 10 == 0:  # Instead of every step
    writer.add_scalar('loss', loss, step)
```

2. Disable expensive operations:

```python
config = {
    "track_system": False,      # Disable system monitoring
    "track_git": False,         # Disable git tracking
    "capture_histograms": False # Disable histogram capture
}
```

### Network Timeouts

**Problem**: Timeouts when logging to cloud backends.

**Solutions**:

1. Use local backend as fallback:

```python
try:
    tracelet.start_logging("exp", backend="wandb")
except Exception:
    print("W&B failed, falling back to MLflow")
    tracelet.start_logging("exp", backend="mlflow")
```

2. Configure timeout settings:

```python
import wandb
wandb.Settings(base_url="https://api.wandb.ai", timeout=60)
```

## Platform-Specific Issues

### Windows Path Issues

**Problem**: File path errors on Windows.

**Solution**:

```python
import os
from pathlib import Path

# Use pathlib for cross-platform paths
log_dir = Path("./runs") / "experiment_1"
writer = SummaryWriter(log_dir=str(log_dir))
```

### M1 Mac Compatibility

**Problem**: Some backends not working on Apple Silicon.

**Solutions**:

1. Install x86_64 version:

```bash
arch -x86_64 pip install tracelet[all]
```

2. Use conda-forge:

```bash
conda install -c conda-forge tracelet
```

### Docker Container Issues

**Problem**: Backends not accessible from Docker.

**Solution**:

```dockerfile
# In Dockerfile
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000
ENV WANDB_API_KEY=your_api_key

# For MLflow server access
EXPOSE 5000
```

## Debugging Tips

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or Tracelet-specific logging
tracelet_logger = logging.getLogger('tracelet')
tracelet_logger.setLevel(logging.DEBUG)
```

### Check Configuration

```python
# Print current configuration
experiment = tracelet.get_active_experiment()
print(f"Experiment: {experiment}")
print(f"Config: {experiment.config if experiment else 'No active experiment'}")
```

### Test Backend Connectivity

```python
def test_backend(backend_name):
    """Test if backend is accessible"""
    try:
        tracelet.start_logging(f"test_{backend_name}", backend=backend_name)
        experiment = tracelet.get_active_experiment()
        experiment.log_metric("test_metric", 1.0, 0)
        print(f"✅ {backend_name} working")
        tracelet.stop_logging()
        return True
    except Exception as e:
        print(f"❌ {backend_name} failed: {e}")
        return False

# Test all backends
for backend in ["mlflow", "wandb", "clearml", "aim"]:
    test_backend(backend)
```

## Getting Help

### Check System Information

```python
import tracelet
print(tracelet.__version__)

import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
```

### Minimal Reproduction

When reporting issues, provide a minimal example:

```python
import tracelet
from torch.utils.tensorboard import SummaryWriter

# Minimal failing example
tracelet.start_logging("test_experiment", backend="mlflow")
writer = SummaryWriter()
writer.add_scalar("test", 1.0, 0)
writer.close()
tracelet.stop_logging()
```

### Common Error Messages

| Error Message                                                     | Likely Cause               | Solution                                     |
| ----------------------------------------------------------------- | -------------------------- | -------------------------------------------- |
| `ModuleNotFoundError: No module named 'mlflow'`                   | Backend not installed      | `pip install tracelet[mlflow]`               |
| `ConnectionError: HTTPConnectionPool`                             | Backend server not running | Start MLflow server                          |
| `wandb.errors.UsageError: api_key not configured`                 | W&B not authenticated      | `wandb login`                                |
| `AttributeError: 'NoneType' object has no attribute 'log_metric'` | No active experiment       | Call `start_logging()` first                 |
| `RuntimeError: CUDA out of memory`                                | GPU memory exhausted       | Reduce batch size or disable system tracking |

For additional help, please:

- Check our [GitHub Issues](https://github.com/prassanna-ravishankar/tracelet/issues)
- Join our [Discussions](https://github.com/prassanna-ravishankar/tracelet/discussions)
- Email us at [support@tracelet.io](mailto:support@tracelet.io)
