# Tracelet

<p align="center">
  <img src="https://raw.githubusercontent.com/prassanna-ravishankar/tracelet/main/docs/tracelet.webp" alt="Tracelet Logo" width="200">
</p>

<p align="center">
  <a href="https://github.com/prassanna-ravishankar/tracelet/releases">
    <img src="https://img.shields.io/github/v/release/prassanna-ravishankar/tracelet" alt="Release">
  </a>
  <a href="https://github.com/prassanna-ravishankar/tracelet/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/prassanna-ravishankar/tracelet/main.yml?branch=main" alt="Build">
  </a>
  <a href="https://github.com/prassanna-ravishankar/tracelet/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/prassanna-ravishankar/tracelet" alt="License">
  </a>
</p>

Tracelet is a Python library for tracking machine learning experiments. It works with popular backends like MLflow, Weights & Biases, ClearML, and AIM, and can automatically detect hyperparameters from your code.

## What it does

- Tracks metrics, parameters, and artifacts across ML backends
- Automatically captures hyperparameters without manual logging
- Integrates with PyTorch, PyTorch Lightning, and TensorBoard
- Handles models, checkpoints, images, and other ML artifacts
- Monitors system resources during training

## Installation

```bash
pip install tracelet
```

Install backend support as needed:

```bash
pip install tracelet[mlflow]     # For MLflow
pip install tracelet[wandb]      # For Weights & Biases
pip install tracelet[clearml]    # For ClearML
pip install tracelet[aim]        # For AIM
pip install tracelet[all]        # Everything
```

## Usage

### Basic Example

```python
from tracelet import Experiment

# Start tracking
exp = Experiment(name="my_experiment", backend=["mlflow"])
exp.start()

# Train your model
for epoch in range(10):
    loss = train_model()
    exp.log_metric("loss", loss, epoch)

exp.stop()
```

### Automatic Detection

```python
from tracelet import Experiment

# Enable automagic to detect hyperparameters automatically
exp = Experiment(name="auto_experiment", backend=["wandb"], automagic=True)
exp.start()

# Define hyperparameters normally - they're captured automatically
learning_rate = 0.001
batch_size = 32

# Your training code here
for epoch in range(epochs):
    loss = train_model()
    # Metrics logged automatically if using TensorBoard

exp.stop()
```

### PyTorch Lightning

```python
from tracelet import Experiment
import pytorch_lightning as pl

# Add to existing Lightning code
exp = Experiment(name="lightning_model", backend=["clearml"], automagic=True)
exp.start()

# Your existing code - all self.log() calls tracked automatically
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, datamodule)

exp.stop()
```

### Multiple Backends

```python
# Log to multiple backends simultaneously
exp = Experiment(
    name="comparison",
    backend=["mlflow", "wandb", "clearml"]
)
```

## Configuration

Set defaults with environment variables:

```bash
export TRACELET_PROJECT="my_project"
export TRACELET_BACKEND="mlflow"
export TRACELET_ENABLE_AUTOMAGIC="true"
```

Or programmatically:

```python
from tracelet.settings import TraceletSettings

settings = TraceletSettings(
    project="my_project",
    backend=["mlflow"],
    enable_automagic=True,
    track_system=True
)
```

## Examples

The `examples/` directory contains runnable examples:

- `01_manual_tracking/` - Basic usage examples
- `02_automagic_tracking/` - Automatic hyperparameter detection
- `03_backend_integrations/` - Backend-specific features
- `05_lightning_automagic/` - PyTorch Lightning integration

```bash
cd examples
python 01_manual_tracking/01_basic_manual.py
```

## Documentation

- [Documentation](https://prassanna-ravishankar.github.io/tracelet/)
- [API Reference](https://prassanna-ravishankar.github.io/tracelet/api/)
- [Configuration Guide](https://prassanna-ravishankar.github.io/tracelet/settings/)

## Development

```bash
git clone https://github.com/prassanna-ravishankar/tracelet.git
cd tracelet
uv sync
uv run pytest
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) file.

## Support

- [Issues](https://github.com/prassanna-ravishankar/tracelet/issues) for bug reports
- [Discussions](https://github.com/prassanna-ravishankar/tracelet/discussions) for questions
