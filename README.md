# Tracelet

[![Release](https://img.shields.io/github/v/release/prassanna-ravishankar/tracelet)](https://img.shields.io/github/v/release/prassanna-ravishankar/tracelet)
[![Build status](https://img.shields.io/github/actions/workflow/status/prassanna-ravishankar/tracelet/main.yml?branch=main)](https://github.com/prassanna-ravishankar/tracelet/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/prassanna-ravishankar/tracelet/branch/main/graph/badge.svg)](https://codecov.io/gh/prassanna-ravishankar/tracelet)
[![Commit activity](https://img.shields.io/github/commit-activity/m/prassanna-ravishankar/tracelet)](https://github.com/prassanna-ravishankar/tracelet/commit-activity/m/prassanna-ravishankar/tracelet)
[![License](https://img.shields.io/github/license/prassanna-ravishankar/tracelet)](https://github.com/prassanna-ravishankar/tracelet/license)

Tracelet is an intelligent experiment tracking library for machine learning that automatically captures PyTorch and PyTorch Lightning metrics. It seamlessly integrates with popular MLOps platforms through a modular plugin system and features automatic hyperparameter detection with zero configuration required.

## Features

### Automatic Instrumentation

- **Automagic Detection**: Zero-configuration hyperparameter detection and logging
- **PyTorch Integration**: Automatically captures TensorBoard `writer.add_scalar()` calls
- **Lightning Support**: Seamlessly tracks PyTorch Lightning trainer metrics
- **System Monitoring**: CPU, memory, and GPU usage tracking
- **Environment Capture**: Automatic Git repository and environment information

### Unified Artifact Management

- **Universal API**: Handle models, checkpoints, images, audio, datasets, and reports
- **Intelligent Routing**: Automatically routes artifacts to optimal backends
- **Framework Integration**: Auto-capture Lightning checkpoints and validation samples
- **Flexible Storage**: Support for files, objects, external references, and metadata
- **Rich Media**: Images, audio, and video with automatic visualization

### Multi-Backend Support

- **MLflow**: Local and remote server support with full experiment tracking
- **ClearML**: Enterprise-grade experiment management with artifact storage
- **Weights & Biases**: Cloud-based tracking with rich visualizations
- **AIM**: Open-source experiment tracking with powerful UI

### Production-Ready Architecture

- **Thread-Safe**: Concurrent metric routing with configurable workers
- **Robust**: Backpressure handling for high-frequency metrics
- **Extensible**: Plugin system for custom backends and collectors
- **Reliable**: Comprehensive error handling and logging

## Installation

Install the base package:

```bash
pip install tracelet
```

### Backend Dependencies

Install specific backends as needed:

```bash
# Backend integrations
pip install tracelet[mlflow]     # MLflow backend
pip install tracelet[clearml]    # ClearML backend
pip install tracelet[aim]        # AIM backend (Python <3.13)

# Framework integrations
pip install tracelet[lightning]  # PyTorch Lightning support
pip install tracelet[automagic]  # Automagic instrumentation

# Install combinations
pip install tracelet[mlflow,clearml]  # Multiple backends
pip install tracelet[backends]        # All backends
pip install tracelet[all]             # Everything
```

**Supported Python versions**: 3.9, 3.10, 3.11, 3.12, 3.13

## Quick Start

### Basic Usage

```python
from tracelet import Experiment

# Create experiment with automatic detection
exp = Experiment(name="my_model", backend=["wandb"], automagic=True)
exp.start()

# Your existing training code - metrics logged automatically
for epoch in range(10):
    loss = model.train_one_epoch()
    print(f"Loss: {loss}")

exp.stop()
```

### Manual Tracking

```python
from tracelet import Experiment

# Create experiment
experiment = Experiment(
    name="my_experiment",
    backend=["mlflow"]
)
experiment.start()

# Log metrics manually
for epoch in range(10):
    loss = train_model()
    experiment.log_metric("loss", loss, epoch)
    experiment.log_metric("accuracy", accuracy, epoch)

experiment.stop()
```

### PyTorch Lightning Integration

```python
from tracelet import Experiment
import pytorch_lightning as pl

# Add tracking to existing Lightning code
exp = Experiment(name="lightning_model", backend=["clearml"], automagic=True)
exp.start()

# Your existing Lightning code unchanged
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, datamodule)  # All self.log() calls tracked automatically

exp.stop()
```

### Artifact Tracking

```python
from tracelet import Experiment
from tracelet.core.artifacts import ArtifactType

# Enable artifact tracking
exp = Experiment(
    name="model_training",
    backend=["mlflow", "wandb"],
    artifacts=True,
    automagic_artifacts=True
)
exp.start()

# Log a trained model
model_artifact = exp.create_artifact(
    name="classifier",
    artifact_type=ArtifactType.MODEL,
    description="Trained image classifier"
)
model_artifact.add_file("model.pth", "model/classifier.pth")
model_artifact.add_model(pytorch_model, framework="pytorch")
exp.log_artifact(model_artifact)

# Log prediction samples
image_artifact = exp.create_artifact(
    name="predictions",
    artifact_type=ArtifactType.IMAGE
).add_file("sample.png", "samples/prediction.png")
exp.log_artifact(image_artifact)

# Log external dataset reference
dataset_artifact = exp.create_artifact(
    name="training_data",
    artifact_type=ArtifactType.DATASET
).add_reference(
    "s3://bucket/dataset.tar.gz",
    size_bytes=5_000_000_000
)
exp.log_artifact(dataset_artifact)

exp.stop()
```

## Configuration

### Environment Variables

Configure Tracelet using environment variables:

```bash
export TRACELET_PROJECT="my_project"
export TRACELET_BACKEND="mlflow,wandb"
export TRACELET_ENABLE_AUTOMAGIC="true"
export TRACELET_ENABLE_ARTIFACTS="true"
export TRACELET_TRACK_SYSTEM="true"
```

### Programmatic Configuration

```python
from tracelet.settings import TraceletSettings

settings = TraceletSettings(
    project="my_project",
    backend=["mlflow"],
    track_system=True,
    enable_automagic=True,
    enable_artifacts=True,
    automagic_artifacts=True
)
```

## Documentation

- **[Documentation](https://prassanna-ravishankar.github.io/tracelet/)** - Comprehensive guides and API reference
- **[Quick Start Guide](https://prassanna-ravishankar.github.io/tracelet/quick-start/)** - Get started in 5 minutes
- **[API Reference](https://prassanna-ravishankar.github.io/tracelet/api/)** - Complete API documentation
- **[Examples](https://github.com/prassanna-ravishankar/tracelet/tree/main/examples)** - Real-world usage examples

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](https://github.com/prassanna-ravishankar/tracelet/blob/main/CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/prassanna-ravishankar/tracelet.git
cd tracelet
uv sync
uv run pytest
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/prassanna-ravishankar/tracelet/blob/main/LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/prassanna-ravishankar/tracelet/issues)
- **Discussions**: [GitHub Discussions](https://github.com/prassanna-ravishankar/tracelet/discussions)
- **Email**: [me@prassanna.io](mailto:me@prassanna.io)

---

Built with [uv](https://github.com/astral-sh/uv) and powered by the open-source community.
