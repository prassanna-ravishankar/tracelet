# tracelet

[![Release](https://img.shields.io/github/v/release/prassanna-ravishankar/tracelet)](https://img.shields.io/github/v/release/prassanna-ravishankar/tracelet)
[![Build status](https://img.shields.io/github/actions/workflow/status/prassanna-ravishankar/tracelet/main.yml?branch=main)](https://github.com/prassanna-ravishankar/tracelet/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/prassanna-ravishankar/tracelet/branch/main/graph/badge.svg)](https://codecov.io/gh/prassanna-ravishankar/tracelet)
[![Commit activity](https://img.shields.io/github/commit-activity/m/prassanna-ravishankar/tracelet)](https://img.shields.io/github/commit-activity/m/prassanna-ravishankar/tracelet)
[![License](https://img.shields.io/github/license/prassanna-ravishankar/tracelet)](https://img.shields.io/github/license/prassanna-ravishankar/tracelet)

Tracelet is an automagic PyTorch metric exporter that seamlessly integrates with popular experiment tracking tools.

## Features

- 🔄 Automatic capture of PyTorch metrics and TensorBoard logs
- 🤝 Integration with multiple tracking backends (MLflow, Weights & Biases, AIM)
- 📊 System metrics monitoring (CPU, GPU, Memory)
- 📝 Git repository tracking
- ⚡ Lightning integration support
- 🔧 Environment variable tracking
- 🎨 Matplotlib figure export support

## Installation

```bash
pip install tracelet
```

## Quick Start

```python
import tracelet
import torch

# Start experiment tracking
experiment = tracelet.start_logging(
    exp_name="my_experiment",
    project="my_project",
    backend="mlflow"  # or "wandb", "aim"
)

# Your PyTorch training code
# Tracelet will automatically capture metrics from TensorBoard, Lightning, etc.

# Stop tracking when done
tracelet.stop_logging()
```

## Configuration

Tracelet can be configured via environment variables or through the settings interface:

```python
from tracelet.settings import TraceletSettings

settings = TraceletSettings(
    project_name="my_project",
    backend="mlflow",
    track_system_metrics=True,
    system_metrics_interval=10.0
)
```

Key environment variables:
- `TRACELET_PROJECT`: Project name
- `TRACELET_BACKEND`: Tracking backend ("mlflow", "wandb", "aim")
- `TRACELET_BACKEND_URL`: Backend server URL
- `TRACELET_API_KEY`: API key for backend service
- `TRACELET_TRACK_SYSTEM`: Enable system metrics tracking
- `TRACELET_METRICS_INTERVAL`: System metrics collection interval

## Documentation

For more detailed documentation, visit:
- [Documentation](https://prassanna-ravishankar.github.io/tracelet/)
- [GitHub Repository](https://github.com/prassanna-ravishankar/tracelet/)

## Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
