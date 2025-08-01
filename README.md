# tracelet

[![Release](https://img.shields.io/github/v/release/prassanna-ravishankar/tracelet)](https://img.shields.io/github/v/release/prassanna-ravishankar/tracelet)
[![Build status](https://img.shields.io/github/actions/workflow/status/prassanna-ravishankar/tracelet/main.yml?branch=main)](https://github.com/prassanna-ravishankar/tracelet/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/prassanna-ravishankar/tracelet/branch/main/graph/badge.svg)](https://codecov.io/gh/prassanna-ravishankar/tracelet)
[![Commit activity](https://img.shields.io/github/commit-activity/m/prassanna-ravishankar/tracelet)](https://github.com/prassanna-ravishankar/tracelet/commit-activity/m/prassanna-ravishankar/tracelet)
[![License](https://img.shields.io/github/license/prassanna-ravishankar/tracelet)](https://img.shields.io/github/license/prassanna-ravishankar/tracelet)

<p align="center">
  <img src="https://github.com/prassanna-ravishankar/tracelet/raw/main/docs/tracelet.webp" alt="Tracelet Logo" width="120" height="120">
</p>

Tracelet is an intelligent experiment tracking library that automatically captures PyTorch and PyTorch Lightning metrics, seamlessly integrating with popular experiment tracking platforms through a modular plugin system. With **automagic instrumentation**, Tracelet can automatically detect and log hyperparameters from your code with zero configuration.

## Key Features

### 🔌 Modular Plugin System

- Dynamic plugin discovery and lifecycle management
- Easy to extend with custom backends and collectors
- Thread-safe metric routing with configurable workers
- Dependency resolution for complex plugin hierarchies

### 🚀 Automatic Metric Capture

- 🔮 **Automagic Instrumentation** - Zero-config hyperparameter detection and logging
- 🔄 PyTorch TensorBoard integration - automatically captures `writer.add_scalar()` calls
- ⚡ PyTorch Lightning support - seamlessly tracks trainer metrics
- 📊 System metrics monitoring (CPU, Memory, GPU support planned)
- 📝 Automatic Git repository and environment tracking

### 🎯 Unified Artifact System

- 📦 **Universal Artifact API** - Models, checkpoints, images, audio, datasets, reports
- 🧠 **Intelligent Platform Routing** - Automatically routes artifacts to optimal backends
- ⚡ **Framework Integration** - Auto-capture Lightning checkpoints and validation samples
- 🗂️ **Flexible Storage** - Files, objects, external references, and metadata
- 🎨 **Rich Media Support** - Images, audio, video with automatic visualization

### 🎯 Production-Ready Backends

- **MLflow** - Local and remote server support with full experiment tracking
- **ClearML** - Enterprise-grade experiment management with artifact storage
- **Weights & Biases** - Cloud-based tracking with rich visualizations
- **AIM** - Open-source experiment tracking with powerful UI

### 🛡️ Robust Architecture

- Thread-safe data flow orchestration
- Backpressure handling for high-frequency metrics
- Configurable metric routing and filtering
- Comprehensive error handling and logging

## Installation

Install the base package (includes PyTorch, TensorBoard, and W&B):

```bash
pip install tracelet
```

### Additional Backend Dependencies

Install specific backends as needed:

```bash
# Additional backend integrations
pip install tracelet[mlflow]     # MLflow backend
pip install tracelet[clearml]    # ClearML backend
pip install tracelet[aim]        # AIM backend (Python <3.13)

# Framework integrations
pip install tracelet[lightning]  # PyTorch Lightning support
pip install tracelet[automagic]  # Automagic instrumentation

# Install multiple extras
pip install tracelet[mlflow,clearml]        # Multiple backends
pip install tracelet[backends]              # All backends
pip install tracelet[all]                   # Everything
```

**Base dependencies included**: PyTorch, TorchVision, TensorBoard, Weights & Biases, GitPython, Psutil

**Supported Python versions**: 3.9, 3.10, 3.11, 3.12, 3.13

**Note**: The AIM backend currently requires Python <3.13 due to dependency constraints.

## Demo

<p align="center">
  <a href="https://github.com/prassanna-ravishankar/tracelet/raw/main/docs/video.mp4">
    <img src="https://img.shields.io/badge/🎥_Watch_Demo_Video-4.7MB_MP4-red?style=for-the-badge&logo=youtube&logoColor=white" alt="Watch Demo Video" />
  </a>
</p>

**📺 See Tracelet in action!** Click the button above to download and watch our demo video showing how easy it is to get started with automatic experiment tracking.

> **Note**: GitHub doesn't support embedded video playback in README files. The link above will download the MP4 file directly.

## Quick Start

### The Simplest Way - Automagic! ✨

```python
from tracelet import Experiment

# Just 3 lines to add to ANY training code!
exp = Experiment(name="my_model", backend=["wandb"], automagic=True)
exp.start()

# Your normal PyTorch training - metrics logged automatically!
for epoch in range(10):
    loss = model.train_one_epoch()  # We capture everything!
    print(f"Loss: {loss}")  # Even prints can be captured!

exp.stop()
```

### Manual Tracking

```python
from tracelet import Experiment

# Create an experiment
experiment = Experiment(
    name="my_experiment",
    backend=["mlflow"]  # or ["wandb"], ["clearml"], or multiple!
)
experiment.start()

# Log metrics manually
for epoch in range(10):
    loss = train_model()
    experiment.log_metric("loss", loss, epoch)
    experiment.log_metric("accuracy", accuracy, epoch)

experiment.stop()
```

### PyTorch Lightning Integration ⚡

```python
from tracelet import Experiment
import pytorch_lightning as pl

# Add tracking to ANY Lightning code in 3 lines!
exp = Experiment(name="lightning_model", backend=["clearml"], automagic=True)
exp.start()

# Your existing Lightning code - completely unchanged!
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, datamodule)  # All self.log() calls tracked automatically!

exp.stop()
```

### 🎯 Artifact Tracking

Tracelet's unified artifact system handles models, checkpoints, images, datasets, and more across all backends:

```python
from tracelet import Experiment
from tracelet.core.artifacts import ArtifactType

# Enable artifact tracking
exp = Experiment(
    name="model_training",
    backend=["mlflow", "wandb"],
    artifacts=True,  # Enable artifact system
    automagic_artifacts=True  # Auto-capture Lightning checkpoints
)
exp.start()

# 1. Log a trained model
model_artifact = exp.create_artifact(
    name="classifier",
    artifact_type=ArtifactType.MODEL,
    description="Trained image classifier"
)
model_artifact.add_file("model.pth", "model/classifier.pth")
model_artifact.add_model(pytorch_model, framework="pytorch")
exp.log_artifact(model_artifact)

# 2. Log prediction samples
image_artifact = exp.create_artifact(
    name="predictions",
    artifact_type=ArtifactType.IMAGE
).add_file("sample.png", "samples/prediction.png")
exp.log_artifact(image_artifact)

# 3. Log external dataset reference
dataset_artifact = exp.create_artifact(
    name="training_data",
    artifact_type=ArtifactType.DATASET
).add_reference(
    "s3://bucket/dataset.tar.gz",
    size_bytes=5_000_000_000  # 5GB dataset
)
exp.log_artifact(dataset_artifact)

exp.stop()
```

**Key Features:**

- **13 artifact types**: MODEL, CHECKPOINT, IMAGE, AUDIO, VIDEO, DATASET, etc.
- **Intelligent routing**: Models → MLflow, Media → W&B, optimized per backend
- **Large file handling**: External references for files >100MB
- **Rich metadata**: Framework info, performance metrics, descriptions
- **Auto-detection**: Lightning checkpoints captured automatically

### 📚 More Examples

Check out the [examples/](examples/) directory for:

- Simple tutorials for beginners
- Backend-specific features (W&B, MLflow, ClearML)
- PyTorch Lightning integration
- **Artifact tracking examples** (examples/06_artifacts/)
- Advanced multi-backend setups
- Complete ML pipelines

### 🔮 Automagic Instrumentation

Tracelet's most powerful feature is **automagic instrumentation** - zero-configuration automatic detection and logging of machine learning hyperparameters. Just enable automagic and Tracelet intelligently captures your experiment parameters:

```python
import tracelet
from tracelet import Experiment

# Enable automagic mode - that's it!
experiment = Experiment(
    name="automagic_experiment",
    backend=["mlflow"],
    automagic=True  # ✨ The magic happens here!
)
experiment.start()

# Define your hyperparameters normally
learning_rate = 0.001
batch_size = 64
epochs = 100
dropout = 0.3
hidden_layers = [256, 128, 64]
optimizer = "adam"

# Your training code here...
# All hyperparameters are automatically captured and logged!

experiment.end()
```

#### How Automagic Works

Automagic uses intelligent heuristics to detect ML-relevant parameters:

- **📝 Name patterns**: `learning_rate`, `batch_size`, `num_layers`
- **🔢 Value ranges**: 0.001-0.1 for learning rates, 16-512 for batch sizes
- **📊 Data types**: floats in (0,1) for rates, ints for counts
- **🏷️ Keywords**: `rate`, `size`, `dim`, `num`, `alpha`, `beta`
- **✅ Boolean flags**: `use_*`, `enable_*`, `has_*`
- **📝 String configs**: optimizer names, activation functions

#### Framework Integration

Automagic automatically hooks into popular ML frameworks:

```python
# PyTorch - automatic capture of model parameters, loss, gradients
model = torch.nn.Sequential(...)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Training loop - metrics captured automatically via hooks
for epoch in range(epochs):
    loss = criterion(model(data), targets)
    optimizer.step()  # Learning rate automatically logged
    # Loss and gradient norms captured via framework hooks
```

#### Comparison: Manual vs Automagic

**Manual Tracking** (traditional approach):

```python
# 🔧 MANUAL: Explicit logging required
experiment.log_params({
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 100,
    "dropout": 0.3,
    "hidden_layers": [256, 128, 64],
    "optimizer": "adam",
    # ... 20+ more parameters
})
```

**Automagic Tracking** (zero-config):

```python
# 🔮 AUTOMAGIC: Just define variables normally
learning_rate = 0.001
batch_size = 64
epochs = 100
dropout = 0.3
hidden_layers = [256, 128, 64]
optimizer = "adam"
# All parameters automatically captured! ✨
```

**Benefits:**

- 🚀 **95% fewer logging calls** compared to manual tracking
- 🧠 **Intelligent parameter detection** with ML-specific heuristics
- 🔧 **Framework hooks** automatically capture training metrics
- ⚡ **Real-time monitoring** with minimal overhead
- 🎯 **Focus on research**, not logging boilerplate

### Advanced Configuration

```python
import tracelet
from tracelet import get_active_experiment

# Start with custom configuration
experiment = tracelet.start_logging(
    exp_name="advanced_example",
    project="my_project",
    backend="mlflow",
    config={
        "track_system": True,              # System monitoring
        "metrics_interval": 5.0,           # Log every 5 seconds
        "track_git": True,                 # Git info tracking
        "track_env": True,                 # Environment capture
        "track_tensorboard": True,         # Auto-capture TB metrics
        "track_lightning": True,           # PyTorch Lightning support

        # Artifact tracking settings
        "enable_artifacts": True,          # Enable artifact system
        "automagic_artifacts": True,       # Auto-capture framework artifacts
        "watch_filesystem": False,         # File system watching (resource intensive)
        "artifact_watch_dirs": ["./checkpoints", "./outputs"]  # Directories to watch
    },
    automagic=True                         # Enable automagic instrumentation
)

# Log custom parameters
experiment.log_params({
    "model": "resnet50",
    "batch_size": 32,
    "learning_rate": 0.001
})

# Log custom metrics programmatically
for epoch in range(10):
    metrics = train_epoch()
    experiment.log_metric("accuracy", metrics["acc"], epoch)
    experiment.log_metric("loss", metrics["loss"], epoch)
```

## Configuration

Tracelet can be configured via environment variables or through the settings interface:

```python
from tracelet.settings import TraceletSettings

settings = TraceletSettings(
    project="my_project",               # or project_name (alias)
    backend=["mlflow"],                 # List of backends
    track_system=True,                  # System metrics tracking
    metrics_interval=10.0,              # Collection interval
    track_tensorboard=True,             # TensorBoard integration
    track_lightning=True,               # PyTorch Lightning support
    track_git=True,                     # Git repository info
    track_env=True,                     # Environment capture
    enable_automagic=True,              # Enable automagic instrumentation
    automagic_frameworks={"pytorch", "sklearn", "xgboost"},  # Frameworks to instrument

    # Artifact tracking settings
    enable_artifacts=True,              # Enable artifact system
    automagic_artifacts=True,           # Auto-capture framework artifacts
    watch_filesystem=False,             # File system watching
    artifact_watch_dirs=["./checkpoints", "./outputs"]  # Watch directories
)
```

Key environment variables:

- `TRACELET_PROJECT`: Project name
- `TRACELET_BACKEND`: Comma-separated backends ("mlflow,wandb")
- `TRACELET_BACKEND_URL`: Backend server URL
- `TRACELET_API_KEY`: API key for backend service
- `TRACELET_TRACK_SYSTEM`: Enable system metrics tracking
- `TRACELET_METRICS_INTERVAL`: System metrics collection interval
- `TRACELET_TRACK_TENSORBOARD`: Enable TensorBoard integration
- `TRACELET_TRACK_LIGHTNING`: Enable PyTorch Lightning support
- `TRACELET_TRACK_GIT`: Enable Git repository tracking
- `TRACELET_TRACK_ENV`: Enable environment capture
- `TRACELET_ENABLE_AUTOMAGIC`: Enable automagic instrumentation
- `TRACELET_AUTOMAGIC_FRAMEWORKS`: Comma-separated frameworks ("pytorch,sklearn")
- `TRACELET_ENABLE_ARTIFACTS`: Enable artifact tracking system
- `TRACELET_AUTOMAGIC_ARTIFACTS`: Enable automatic artifact detection
- `TRACELET_WATCH_FILESYSTEM`: Enable file system watching for artifacts
- `TRACELET_ARTIFACT_WATCH_DIRS`: Comma-separated watch directories

## Plugin Development

Tracelet's plugin system makes it easy to add new backends or metric collectors:

```python
from tracelet.core.plugins import BackendPlugin, PluginMetadata, PluginType

class MyCustomBackend(BackendPlugin):
    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="my_backend",
            version="1.0.0",
            type=PluginType.BACKEND,
            description="My custom experiment tracking backend"
        )

    def initialize(self, config: dict):
        # Set up your backend connection
        self.client = MyBackendClient(config["api_key"])

    def log_metric(self, name: str, value: float, iteration: int):
        # Send metrics to your backend
        self.client.log(name, value, iteration)
```

Backends are automatically discovered based on installed dependencies:

```python
import tracelet

# Check available backends
print(tracelet.available_backends())  # ['mlflow', 'wandb', 'clearml']

# Get a specific backend dynamically
mlflow_backend = tracelet.get_backend('mlflow')
```

## Documentation

For more detailed documentation, visit:

- [Documentation](https://prassanna-ravishankar.github.io/tracelet/)
- [GitHub Repository](https://github.com/prassanna-ravishankar/tracelet/)
- [Examples](https://github.com/prassanna-ravishankar/tracelet/tree/main/examples)

## Architecture

Tracelet uses a sophisticated multi-threaded architecture:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Framework  │────▶│ Orchestrator │────▶│   Backend   │
│  (PyTorch)  │     │   (Router)   │     │  (MLflow)   │
└─────────────┘     └──────────────┘     └─────────────┘
       │                    │                     │
       ▼                    ▼                     ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Collector  │────▶│    Queue     │────▶│   Plugin    │
│  (System)   │     │  (Threaded)  │     │  (ClearML)  │
└─────────────┘     └──────────────┘     └─────────────┘
```

- **Metric Sources**: Frameworks and collectors that generate metrics
- **Orchestrator**: Routes metrics to appropriate backends based on rules
- **Backends**: Plugins that handle experiment tracking and storage

## Performance

Tracelet is designed for minimal overhead:

- Non-blocking metric collection using thread-safe queues
- Configurable worker threads for parallel processing
- Automatic backpressure handling to prevent memory issues
- Efficient metric batching for reduced network calls

## Troubleshooting

### Common Issues

**Import errors for backends**: Make sure you've installed the appropriate extras:

```bash
# If you see: ImportError: MLflow is not installed
pip install tracelet[mlflow]
```

**ClearML offline mode**: For testing or CI environments without ClearML credentials:

```python
import os
os.environ["CLEARML_WEB_HOST"] = ""
os.environ["CLEARML_API_HOST"] = ""
os.environ["CLEARML_FILES_HOST"] = ""
```

**High memory usage**: Disable unnecessary tracking features:

```python
experiment = tracelet.start_logging(
    config={
        "track_system": False,          # Disable system metrics
        "track_git": False,             # Disable git tracking
        "metrics_interval": 30.0,       # Reduce collection frequency
    }
)
```

## Roadmap

- [x] 🔮 **Automagic Instrumentation** - Zero-config hyperparameter detection
- [x] 🎯 **Unified Artifact System** - Models, checkpoints, media, datasets with intelligent routing
- [ ] AWS SageMaker integration
- [ ] Prometheus metrics export
- [ ] Real-time metric streaming
- [ ] Web UI for local experiments
- [ ] Distributed training support
- [ ] Enhanced automagic model architecture capture
- [ ] Automagic dataset profiling and statistics
- [ ] Artifact versioning and lineage tracking
- [ ] Cross-platform artifact synchronization

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](https://github.com/prassanna-ravishankar/tracelet/blob/main/CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/prassanna-ravishankar/tracelet.git
cd tracelet

# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
make check
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/prassanna-ravishankar/tracelet/blob/main/LICENSE) file for details.

## Acknowledgments

- Built with the excellent [uv](https://github.com/astral-sh/uv) package manager
- Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv)
- Thanks to all contributors and the open-source community!
