# Artifact Management System

Tracelet's unified artifact system provides a consistent API for handling models, checkpoints, images, datasets, and other ML artifacts across all supported backends. The system features intelligent routing, automatic type detection, and framework-specific integrations.

## Overview

The artifact system solves the problem of platform-specific artifact APIs by providing a unified interface that automatically routes artifacts to optimal backends based on their type and characteristics.

### Key Benefits

- **Universal API**: Same code works across MLflow, W&B, ClearML, and AIM
- **Intelligent Routing**: Automatically selects optimal backend for each artifact type
- **Rich Metadata**: Comprehensive artifact descriptions and versioning
- **Large File Support**: External references for files >100MB to avoid uploads
- **Framework Integration**: Automatic detection of Lightning checkpoints and model saves

## Quick Start

### Enable Artifact Tracking

```python
from tracelet import Experiment

# Enable artifact tracking
exp = Experiment(
    name="my_experiment",
    backend=["mlflow", "wandb"],
    artifacts=True,              # Enable artifact system
    automagic_artifacts=True     # Enable automatic detection
)
exp.start()
```

### Basic Artifact Logging

```python
from tracelet.core.artifacts import ArtifactType

# Create and log a model artifact
model_artifact = exp.create_artifact(
    name="trained_classifier",
    artifact_type=ArtifactType.MODEL,
    description="Trained image classifier model"
)

# Add files to the artifact
model_artifact.add_file("model.pth", "model/classifier.pth")
model_artifact.add_file("config.yaml", "model/config.yaml")

# Add model object with metadata
model_artifact.add_model(pytorch_model, framework="pytorch")

# Log to all backends
results = exp.log_artifact(model_artifact)
print(f"Logged to {len(results)} backends: {list(results.keys())}")
```

## Artifact Types

Tracelet supports 13 semantic artifact types for intelligent routing:

### ML Assets

- **`MODEL`**: Trained models (PyTorch, scikit-learn, etc.)
- **`CHECKPOINT`**: Training checkpoints (.ckpt, .pth)
- **`WEIGHTS`**: Model weights only

### Data Assets

- **`DATASET`**: Training/validation datasets
- **`SAMPLE`**: Evaluation samples and predictions

### Media Assets

- **`IMAGE`**: Single images or image batches
- **`AUDIO`**: Audio files or arrays
- **`VIDEO`**: Video files

### Analysis Assets

- **`VISUALIZATION`**: Plots, charts, attention maps
- **`REPORT`**: HTML reports, notebooks

### Configuration Assets

- **`CONFIG`**: Configuration files (.yaml, .json)
- **`CODE`**: Source code snapshots

### General Assets

- **`CUSTOM`**: User-defined artifacts

## Artifact Creation

### Using the Builder Pattern

```python
# Create artifact with builder pattern
artifact = exp.create_artifact(
    name="experiment_results",
    artifact_type=ArtifactType.REPORT,
    description="Complete experiment results and analysis"
)

# Chain operations
artifact.add_file("report.html", "reports/experiment.html") \
        .add_file("metrics.json", "data/metrics.json") \
        .add_object(summary_dict, "summary", "json")
```

### Adding Files

```python
# Add local files
artifact.add_file(
    local_path="model.pth",
    artifact_path="model/checkpoint.pth",  # Optional: path in artifact
    description="Best model checkpoint"    # Optional: file description
)

# Add multiple files
files = ["model.pth", "config.yaml", "metrics.json"]
for file_path in files:
    artifact.add_file(file_path)
```

### Adding Objects

```python
# Add Python objects with serialization
config = {"learning_rate": 0.001, "batch_size": 32}
artifact.add_object(
    obj=config,
    name="hyperparameters",
    serializer="json",  # or "pickle", "yaml"
    description="Training hyperparameters"
)

# Supported serializers: json, pickle, yaml
```

### Adding External References

```python
# Add reference to external data (large files, cloud storage)
artifact.add_reference(
    uri="s3://my-bucket/large-dataset.tar.gz",
    size_bytes=5_000_000_000,  # 5GB
    description="Training dataset stored in S3"
)

# Supports any URI scheme: s3://, gs://, http://, file://
```

### Adding Model Objects

```python
# Add PyTorch model with framework detection
artifact.add_model(
    model=pytorch_model,
    framework="pytorch",  # auto-detected if not specified
    input_example=sample_input,  # optional
    description="Trained ResNet50 classifier"
)

# Framework auto-detection supports:
# - pytorch, tensorflow, sklearn, xgboost, lightgbm
```

## Intelligent Routing

The artifact system automatically routes artifacts to optimal backends based on type:

### Routing Rules

| Artifact Type              | Optimal Backend | Reason                      |
| -------------------------- | --------------- | --------------------------- |
| MODEL, CHECKPOINT, WEIGHTS | MLflow          | Best model registry support |
| IMAGE, AUDIO, VIDEO        | W&B             | Rich media visualization    |
| DATASET                    | ClearML         | Advanced data management    |
| VISUALIZATION              | W&B             | Interactive plotting        |
| CONFIG, CODE               | All backends    | Universal support           |

### Custom Routing

```python
# Override routing for specific artifacts
artifact.metadata["preferred_backends"] = ["mlflow", "clearml"]

# Size-based routing (files >100MB use references)
artifact.metadata["force_reference"] = True
```

## Automatic Detection

### Framework Integration

Enable automatic artifact detection for supported frameworks:

```python
exp = Experiment(
    name="auto_artifacts",
    backend=["mlflow"],
    artifacts=True,
    automagic_artifacts=True  # Enable auto-detection
)
```

### PyTorch Lightning Integration

Automatically captures:

- **Checkpoints**: Best, last, and periodic checkpoints
- **Final Models**: Complete model at training end
- **Validation Samples**: Periodic validation predictions
- **Training Metadata**: Epoch, step, metrics, hyperparameters

```python
import pytorch_lightning as pl

# Enable artifact auto-detection
exp = Experiment("lightning_training", artifacts=True, automagic_artifacts=True)
exp.start()

# Your normal Lightning training - artifacts captured automatically
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, datamodule)
# Checkpoints, final model, and samples logged automatically

exp.stop()
```

### PyTorch Integration

Automatically captures:

- **Model Saves**: `torch.save()` calls
- **State Dicts**: Model and optimizer states
- **Checkpoints**: Training resume points

```python
import torch

# Enable auto-detection
exp = Experiment("pytorch_training", artifacts=True, automagic_artifacts=True)
exp.start()

# Your normal PyTorch code - saves captured automatically
torch.save(model.state_dict(), "model.pth")  # Automatically logged
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, "checkpoint.pth")  # Automatically logged as checkpoint

exp.stop()
```

### Filesystem Watching

Monitor directories for new artifacts:

```python
from tracelet.core.experiment import ExperimentConfig

config = ExperimentConfig(
    enable_artifacts=True,
    watch_filesystem=True,  # Enable filesystem watching
    artifact_watch_dirs=["./checkpoints", "./outputs", "./models"]
)

exp = Experiment("filesystem_watch", config=config, artifacts=True)
exp.start()

# Any files created in watched directories are automatically logged
# with type detection based on filename and content
```

## Artifact Retrieval

### List Artifacts

```python
# List all artifacts
artifacts = exp.list_artifacts()

# Filter by type
model_artifacts = exp.list_artifacts(type_filter=ArtifactType.MODEL)

# Filter by backend
mlflow_artifacts = exp.list_artifacts(backend="mlflow")
```

### Get Specific Artifact

```python
# Get latest version
artifact = exp.get_artifact("trained_classifier")

# Get specific version
artifact = exp.get_artifact("trained_classifier", version="v1.2.0")

# Get from specific backend
artifact = exp.get_artifact("trained_classifier", backend="mlflow")
```

## Configuration

### Experiment Configuration

```python
from tracelet.core.experiment import ExperimentConfig

config = ExperimentConfig(
    # Artifact settings
    enable_artifacts=True,
    automagic_artifacts=True,
    watch_filesystem=False,  # Resource intensive
    artifact_watch_dirs=["./checkpoints", "./outputs"]
)

exp = Experiment("configured_exp", config=config)
```

### Environment Variables

```bash
# Enable artifact tracking
export TRACELET_ENABLE_ARTIFACTS=true
export TRACELET_AUTOMAGIC_ARTIFACTS=true

# Filesystem watching
export TRACELET_WATCH_FILESYSTEM=false
export TRACELET_ARTIFACT_WATCH_DIRS="./checkpoints,./outputs"
```

### Settings Configuration

```python
from tracelet.settings import TraceletSettings

settings = TraceletSettings(
    enable_artifacts=True,
    automagic_artifacts=True,
    watch_filesystem=False,
    artifact_watch_dirs=["./checkpoints", "./outputs"]
)
```

## Advanced Usage

### Custom Artifact Types

```python
# Use CUSTOM type for specialized artifacts
custom_artifact = exp.create_artifact(
    name="optimization_trace",
    artifact_type=ArtifactType.CUSTOM,
    description="Custom optimization algorithm trace"
)

# Add custom metadata
custom_artifact.metadata.update({
    "algorithm": "custom_optimizer",
    "convergence_criteria": "gradient_norm < 1e-6",
    "custom_type": "optimization_trace"
})
```

### Batch Operations

```python
# Create multiple related artifacts
artifacts = []
for fold in range(5):
    artifact = exp.create_artifact(
        name=f"cv_fold_{fold}",
        artifact_type=ArtifactType.MODEL
    )
    artifact.add_file(f"fold_{fold}_model.pth")
    artifacts.append(artifact)

# Log all at once
results = {}
for artifact in artifacts:
    results.update(exp.log_artifact(artifact))
```

### Cross-Platform Compatibility

```python
# Ensure artifacts work across all backends
artifact = exp.create_artifact("universal_model", ArtifactType.MODEL)

# Add files in platform-neutral paths
artifact.add_file("model.pth", "model/weights.pth")  # MLflow style
artifact.add_file("config.json", "config.json")     # W&B style

# Add platform-specific metadata
artifact.metadata.update({
    "mlflow_flavor": "pytorch",
    "wandb_type": "model",
    "clearml_framework": "pytorch"
})
```

## Performance Considerations

### Large Files

```python
# For files >100MB, use references instead of uploads
large_artifact = exp.create_artifact("large_dataset", ArtifactType.DATASET)

# Upload small metadata files
large_artifact.add_file("dataset_info.json")
large_artifact.add_file("sample.png")  # Preview

# Reference large data externally
large_artifact.add_reference(
    uri="s3://my-bucket/full-dataset.tar.gz",
    size_bytes=10_000_000_000,  # 10GB
    description="Complete training dataset"
)
```

### Memory Usage

```python
# Avoid loading large objects into memory
artifact = exp.create_artifact("efficient_logging", ArtifactType.REPORT)

# Stream large files instead of loading
with open("large_report.html", "rb") as f:
    artifact.add_file("large_report.html")  # File streamed, not loaded

# Use references for very large objects
artifact.add_reference("file:///path/to/large/file.bin")
```

## Troubleshooting

### Common Issues

**Artifacts not being logged**:

```python
# Check if artifacts are enabled
exp = Experiment("test", artifacts=True)  # Must enable artifacts

# Check backend support
print(exp._artifact_manager.get_stats())  # Debug info
```

**Automagic detection not working**:

```python
# Enable debug logging
import logging
logging.getLogger("tracelet.automagic").setLevel(logging.DEBUG)

# Check framework availability
exp._is_framework_available("pytorch_lightning")  # Returns bool
```

**File stability issues**:

```python
# For large files being written, the system waits for stability
# Default: 5 seconds max wait, 3 stable checks
# Files are checked for size stability and lock status
```

**Memory issues with large artifacts**:

```python
# Use references for large files
artifact.add_reference("s3://bucket/large-file.bin", size_bytes=size)

# Or disable filesystem watching
config = ExperimentConfig(watch_filesystem=False)
```

## API Reference

### TraceletArtifact

Main artifact class with builder pattern support.

**Constructor**:

```python
TraceletArtifact(name: str, artifact_type: ArtifactType, description: str = None)
```

**Methods**:

- `add_file(local_path, artifact_path=None, description=None)` - Add local file
- `add_object(obj, name, serializer="json", description=None)` - Add Python object
- `add_reference(uri, size_bytes=None, description=None)` - Add external reference
- `add_model(model, framework=None, input_example=None, description=None)` - Add ML model
- `to_dict()` - Serialize to dictionary

**Properties**:

- `name: str` - Artifact name
- `type: ArtifactType` - Artifact type
- `description: str` - Artifact description
- `size_bytes: int` - Total size in bytes
- `files: List[ArtifactFile]` - List of files
- `objects: List[ArtifactObject]` - List of objects
- `references: List[ArtifactReference]` - List of external references
- `metadata: dict` - Additional metadata

### Experiment Artifact Methods

**`create_artifact(name, artifact_type, description=None)`**:
Create new artifact builder.

**`log_artifact(artifact)`**:
Log artifact to all configured backends. Returns dict of backend results.

**`get_artifact(name, version="latest", backend=None)`**:
Retrieve artifact by name and version.

**`list_artifacts(type_filter=None, backend=None)`**:
List available artifacts with optional filtering.

## Examples

See the [examples directory](https://github.com/prassanna-ravishankar/tracelet/tree/main/examples/06_artifacts) for comprehensive usage examples:

- `basic_artifact_example.py` - Basic artifact logging
- `lightning_artifacts.py` - PyTorch Lightning integration
- `pytorch_artifacts.py` - PyTorch integration
- `multimedia_artifacts.py` - Image, audio, video handling
- `large_artifacts.py` - Handling large files and datasets
