# Core API Reference

This section covers the core APIs and classes that form the foundation of Tracelet.

## Main Interface

### `tracelet.start_logging()`

Start experiment tracking with the specified configuration.

**Parameters:**

- `exp_name: str` - Name of the experiment
- `project: str` - Project name (optional)
- `backend: str | List[str]` - Backend(s) to use
- `config: dict` - Additional configuration (optional)

**Returns:** `Experiment` - The created experiment instance

### `tracelet.stop_logging()`

Stop the current experiment tracking session.

### `tracelet.get_active_experiment()`

Get the currently active experiment instance.

**Returns:** `Experiment | None` - The active experiment, or None if no experiment is active

## Core Classes

### Experiment

The main experiment tracking interface.

#### Constructor

```python
Experiment(
    name: str,
    config: Optional[ExperimentConfig] = None,
    backend: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    automagic: bool = False,
    automagic_config: Optional[AutomagicConfig] = None,
    artifacts: bool = False,
    automagic_artifacts: bool = False
)
```

**Parameters:**

- `name` - Experiment name
- `config` - ExperimentConfig instance for detailed configuration
- `backend` - List of backend names (e.g., ["mlflow", "wandb"])
- `tags` - List of experiment tags
- `automagic` - Enable automatic hyperparameter detection
- `automagic_config` - Custom automagic configuration
- `artifacts` - Enable artifact tracking system
- `automagic_artifacts` - Enable automatic artifact detection

#### Core Methods

**`start()`** - Start the experiment and initialize all backends

**`stop()`** - Stop the experiment and cleanup resources

**`end()`** - Alias for stop()

#### Metric Logging

**`log_metric(name: str, value: Any, iteration: Optional[int] = None)`**
Log a scalar metric value.

**`log_params(params: Dict[str, Any])`**
Log experiment parameters/hyperparameters.

**`log_hyperparameter(name: str, value: Any)`**
Log a single hyperparameter (alias for compatibility).

**`set_iteration(iteration: int)`**
Set the current iteration/step number.

#### Artifact Management

**`create_artifact(name: str, artifact_type: ArtifactType, description: Optional[str] = None) -> TraceletArtifact`**
Create a new artifact builder.

**`log_artifact(artifact: TraceletArtifact) -> Dict[str, ArtifactResult]`**
Log artifact to all configured backends. Returns results from each backend.

**`get_artifact(name: str, version: str = "latest", backend: Optional[str] = None) -> Optional[TraceletArtifact]`**
Retrieve artifact by name and version.

**`list_artifacts(type_filter: Optional[ArtifactType] = None, backend: Optional[str] = None) -> List[TraceletArtifact]`**
List available artifacts with optional filtering.

**`log_file_artifact(local_path: str, artifact_path: Optional[str] = None)`**
Legacy method for logging individual files as artifacts.

#### Automagic Methods

**`capture_hyperparams() -> Dict[str, Any]`**
Manually capture hyperparameters from calling context.

**`capture_model(model: Any) -> Dict[str, Any]`**
Capture model information using automagic instrumentation.

**`capture_dataset(dataset: Any) -> Dict[str, Any]`**
Capture dataset information using automagic instrumentation.

#### Properties

**`iteration: int`** - Current iteration/step number

**`name: str`** - Experiment name

**`id: str`** - Unique experiment ID

**`created_at: datetime`** - Experiment creation timestamp

**`tags: List[str]`** - Experiment tags

### Orchestrator

The Orchestrator class manages metric routing and backend coordination.

**Key Methods:**

- `start()` - Start the orchestrator and worker threads
- `stop()` - Stop all operations gracefully
- `route_metric(metric)` - Route a metric to all configured backends
- `add_backend(backend)` - Add a new backend to the orchestrator
- `remove_backend(backend)` - Remove a backend from the orchestrator

_Full API documentation coming soon._

## Plugin System

### Plugin Interfaces

Tracelet uses a plugin-based architecture for extensibility.

**PluginInterface** - Base interface for all plugins
**BackendPlugin** - Interface for experiment tracking backends
**FrameworkPlugin** - Interface for ML framework integrations

### Plugin Metadata

**PluginMetadata** - Contains plugin information (name, version, description)
**PluginType** - Enum defining plugin types (BACKEND, FRAMEWORK, COLLECTOR)

_Full plugin API documentation coming soon._

## Artifact System

### ArtifactType

Enum defining semantic artifact types for intelligent routing.

```python
class ArtifactType(Enum):
    # ML Assets
    MODEL = "model"          # Trained models
    CHECKPOINT = "checkpoint" # Training checkpoints
    WEIGHTS = "weights"      # Model weights only

    # Data Assets
    DATASET = "dataset"      # Training/validation datasets
    SAMPLE = "sample"        # Evaluation samples/predictions

    # Media Assets
    IMAGE = "image"          # Images or image batches
    AUDIO = "audio"          # Audio files or arrays
    VIDEO = "video"          # Video files

    # Analysis Assets
    VISUALIZATION = "viz"    # Plots, charts, attention maps
    REPORT = "report"        # HTML reports, notebooks

    # Configuration Assets
    CONFIG = "config"        # Configuration files
    CODE = "code"           # Source code snapshots

    # General Assets
    CUSTOM = "custom"       # User-defined artifacts
```

### TraceletArtifact

Main artifact class with builder pattern support.

#### Constructor

```python
TraceletArtifact(
    name: str,
    artifact_type: ArtifactType,
    description: Optional[str] = None
)
```

#### Methods

**`add_file(local_path: str, artifact_path: Optional[str] = None, description: Optional[str] = None) -> TraceletArtifact`**
Add a local file to the artifact. Returns self for chaining.

**`add_object(obj: Any, name: str, serializer: str = "json", description: Optional[str] = None) -> TraceletArtifact`**
Add a Python object with serialization. Supported serializers: "json", "pickle", "yaml".

**`add_reference(uri: str, size_bytes: Optional[int] = None, description: Optional[str] = None) -> TraceletArtifact`**
Add external reference (S3, GCS, HTTP, etc.) for large files.

**`add_model(model: Any, framework: Optional[str] = None, input_example: Optional[Any] = None, description: Optional[str] = None) -> TraceletArtifact`**
Add ML model object with framework detection and metadata.

**`serialize_object(obj: ArtifactObject, temp_dir: str) -> str`**
Serialize object to temporary file for logging.

**`to_dict() -> Dict[str, Any]`**
Convert artifact to dictionary representation.

#### Properties

- `name: str` - Artifact name
- `type: ArtifactType` - Artifact type
- `description: str` - Artifact description
- `size_bytes: int` - Total size in bytes
- `created_at: datetime` - Creation timestamp
- `files: List[ArtifactFile]` - List of files
- `objects: List[ArtifactObject]` - List of objects
- `references: List[ArtifactReference]` - List of external references
- `metadata: Dict[str, Any]` - Additional metadata

### ArtifactFile

Represents a file within an artifact.

#### Properties

- `local_path: str` - Path to local file
- `artifact_path: str` - Path within artifact
- `description: Optional[str]` - File description
- `size_bytes: int` - File size
- `checksum: Optional[str]` - File checksum

### ArtifactObject

Represents a Python object within an artifact.

#### Properties

- `obj: Any` - The Python object
- `name: str` - Object name
- `serializer: str` - Serialization method
- `description: Optional[str]` - Object description

### ArtifactReference

Represents an external reference within an artifact.

#### Properties

- `uri: str` - External URI
- `size_bytes: Optional[int]` - Reference size
- `description: Optional[str]` - Reference description

### ArtifactResult

Result of logging an artifact to a backend.

#### Properties

- `backend: str` - Backend name
- `uri: str` - Artifact URI in backend
- `version: str` - Artifact version
- `metadata: Dict[str, Any]` - Backend-specific metadata

## Configuration

### ExperimentConfig

Configuration class for detailed experiment settings.

```python
@dataclass
class ExperimentConfig:
    # Core tracking settings
    track_metrics: bool = True
    track_environment: bool = True
    track_args: bool = True
    track_stdout: bool = True
    track_checkpoints: bool = True
    track_system_metrics: bool = True
    track_git: bool = True

    # Automagic instrumentation settings
    enable_automagic: bool = False
    automagic_frameworks: Optional[Set[str]] = None

    # Artifact tracking settings
    enable_artifacts: bool = False
    automagic_artifacts: bool = False
    artifact_watch_dirs: Optional[List[str]] = None
    watch_filesystem: bool = False
```

### TraceletSettings

Main configuration class for global settings.

```python
class TraceletSettings:
    project: str                    # Default project name
    backend: List[str]              # Default backends to use
    track_system: bool              # Enable system metrics tracking
    track_git: bool                 # Enable git repository tracking
    track_env: bool                 # Enable environment tracking
    metrics_interval: float         # System metrics collection interval

    # Automagic settings
    enable_automagic: bool          # Enable automagic instrumentation
    automagic_frameworks: Set[str]  # Frameworks to instrument

    # Artifact settings
    enable_artifacts: bool          # Enable artifact tracking
    automagic_artifacts: bool       # Enable automatic artifact detection
    watch_filesystem: bool          # Enable filesystem watching
    artifact_watch_dirs: List[str]  # Directories to watch
```

#### Environment Variable Mapping

| Setting                | Environment Variable            |
| ---------------------- | ------------------------------- |
| `project`              | `TRACELET_PROJECT`              |
| `backend`              | `TRACELET_BACKEND`              |
| `track_system`         | `TRACELET_TRACK_SYSTEM`         |
| `track_git`            | `TRACELET_TRACK_GIT`            |
| `track_env`            | `TRACELET_TRACK_ENV`            |
| `enable_automagic`     | `TRACELET_ENABLE_AUTOMAGIC`     |
| `automagic_frameworks` | `TRACELET_AUTOMAGIC_FRAMEWORKS` |
| `enable_artifacts`     | `TRACELET_ENABLE_ARTIFACTS`     |
| `automagic_artifacts`  | `TRACELET_AUTOMAGIC_ARTIFACTS`  |
| `watch_filesystem`     | `TRACELET_WATCH_FILESYSTEM`     |
| `artifact_watch_dirs`  | `TRACELET_ARTIFACT_WATCH_DIRS`  |

## Exceptions

### Base Exceptions

**TraceletException** - Base exception for all Tracelet errors
**BackendError** - Errors related to backend operations
**ConfigurationError** - Errors in configuration or setup

_Full exception API documentation coming soon._

## Usage Examples

### Basic Usage

```python
import tracelet
from torch.utils.tensorboard import SummaryWriter

# Start experiment tracking
experiment = tracelet.start_logging(
    exp_name="my_experiment",
    project="my_project",
    backend="mlflow"
)

# Use TensorBoard normally - metrics automatically captured
writer = SummaryWriter()
writer.add_scalar('loss', 0.5, 0)
writer.add_scalar('accuracy', 0.95, 0)

# Direct API usage
experiment.log_params({
    "learning_rate": 0.01,
    "batch_size": 32
})

experiment.log_artifact("model.pth")

# Stop tracking
tracelet.stop_logging()
```

### Advanced Configuration

```python
import tracelet

# Multi-backend logging with custom configuration
experiment = tracelet.start_logging(
    exp_name="advanced_experiment",
    project="research_project",
    backend=["mlflow", "wandb"],
    config={
        "track_system": True,
        "track_git": True,
        "track_env": True,
        "metrics_interval": 10.0,
        "mlflow_tracking_uri": "http://localhost:5000",
        "wandb_project": "my-wandb-project"
    }
)

# Get current experiment for direct manipulation
current_exp = tracelet.get_active_experiment()
print(f"Experiment ID: {current_exp.experiment_id}")
print(f"Active backends: {[b.name for b in current_exp.backends]}")
```

### Context Manager Usage

```python
import tracelet

# Use as context manager for automatic cleanup
with tracelet.start_logging("context_experiment", backend="mlflow") as exp:
    # Training code here
    for epoch in range(10):
        loss = train_epoch()
        exp.log_metric("loss", loss, epoch)

    # Experiment automatically closed when exiting context
```

## Type Hints

Tracelet provides comprehensive type hints for better IDE support:

```python
from typing import Dict, List, Optional, Union
from tracelet.core.experiment import Experiment
from tracelet.core.plugins import BackendPlugin

def my_training_function(
    experiment: Experiment,
    hyperparams: Dict[str, Union[str, int, float]],
    backends: Optional[List[str]] = None
) -> None:
    """Example function with proper type hints"""
    experiment.log_params(hyperparams)
    # Training logic here
```

For more detailed information about specific modules, see the dedicated API reference pages for each component.
