# Plugin Development Guide

This guide shows how to create custom plugins for Tracelet. Plugins are the primary way to extend Tracelet with new backends, framework integrations, and data collectors.

## Understanding Tracelet's Architecture

### Plugin vs Backend - Clarification

In Tracelet, **backends are a specific type of plugin**. This is a common source of confusion for contributors, so let's clarify:

- **Plugin**: Generic extensibility system that can be any type of component
- **Backend**: A plugin specifically of type `PluginType.BACKEND` for experiment tracking

```python
# All backends are plugins, but not all plugins are backends
class MLflowBackend(BackendPlugin):  # This IS a plugin
    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="mlflow",
            type=PluginType.BACKEND,  # This makes it a backend-type plugin
            # ...
        )

class SystemCollector(PluginBase):  # This is also a plugin
    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="system",
            type=PluginType.COLLECTOR,  # This makes it a collector-type plugin
            # ...
        )
```

### The Plugin Hierarchy

```
PluginBase (Abstract Base Class)
├── BackendPlugin (inherits from PluginBase)
│   ├── MLflowBackend
│   ├── WandBBackend
│   └── ClearMLBackend
├── CollectorPlugin (inherits from PluginBase)
│   ├── SystemCollector
│   └── GitCollector
├── FrameworkPlugin (inherits from PluginBase)
│   ├── PyTorchFramework
│   └── LightningFramework
└── ProcessorPlugin (inherits from PluginBase)
    └── CustomProcessors
```

### Key Concepts

- **Plugin System**: The overall architecture for extensibility
- **Plugin Types**: Categories of plugins (BACKEND, COLLECTOR, FRAMEWORK, PROCESSOR)
- **Plugin Manager**: Discovers, loads, and manages plugin lifecycle
- **Orchestrator**: Routes data between sources (frameworks) and sinks (backends)

## Plugin Types

Tracelet supports four types of plugins:

- **Backend Plugins**: Experiment tracking backends (MLflow, W&B, etc.)
- **Framework Plugins**: ML framework integrations (PyTorch, Lightning, etc.)
- **Collector Plugins**: Data collectors (system metrics, git info, etc.)
- **Processor Plugins**: Data processors and transformers

## Creating a Backend Plugin

Backend plugins integrate Tracelet with experiment tracking platforms.

### 1. Basic Backend Plugin Structure

```python
# tracelet/backends/neptune_backend.py
from typing import Any, Dict, Optional
from tracelet.core.plugins import BackendPlugin, PluginMetadata, PluginType
from tracelet.core.orchestrator import MetricData, MetricType
from tracelet.utils.imports import require

class NeptuneBackend(BackendPlugin):
    """Neptune.ai backend plugin for experiment tracking."""

    def __init__(self):
        super().__init__()
        self._run = None
        self._project = None

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="neptune",
            version="1.0.0",
            type=PluginType.BACKEND,
            description="Neptune.ai experiment tracking backend",
            author="Your Name",
            dependencies=["neptune"],
            capabilities={"metrics", "parameters", "artifacts", "logging"}
        )

    def initialize(self, config: Dict[str, Any]):
        """Initialize Neptune backend with configuration."""
        # Use dynamic import for optional dependency
        neptune = require("neptune", "Neptune backend")

        self._config = config
        project_name = config.get("project", "workspace/project")
        api_token = config.get("api_token")

        # Initialize Neptune
        self._run = neptune.init_run(
            project=project_name,
            api_token=api_token,
            name=config.get("run_name")
        )

    def start(self):
        """Start the backend."""
        self._active = True

    def stop(self):
        """Stop the backend and cleanup."""
        if self._run:
            self._run.stop()
        self._active = False

    def get_status(self) -> Dict[str, Any]:
        """Get backend status."""
        return {
            "active": self._active,
            "run_id": self._run["sys/id"].fetch() if self._run else None
        }

    def handle_metric(self, metric: MetricData):
        """Handle incoming metric data."""
        if not self._run or not self._active:
            return

        if metric.type == MetricType.SCALAR:
            self._run[metric.name].append(metric.value, step=metric.iteration)
        elif metric.type == MetricType.PARAMETER:
            self._run[f"parameters/{metric.name}"] = metric.value
        elif metric.type == MetricType.ARTIFACT:
            self._run[f"artifacts/{metric.name}"].upload(metric.value)
```

### 2. Register the Backend Plugin

```python
# tracelet/backends/__init__.py
from .neptune_backend import NeptuneBackend

def get_backend(name: str):
    """Get backend plugin by name."""
    backends = {
        "neptune": NeptuneBackend,
        # ... other backends
    }
    return backends.get(name)
```

### 3. Add Configuration Support

```python
# Update tracelet/settings.py to include Neptune settings
from pydantic import BaseSettings

class TraceletSettings(BaseSettings):
    # ... existing settings

    # Neptune-specific settings
    neptune_project: Optional[str] = None
    neptune_api_token: Optional[str] = None
    neptune_mode: str = "async"

    class Config:
        env_prefix = "TRACELET_"
```

### 4. Write Tests

```python
# tests/unit/backends/test_neptune_backend.py
import pytest
from unittest.mock import Mock, patch
from tracelet.backends.neptune_backend import NeptuneBackend
from tracelet.core.orchestrator import MetricData, MetricType

class TestNeptuneBackend:
    @patch('tracelet.backends.neptune_backend.require')
    def test_initialize(self, mock_require):
        """Test Neptune backend initialization."""
        mock_neptune = Mock()
        mock_require.return_value = mock_neptune

        backend = NeptuneBackend()
        config = {
            "project": "workspace/test-project",
            "api_token": "test-token"
        }

        backend.initialize(config)

        mock_neptune.init_run.assert_called_once_with(
            project="workspace/test-project",
            api_token="test-token",
            name=None
        )

    def test_handle_scalar_metric(self):
        """Test handling scalar metrics."""
        backend = NeptuneBackend()
        backend._run = Mock()
        backend._active = True

        metric = MetricData(
            name="accuracy",
            value=0.95,
            type=MetricType.SCALAR,
            iteration=100
        )

        backend.handle_metric(metric)

        backend._run["accuracy"].append.assert_called_once_with(0.95, step=100)
```

## Creating a Framework Plugin

Framework plugins integrate Tracelet with ML frameworks to automatically capture metrics.

### 1. Basic Framework Plugin Structure

```python
# tracelet/frameworks/jax_framework.py
from typing import Any, Dict
from tracelet.core.plugins import PluginBase, PluginMetadata, PluginType
from tracelet.core.orchestrator import MetricData, MetricType, MetricSource

class JAXFramework(PluginBase, MetricSource):
    """JAX framework integration plugin."""

    def __init__(self):
        self._experiment = None
        self._original_functions = {}
        self._patched = False

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="jax",
            version="1.0.0",
            type=PluginType.FRAMEWORK,
            description="JAX framework integration",
            dependencies=["jax", "flax"],
            capabilities={"metric_capture", "parameter_logging"}
        )

    def initialize(self, config: Dict[str, Any]):
        """Initialize JAX framework integration."""
        self._config = config

    def start(self):
        """Start JAX integration by patching functions."""
        if not self._patched:
            self._patch_jax_logging()
            self._patched = True

    def stop(self):
        """Stop JAX integration and restore original functions."""
        if self._patched:
            self._restore_original_functions()
            self._patched = False

    def get_status(self) -> Dict[str, Any]:
        return {"patched": self._patched}

    def get_source_id(self) -> str:
        return "jax_framework"

    def emit_metric(self, metric: MetricData):
        """Emit metric to orchestrator."""
        if self._experiment:
            self._experiment.emit_metric(metric)

    def set_experiment(self, experiment):
        """Set the active experiment."""
        self._experiment = experiment

    def _patch_jax_logging(self):
        """Patch JAX/Flax logging functions."""
        try:
            import flax.training.train_state as train_state

            # Store original function
            self._original_functions['apply_gradients'] = train_state.TrainState.apply_gradients

            def wrapped_apply_gradients(train_state_self, **kwargs):
                # Call original function
                result = self._original_functions['apply_gradients'](train_state_self, **kwargs)

                # Capture metrics
                step = int(train_state_self.step) if hasattr(train_state_self, 'step') else None

                # Emit learning rate if available
                if hasattr(train_state_self, 'opt_state') and hasattr(train_state_self.tx, 'learning_rate'):
                    lr_metric = MetricData(
                        name="learning_rate",
                        value=float(train_state_self.tx.learning_rate),
                        type=MetricType.SCALAR,
                        iteration=step,
                        source=self.get_source_id()
                    )
                    self.emit_metric(lr_metric)

                return result

            # Apply patch
            train_state.TrainState.apply_gradients = wrapped_apply_gradients

        except ImportError:
            # JAX/Flax not available
            pass

    def _restore_original_functions(self):
        """Restore original JAX functions."""
        try:
            import flax.training.train_state as train_state
            if 'apply_gradients' in self._original_functions:
                train_state.TrainState.apply_gradients = self._original_functions['apply_gradients']
        except ImportError:
            pass
```

### 2. Advanced Framework Integration

For more complex integrations, you can hook into training loops:

```python
class AdvancedJAXFramework(JAXFramework):
    """Advanced JAX integration with training loop detection."""

    def _patch_jax_logging(self):
        """Enhanced patching with training loop detection."""
        super()._patch_jax_logging()

        # Patch common JAX training patterns
        self._patch_optax_optimizers()
        self._patch_flax_training()

    def _patch_optax_optimizers(self):
        """Patch Optax optimizers to capture optimization metrics."""
        try:
            import optax

            # Store original update function
            original_update = optax.GradientTransformation.update

            def wrapped_update(tx_self, updates, state, params=None):
                result = original_update(tx_self, updates, state, params)

                # Capture gradient norms
                if updates:
                    grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_leaves(updates)))

                    norm_metric = MetricData(
                        name="gradient_norm",
                        value=float(grad_norm),
                        type=MetricType.SCALAR,
                        source=self.get_source_id()
                    )
                    self.emit_metric(norm_metric)

                return result

            optax.GradientTransformation.update = wrapped_update

        except ImportError:
            pass
```

## Creating a Collector Plugin

Collector plugins gather data from external sources (system metrics, git info, etc.).

### 1. Basic Collector Plugin

```python
# tracelet/collectors/docker_collector.py
import time
from typing import Any, Dict, List
from tracelet.core.plugins import PluginBase, PluginMetadata, PluginType
from tracelet.core.orchestrator import MetricData, MetricType

class DockerCollector(PluginBase):
    """Collects Docker container metrics."""

    def __init__(self):
        self._docker_client = None
        self._container_id = None
        self._collection_interval = 30

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="docker",
            version="1.0.0",
            type=PluginType.COLLECTOR,
            description="Docker container metrics collector",
            dependencies=["docker"],
            capabilities={"system_metrics", "resource_monitoring"}
        )

    def initialize(self, config: Dict[str, Any]):
        """Initialize Docker collector."""
        from docker import DockerClient

        self._config = config
        self._collection_interval = config.get("interval", 30)
        self._docker_client = DockerClient.from_env()

        # Auto-detect current container
        self._container_id = self._detect_current_container()

    def start(self):
        """Start the collector."""
        self._active = True

    def stop(self):
        """Stop the collector."""
        self._active = False

    def get_status(self) -> Dict[str, Any]:
        return {
            "active": self._active,
            "container_id": self._container_id,
            "interval": self._collection_interval
        }

    def collect(self) -> List[MetricData]:
        """Collect Docker container metrics."""
        if not self._active or not self._container_id:
            return []

        try:
            container = self._docker_client.containers.get(self._container_id)
            stats = container.stats(stream=False)

            metrics = []
            timestamp = time.time()

            # CPU metrics
            cpu_percent = self._calculate_cpu_percent(stats)
            metrics.append(MetricData(
                name="docker/cpu_percent",
                value=cpu_percent,
                type=MetricType.SYSTEM,
                timestamp=timestamp,
                source="docker_collector"
            ))

            # Memory metrics
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100

            metrics.append(MetricData(
                name="docker/memory_percent",
                value=memory_percent,
                type=MetricType.SYSTEM,
                timestamp=timestamp,
                source="docker_collector"
            ))

            return metrics

        except Exception as e:
            # Log error and return empty list
            return []

    def _detect_current_container(self) -> str:
        """Auto-detect current container ID."""
        try:
            with open('/proc/self/cgroup', 'r') as f:
                for line in f:
                    if 'docker' in line:
                        return line.split('/')[-1].strip()
        except FileNotFoundError:
            pass
        return None

    def _calculate_cpu_percent(self, stats: Dict) -> float:
        """Calculate CPU percentage from Docker stats."""
        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                   stats['precpu_stats']['cpu_usage']['total_usage']
        system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                      stats['precpu_stats']['system_cpu_usage']

        if system_delta > 0:
            return (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100
        return 0.0
```

## Plugin Registration and Discovery

### 1. Manual Registration

```python
# Register plugins manually in your application
from tracelet.core.experiment import Experiment
from tracelet.backends.neptune_backend import NeptuneBackend
from tracelet.frameworks.jax_framework import JAXFramework

# Create experiment with custom plugins
exp = Experiment(name="custom_experiment")

# Add custom backend
neptune_backend = NeptuneBackend()
exp._plugin_manager.register_plugin(neptune_backend)

# Add framework integration
jax_framework = JAXFramework()
exp._plugin_manager.register_plugin(jax_framework)

exp.start()
```

### 2. Automatic Discovery

```python
# Create plugin entry points in setup.py or pyproject.toml
[project.entry-points."tracelet.plugins"]
neptune = "tracelet.backends.neptune_backend:NeptuneBackend"
jax = "tracelet.frameworks.jax_framework:JAXFramework"
docker = "tracelet.collectors.docker_collector:DockerCollector"
```

## Testing Plugins

### 1. Unit Tests

```python
# tests/unit/test_neptune_backend.py
import pytest
from unittest.mock import Mock, patch
from tracelet.backends.neptune_backend import NeptuneBackend

@pytest.fixture
def mock_neptune():
    with patch('tracelet.backends.neptune_backend.require') as mock_require:
        mock_neptune = Mock()
        mock_require.return_value = mock_neptune
        yield mock_neptune

class TestNeptuneBackend:
    def test_initialization(self, mock_neptune):
        backend = NeptuneBackend()
        config = {"project": "test/project", "api_token": "token"}

        backend.initialize(config)

        mock_neptune.init_run.assert_called_once()

    def test_metric_handling(self, mock_neptune):
        backend = NeptuneBackend()
        backend._run = Mock()
        backend._active = True

        from tracelet.core.orchestrator import MetricData, MetricType
        metric = MetricData("test_metric", 1.0, MetricType.SCALAR)

        backend.handle_metric(metric)

        backend._run["test_metric"].append.assert_called_once_with(1.0, step=None)
```

### 2. Integration Tests

```python
# tests/integration/test_plugin_integration.py
import pytest
from tracelet import Experiment
from tracelet.backends.neptune_backend import NeptuneBackend

@pytest.mark.integration
def test_neptune_integration():
    """Test Neptune backend integration with real Neptune API."""
    # This test requires NEPTUNE_API_TOKEN environment variable

    exp = Experiment(name="integration_test", backend=["neptune"])
    exp.start()

    # Log some metrics
    exp.log_metric("test_accuracy", 0.95, iteration=1)
    exp.log_params({"learning_rate": 0.001, "batch_size": 32})

    exp.stop()

    # Verify metrics were logged to Neptune
    # (Implementation depends on Neptune API for verification)
```

## Best Practices

### 1. Error Handling

```python
def handle_metric(self, metric: MetricData):
    """Handle metric with robust error handling."""
    try:
        # Validate metric
        if not self._validate_metric(metric):
            return

        # Process metric
        self._process_metric(metric)

    except Exception as e:
        # Log error but don't crash
        logger.error(f"Failed to handle metric {metric.name}: {e}")

        # Optionally, emit error metric
        error_metric = MetricData(
            name="tracelet/plugin_errors",
            value=1.0,
            type=MetricType.SYSTEM,
            metadata={"plugin": self.get_metadata().name, "error": str(e)}
        )
        self.emit_metric(error_metric)
```

### 2. Resource Management

```python
class ResourceManagedPlugin(PluginBase):
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def stop(self):
        """Ensure all resources are cleaned up."""
        try:
            # Cleanup network connections
            if hasattr(self, '_client'):
                self._client.close()

            # Stop background threads
            if hasattr(self, '_threads'):
                for thread in self._threads:
                    thread.join(timeout=5.0)

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            self._active = False
```

### 3. Configuration Validation

```python
from pydantic import BaseModel, validator

class NeptuneConfig(BaseModel):
    project: str
    api_token: str
    mode: str = "async"

    @validator('project')
    def validate_project_format(cls, v):
        if '/' not in v:
            raise ValueError('Project must be in format "workspace/project"')
        return v

    @validator('mode')
    def validate_mode(cls, v):
        if v not in ['async', 'sync', 'offline']:
            raise ValueError('Mode must be one of: async, sync, offline')
        return v

class NeptuneBackend(BackendPlugin):
    def initialize(self, config: Dict[str, Any]):
        # Validate configuration
        validated_config = NeptuneConfig(**config)

        # Use validated config
        self._setup_neptune(validated_config)
```

## Publishing Plugins

### 1. Package Structure

```
tracelet-neptune-plugin/
├── pyproject.toml
├── README.md
├── src/
│   └── tracelet_neptune/
│       ├── __init__.py
│       ├── backend.py
│       └── py.typed
├── tests/
│   ├── test_backend.py
│   └── test_integration.py
└── docs/
    └── usage.md
```

### 2. Setup Configuration

```toml
# pyproject.toml
[project]
name = "tracelet-neptune"
version = "1.0.0"
description = "Neptune.ai backend plugin for Tracelet"
dependencies = ["tracelet>=0.1.0", "neptune>=1.0.0"]

[project.entry-points."tracelet.backends"]
neptune = "tracelet_neptune:NeptuneBackend"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 3. Installation

```bash
# Install from PyPI
pip install tracelet-neptune

# Use in code
from tracelet import Experiment
exp = Experiment(name="test", backend=["neptune"])
```

This guide provides the foundation for creating robust, well-tested plugins that extend Tracelet's capabilities. Follow these patterns and best practices to ensure your plugins integrate smoothly with the Tracelet ecosystem.
