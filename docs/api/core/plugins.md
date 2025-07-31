# Plugin System

::: tracelet.core.plugins
options:
show_source: true
show_bases: true
heading_level: 2

## Overview

Tracelet's plugin system provides a flexible architecture for extending functionality through modular components. The system supports backend plugins, collector plugins, and framework integrations.

## Core Plugin Architecture

### Plugin Base Classes

#### PluginBase

::: tracelet.core.plugins.PluginBase
options:
show_source: true
heading_level: 4

#### BackendPlugin

::: tracelet.core.plugins.BackendPlugin
options:
show_source: true
heading_level: 4

#### CollectorPlugin

::: tracelet.core.plugins.CollectorPlugin
options:
show_source: true
heading_level: 4

### Plugin Metadata

::: tracelet.core.plugins.PluginMetadata
options:
show_source: true
heading_level: 3

**Example Usage:**

```python
from tracelet.core.plugins import PluginMetadata

# Define plugin metadata
metadata = PluginMetadata(
    name="custom_backend",
    version="1.0.0",
    description="Custom experiment tracking backend",
    author="ML Team",
    plugin_type="backend",
    entry_point="my_package.backends.CustomBackend",
    dependencies=["requests>=2.25.0", "pandas>=1.3.0"],
    config_schema={
        "api_endpoint": {"type": "string", "required": True},
        "api_key": {"type": "string", "required": True},
        "timeout": {"type": "number", "default": 30}
    }
)
```

## Plugin Manager

::: tracelet.core.plugins.PluginManager
options:
show_source: true
heading_level: 3

### Plugin Discovery and Registration

```python
from tracelet.core.plugins import PluginManager

# Initialize plugin manager
plugin_manager = PluginManager()

# Discover installed plugins
plugin_manager.discover_plugins()

# List available plugins
backend_plugins = plugin_manager.list_plugins("backend")
collector_plugins = plugin_manager.list_plugins("collector")

print(f"Available backend plugins: {[p.name for p in backend_plugins]}")
print(f"Available collector plugins: {[p.name for p in collector_plugins]}")
```

### Manual Plugin Registration

```python
from my_package.plugins import CustomBackendPlugin

# Register plugin manually
plugin_manager.register_plugin(CustomBackendPlugin)

# Get specific plugin
plugin_class = plugin_manager.get_plugin("backend", "custom_backend")
if plugin_class:
    plugin_instance = plugin_class()
```

## Creating Custom Plugins

### Backend Plugin Example

```python
from tracelet.core.plugins import BackendPlugin, PluginMetadata
from typing import Dict, Any, Optional

class CustomBackendPlugin(BackendPlugin):
    """Custom backend for proprietary experiment tracking system."""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="custom_backend",
            version="1.0.0",
            description="Integration with Custom Tracking System",
            author="Your Team",
            plugin_type="backend",
            entry_point="custom_package.CustomBackendPlugin",
            dependencies=["custom-sdk>=2.0.0"],
            config_schema={
                "server_url": {"type": "string", "required": True},
                "project_id": {"type": "string", "required": True},
                "api_token": {"type": "string", "required": True}
            }
        )

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the backend with configuration."""
        self.server_url = config["server_url"]
        self.project_id = config["project_id"]
        self.api_token = config["api_token"]

        # Initialize SDK
        import custom_sdk
        self.client = custom_sdk.Client(
            url=self.server_url,
            token=self.api_token
        )

        # Create or get project
        self.project = self.client.get_project(self.project_id)
        self.experiment = None

    def start(self) -> None:
        """Start experiment tracking."""
        self.experiment = self.project.create_experiment(
            name=self.config.get("experiment_name", "Default Experiment")
        )
        self.experiment.start()

    def stop(self) -> None:
        """Stop experiment tracking."""
        if self.experiment:
            self.experiment.finish()
            self.experiment = None

    def log_metric(self, name: str, value: float, iteration: Optional[int] = None) -> None:
        """Log a metric value."""
        if self.experiment:
            self.experiment.log_metric(
                name=name,
                value=value,
                step=iteration
            )

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        if self.experiment:
            self.experiment.log_params(params)

    def log_artifact(self, local_path: str, artifact_path: str) -> None:
        """Log an artifact."""
        if self.experiment:
            self.experiment.upload_artifact(local_path, artifact_path)

    def get_status(self) -> Dict[str, Any]:
        """Get backend status."""
        return {
            "backend_name": "custom_backend",
            "connected": self.experiment is not None,
            "experiment_id": getattr(self.experiment, "id", None),
            "project_id": self.project_id
        }
```

### Collector Plugin Example

```python
from tracelet.core.plugins import CollectorPlugin, PluginMetadata
import psutil
import time
from typing import Dict, Any

class AdvancedSystemCollectorPlugin(CollectorPlugin):
    """Advanced system metrics collector with custom metrics."""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="advanced_system_collector",
            version="1.2.0",
            description="Advanced system metrics with process-level details",
            author="System Team",
            plugin_type="collector",
            entry_point="system_package.AdvancedSystemCollectorPlugin",
            dependencies=["psutil>=5.8.0", "GPUtil>=1.4.0"],
            config_schema={
                "include_processes": {"type": "boolean", "default": False},
                "process_limit": {"type": "number", "default": 10},
                "include_network": {"type": "boolean", "default": True}
            }
        )

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize collector with configuration."""
        self.include_processes = config.get("include_processes", False)
        self.process_limit = config.get("process_limit", 10)
        self.include_network = config.get("include_network", True)
        self.last_network = None

    def start(self) -> None:
        """Start collection (if background collection needed)."""
        self.start_time = time.time()
        if self.include_network:
            self.last_network = psutil.net_io_counters()

    def stop(self) -> None:
        """Stop collection."""
        pass

    def collect(self) -> Dict[str, Any]:
        """Collect system metrics."""
        metrics = {
            "timestamp": time.time(),
            "uptime": time.time() - self.start_time
        }

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        metrics.update({
            "cpu_percent": cpu_percent,
            "cpu_count_logical": cpu_count,
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
        })

        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        metrics.update({
            "memory_total": memory.total,
            "memory_available": memory.available,
            "memory_percent": memory.percent,
            "swap_total": swap.total,
            "swap_used": swap.used,
            "swap_percent": swap.percent
        })

        # Disk metrics
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        metrics.update({
            "disk_total": disk_usage.total,
            "disk_used": disk_usage.used,
            "disk_percent": disk_usage.percent,
            "disk_read_bytes": disk_io.read_bytes if disk_io else 0,
            "disk_write_bytes": disk_io.write_bytes if disk_io else 0
        })

        # Network metrics (with rate calculation)
        if self.include_network:
            current_network = psutil.net_io_counters()
            if self.last_network:
                time_delta = 1.0  # Approximate
                metrics.update({
                    "network_bytes_sent": current_network.bytes_sent,
                    "network_bytes_recv": current_network.bytes_recv,
                    "network_bytes_sent_rate": (current_network.bytes_sent - self.last_network.bytes_sent) / time_delta,
                    "network_bytes_recv_rate": (current_network.bytes_recv - self.last_network.bytes_recv) / time_delta
                })
            self.last_network = current_network

        # Process-level metrics
        if self.include_processes:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # Sort by CPU usage and take top N
            processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
            metrics["top_processes"] = processes[:self.process_limit]

        # GPU metrics (if available)
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            gpu_metrics = []
            for gpu in gpus:
                gpu_metrics.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "load": gpu.load * 100,
                    "memory_util": gpu.memoryUtil * 100,
                    "temperature": gpu.temperature
                })
            metrics["gpu"] = gpu_metrics
        except ImportError:
            pass

        return metrics

    def get_status(self) -> Dict[str, Any]:
        """Get collector status."""
        return {
            "collector_name": "advanced_system_collector",
            "active": True,
            "config": {
                "include_processes": self.include_processes,
                "include_network": self.include_network
            }
        }
```

## Plugin Installation and Packaging

### setuptools Entry Points

```python
# setup.py for your plugin package
from setuptools import setup, find_packages

setup(
    name="tracelet-custom-plugins",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "tracelet>=1.0.0",
        "psutil>=5.8.0"
    ],
    entry_points={
        "tracelet.plugins": [
            "custom_backend = my_package.backends:CustomBackendPlugin",
            "advanced_system = my_package.collectors:AdvancedSystemCollectorPlugin"
        ]
    }
)
```

### Plugin Discovery

```python
# Tracelet automatically discovers plugins via entry points
import pkg_resources

def discover_plugins():
    """Discover installed Tracelet plugins."""
    plugins = {}

    for entry_point in pkg_resources.iter_entry_points("tracelet.plugins"):
        try:
            plugin_class = entry_point.load()
            plugin_instance = plugin_class()
            metadata = plugin_instance.get_metadata()
            plugins[metadata.name] = {
                "class": plugin_class,
                "metadata": metadata,
                "entry_point": entry_point
            }
        except Exception as e:
            print(f"Failed to load plugin {entry_point.name}: {e}")

    return plugins
```

## Advanced Plugin Features

### Plugin Dependencies

```python
from tracelet.core.plugins import PluginManager
import importlib

def check_plugin_dependencies(plugin_metadata):
    """Check if plugin dependencies are satisfied."""
    missing_deps = []

    for dep in plugin_metadata.dependencies:
        try:
            # Parse dependency specification (name>=version)
            if ">=" in dep:
                package_name = dep.split(">=")[0]
            else:
                package_name = dep

            importlib.import_module(package_name)
        except ImportError:
            missing_deps.append(dep)

    return missing_deps

# Usage
plugin_manager = PluginManager()
plugins = plugin_manager.discover_plugins()

for plugin_name, plugin_info in plugins.items():
    missing = check_plugin_dependencies(plugin_info["metadata"])
    if missing:
        print(f"Plugin {plugin_name} missing dependencies: {missing}")
```

### Configuration Validation

```python
import jsonschema
from typing import Dict, Any

def validate_plugin_config(config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Validate plugin configuration against schema."""
    try:
        jsonschema.validate(config, schema)
        return True
    except jsonschema.ValidationError as e:
        print(f"Configuration validation failed: {e}")
        return False

# Example usage in plugin
class ValidatedBackendPlugin(BackendPlugin):
    def initialize(self, config: Dict[str, Any]) -> None:
        metadata = self.get_metadata()

        if not validate_plugin_config(config, metadata.config_schema):
            raise ValueError("Invalid plugin configuration")

        # Proceed with initialization
        super().initialize(config)
```

### Plugin Lifecycle Management

```python
from tracelet.core.plugins import PluginManager
from typing import List

class PluginLifecycleManager:
    """Manages plugin lifecycle across experiment sessions."""

    def __init__(self):
        self.plugin_manager = PluginManager()
        self.active_plugins: List[PluginBase] = []

    def start_plugins(self, plugin_configs: Dict[str, Dict[str, Any]]):
        """Start configured plugins."""
        for plugin_name, config in plugin_configs.items():
            try:
                plugin_class = self.plugin_manager.get_plugin_by_name(plugin_name)
                if plugin_class:
                    plugin_instance = plugin_class()
                    plugin_instance.initialize(config)
                    plugin_instance.start()
                    self.active_plugins.append(plugin_instance)
            except Exception as e:
                print(f"Failed to start plugin {plugin_name}: {e}")

    def stop_all_plugins(self):
        """Stop all active plugins."""
        for plugin in self.active_plugins:
            try:
                plugin.stop()
            except Exception as e:
                print(f"Error stopping plugin: {e}")

        self.active_plugins.clear()

    def get_plugin_status(self) -> Dict[str, Any]:
        """Get status of all active plugins."""
        status = {}
        for plugin in self.active_plugins:
            try:
                plugin_status = plugin.get_status()
                status[plugin_status.get("name", "unknown")] = plugin_status
            except Exception as e:
                status["error"] = str(e)

        return status
```

## Best Practices

### Plugin Development

1. **Follow Interface Contracts**: Implement all required methods from base classes
2. **Error Handling**: Handle errors gracefully and provide meaningful error messages
3. **Configuration Validation**: Validate configuration early and provide clear feedback
4. **Resource Management**: Properly clean up resources in `stop()` method
5. **Documentation**: Provide clear documentation and examples

### Plugin Usage

1. **Dependency Management**: Check plugin dependencies before use
2. **Configuration Security**: Don't expose sensitive information in configuration
3. **Performance**: Monitor plugin performance impact on main application
4. **Testing**: Test plugins in isolation and integration scenarios
5. **Version Compatibility**: Ensure plugin versions are compatible with Tracelet core
