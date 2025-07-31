import importlib
import importlib.util
import inspect
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Optional

from .interfaces import BackendInterface, CollectorInterface
from .orchestrator import MetricSink, MetricType

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Types of plugins supported by the system"""

    BACKEND = "backend"
    COLLECTOR = "collector"
    FRAMEWORK = "framework"
    PROCESSOR = "processor"


class PluginState(IntEnum):
    """Plugin lifecycle states"""

    DISCOVERED = 1
    LOADED = 2
    VALIDATED = 3
    INITIALIZED = 4
    ACTIVE = 5
    ERROR = 6
    STOPPED = 7


@dataclass
class PluginMetadata:
    """Metadata for a plugin"""

    name: str
    version: str
    type: PluginType
    description: str = ""
    author: str = ""
    dependencies: list[str] = field(default_factory=list)
    config_schema: Optional[dict[str, Any]] = None
    capabilities: set[str] = field(default_factory=set)


@dataclass
class PluginInfo:
    """Information about a discovered plugin"""

    metadata: PluginMetadata
    module_path: str
    class_name: str
    state: PluginState = PluginState.DISCOVERED
    instance: Optional[Any] = None
    error: Optional[str] = None


class PluginBase(ABC):
    """Base class for all plugins"""

    @classmethod
    @abstractmethod
    def get_metadata(cls) -> PluginMetadata:
        """Return plugin metadata"""
        pass

    @abstractmethod
    def initialize(self, config: dict[str, Any]):
        """Initialize the plugin with configuration"""
        pass

    @abstractmethod
    def start(self):
        """Start the plugin"""
        pass

    @abstractmethod
    def stop(self):
        """Stop the plugin"""
        pass

    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        """Get plugin status"""
        pass


class BackendPlugin(PluginBase, BackendInterface, MetricSink):
    """Base class for backend plugins"""

    def __init__(self):
        self._config = {}
        self._active = False

    def get_sink_id(self) -> str:
        """Return sink ID for orchestrator integration"""
        return f"backend_{self.get_metadata().name}"

    def can_handle_type(self, metric_type: MetricType) -> bool:
        """Check if this backend can handle the metric type"""
        # By default, backends handle all metric types
        return True


class CollectorPlugin(PluginBase, CollectorInterface):
    """Base class for collector plugins"""

    def __init__(self):
        self._config = {}
        self._active = False
        self._collection_interval = 60  # seconds


class PluginManager:
    """Manages plugin discovery, loading, and lifecycle"""

    def __init__(self, plugin_paths: Optional[list[str]] = None, use_default_paths: bool = True):
        self.plugin_paths = plugin_paths or []
        self.plugins: dict[str, PluginInfo] = {}
        self._plugin_instances: dict[str, PluginBase] = {}
        self._dependency_graph: dict[str, set[str]] = {}

        # Add default plugin paths unless explicitly disabled
        if use_default_paths:
            self._add_default_paths()

    def _add_default_paths(self):
        """Add default plugin search paths"""
        # Built-in plugins directory
        builtin_path = Path(__file__).parent.parent / "plugins"
        if builtin_path.exists():
            self.plugin_paths.append(str(builtin_path))

        # Built-in backends directory (where backends actually are)
        backends_path = Path(__file__).parent.parent / "backends"
        if backends_path.exists():
            self.plugin_paths.append(str(backends_path))

        # User plugins directory
        user_path = Path.home() / ".tracelet" / "plugins"
        if user_path.exists():
            self.plugin_paths.append(str(user_path))

        # Environment variable paths
        env_paths = os.environ.get("TRACELET_PLUGIN_PATH", "").split(":")
        self.plugin_paths.extend([p for p in env_paths if p])

    def discover_plugins(self) -> list[PluginInfo]:
        """Discover all available plugins"""
        discovered = []

        for path in self.plugin_paths:
            if os.path.isdir(path):
                discovered.extend(self._discover_in_directory(path))
            elif path.endswith(".py"):
                plugin = self._discover_in_file(path)
                if plugin:
                    discovered.append(plugin)

        # Update internal registry
        for plugin in discovered:
            self.plugins[plugin.metadata.name] = plugin
            logger.info(f"Discovered plugin: {plugin.metadata.name} v{plugin.metadata.version}")

        return discovered

    def _discover_in_directory(self, directory: str) -> list[PluginInfo]:
        """Discover plugins in a directory"""
        discovered = []

        for filename in os.listdir(directory):
            if filename.endswith(".py") and not filename.startswith("_"):
                filepath = os.path.join(directory, filename)
                plugin = self._discover_in_file(filepath)
                if plugin:
                    discovered.append(plugin)

        return discovered

    def _discover_in_file(self, filepath: str) -> Optional[PluginInfo]:
        """Discover plugin in a single file"""
        try:
            # Load module spec
            spec = importlib.util.spec_from_file_location("plugin_module", filepath)
            if not spec or not spec.loader:
                return None

            # Load module
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find plugin classes
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, PluginBase)
                    and obj not in [PluginBase, BackendPlugin, CollectorPlugin]
                ):
                    try:
                        metadata = obj.get_metadata()
                        return PluginInfo(
                            metadata=metadata, module_path=filepath, class_name=name, state=PluginState.DISCOVERED
                        )
                    except Exception:
                        logger.exception(f"Failed to get metadata from {name}")

        except Exception:
            logger.exception(f"Failed to discover plugin in {filepath}")

        return None

    def load_plugin(self, plugin_name: str) -> bool:
        """Load a specific plugin"""
        if plugin_name not in self.plugins:
            logger.error(f"Plugin '{plugin_name}' not found")
            return False

        plugin_info = self.plugins[plugin_name]

        if plugin_info.state >= PluginState.LOADED:
            return True

        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(f"plugin_{plugin_name}", plugin_info.module_path)
            if not spec or not spec.loader:
                raise ImportError(f"Failed to load spec for {plugin_name}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get the plugin class
            plugin_class = getattr(module, plugin_info.class_name)

            # Store class reference
            plugin_info.instance = plugin_class
            plugin_info.state = PluginState.LOADED

            logger.info(f"Loaded plugin: {plugin_name}")
            return True

        except Exception as e:
            plugin_info.state = PluginState.ERROR
            plugin_info.error = str(e)
            logger.exception(f"Failed to load plugin '{plugin_name}'")
            return False

    def validate_plugin(self, plugin_name: str) -> bool:
        """Validate a plugin"""
        if plugin_name not in self.plugins:
            return False

        plugin_info = self.plugins[plugin_name]

        if plugin_info.state < PluginState.LOADED and not self.load_plugin(plugin_name):
            return False

        try:
            # Check required methods
            required_methods = ["get_metadata", "initialize", "start", "stop", "get_status"]
            plugin_class = plugin_info.instance

            for method in required_methods:
                if not hasattr(plugin_class, method):
                    raise ValueError(f"Missing required method: {method}")

            # Validate metadata
            metadata = plugin_class.get_metadata()
            if not metadata.name or not metadata.version:
                raise ValueError("Invalid metadata: name and version required")

            # Check dependencies
            for dep in metadata.dependencies:
                if dep not in self.plugins:
                    raise ValueError(f"Missing dependency: {dep}")

            plugin_info.state = PluginState.VALIDATED
            logger.info(f"Validated plugin: {plugin_name}")
            return True

        except Exception as e:
            plugin_info.state = PluginState.ERROR
            plugin_info.error = str(e)
            logger.exception(f"Plugin validation failed for '{plugin_name}': {e}")
            return False

    def initialize_plugin(self, plugin_name: str, config: Optional[dict[str, Any]] = None) -> bool:
        """Initialize a plugin with configuration"""
        if plugin_name not in self.plugins:
            return False

        plugin_info = self.plugins[plugin_name]

        # Ensure plugin is validated
        if plugin_info.state < PluginState.VALIDATED and not self.validate_plugin(plugin_name):
            return False

        try:
            # Create plugin instance
            plugin_class = plugin_info.instance
            instance = plugin_class()

            # Initialize with config
            instance.initialize(config or {})

            # Store instance
            self._plugin_instances[plugin_name] = instance
            plugin_info.state = PluginState.INITIALIZED

            logger.info(f"Initialized plugin: {plugin_name}")
            return True

        except Exception as e:
            plugin_info.state = PluginState.ERROR
            plugin_info.error = str(e)
            logger.exception(f"Failed to initialize plugin '{plugin_name}': {e}")
            return False

    def start_plugin(self, plugin_name: str) -> bool:
        """Start a plugin"""
        if plugin_name not in self._plugin_instances:
            logger.error(f"Plugin '{plugin_name}' not initialized")
            return False

        plugin_info = self.plugins[plugin_name]
        instance = self._plugin_instances[plugin_name]

        try:
            instance.start()
            plugin_info.state = PluginState.ACTIVE
            logger.info(f"Started plugin: {plugin_name}")
            return True

        except Exception as e:
            plugin_info.state = PluginState.ERROR
            plugin_info.error = str(e)
            logger.exception(f"Failed to start plugin '{plugin_name}': {e}")
            return False

    def stop_plugin(self, plugin_name: str) -> bool:
        """Stop a plugin"""
        if plugin_name not in self._plugin_instances:
            return True

        plugin_info = self.plugins[plugin_name]
        instance = self._plugin_instances[plugin_name]

        try:
            instance.stop()
            plugin_info.state = PluginState.STOPPED
            logger.info(f"Stopped plugin: {plugin_name}")
            return True

        except Exception as e:
            logger.exception(f"Error stopping plugin '{plugin_name}': {e}")
            return False

    def get_plugin_instance(self, plugin_name: str) -> Optional[PluginBase]:
        """Get an initialized plugin instance"""
        return self._plugin_instances.get(plugin_name)

    def get_plugins_by_type(self, plugin_type: PluginType) -> list[PluginInfo]:
        """Get all plugins of a specific type"""
        return [info for info in self.plugins.values() if info.metadata.type == plugin_type]

    def resolve_dependencies(self, plugin_names: list[str]) -> list[str]:
        """Resolve plugin dependencies and return load order"""
        # Build dependency graph
        graph = {}
        for name in plugin_names:
            if name in self.plugins:
                deps = self.plugins[name].metadata.dependencies
                graph[name] = set(deps)

        # Topological sort
        result = []
        visited = set()
        temp_visited = set()

        def visit(node):
            if node in temp_visited:
                raise ValueError(f"Circular dependency detected: {node}")
            if node in visited:
                return

            temp_visited.add(node)
            for dep in graph.get(node, set()):
                visit(dep)
            temp_visited.remove(node)
            visited.add(node)
            result.append(node)

        for name in plugin_names:
            if name not in visited:
                visit(name)

        return result

    def load_config(self, config_path: str) -> dict[str, Any]:
        """Load plugin configuration from file"""
        with open(config_path) as f:
            return json.load(f)

    def get_status(self) -> dict[str, Any]:
        """Get status of all plugins"""
        status = {
            "discovered": len(self.plugins),
            "loaded": sum(1 for p in self.plugins.values() if p.state >= PluginState.LOADED),
            "active": sum(1 for p in self.plugins.values() if p.state == PluginState.ACTIVE),
            "errors": sum(1 for p in self.plugins.values() if p.state == PluginState.ERROR),
            "plugins": {},
        }

        for name, info in self.plugins.items():
            plugin_status = {
                "state": info.state.name.lower(),
                "type": info.metadata.type.value,
                "version": info.metadata.version,
            }

            if info.state == PluginState.ERROR:
                plugin_status["error"] = info.error

            if name in self._plugin_instances:
                try:
                    plugin_status["status"] = self._plugin_instances[name].get_status()
                except Exception as e:
                    plugin_status["status_error"] = str(e)

            status["plugins"][name] = plugin_status

        return status
