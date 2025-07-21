import json
from typing import Optional

import pytest

from tracelet.core.orchestrator import MetricData, MetricType
from tracelet.core.plugins import (
    BackendPlugin,
    CollectorPlugin,
    PluginBase,
    PluginInfo,
    PluginManager,
    PluginMetadata,
    PluginState,
    PluginType,
)


# Test plugin implementations
class MockBackendPlugin(BackendPlugin):
    """Test backend plugin implementation"""

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="test_backend",
            version="1.0.0",
            type=PluginType.BACKEND,
            description="Test backend plugin",
            author="Test Author",
            dependencies=[],
            capabilities={"logging", "storage"}
        )

    def initialize(self, config: dict):
        self._config = config
        self._metrics = []

    def start(self):
        self._active = True

    def stop(self):
        self._active = False

    def get_status(self) -> dict:
        return {
            "active": self._active,
            "metrics_received": len(self._metrics)
        }

    # BackendInterface methods
    def log_metric(self, name: str, value: any, iteration: int):
        self._metrics.append({"name": name, "value": value, "iteration": iteration})

    def log_params(self, params: dict):
        pass

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        pass

    def save_experiment(self, experiment_data: dict):
        pass

    # MetricSink methods
    def receive_metric(self, metric: MetricData):
        self._metrics.append(metric)


class MockCollectorPlugin(CollectorPlugin):
    """Test collector plugin implementation"""

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="test_collector",
            version="1.0.0",
            type=PluginType.COLLECTOR,
            description="Test collector plugin",
            dependencies=["test_backend"]
        )

    def initialize(self, config: dict):
        self._config = config

    def start(self):
        self._active = True

    def stop(self):
        self._active = False

    def get_status(self) -> dict:
        return {"active": self._active}

    def collect(self) -> dict:
        return {"test_metric": 42}


class BrokenPlugin(PluginBase):
    """Plugin that raises errors"""

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        raise ValueError("Broken metadata")


class TestPluginManager:
    """Test suite for PluginManager"""

    def test_initialization(self):
        """Test plugin manager initialization"""
        manager = PluginManager()

        assert isinstance(manager.plugins, dict)
        assert len(manager.plugins) == 0
        assert isinstance(manager.plugin_paths, list)

    def test_discover_plugins_in_file(self, tmp_path):
        """Test plugin discovery in a single file"""
        # Create a plugin file
        plugin_file = tmp_path / "test_plugin.py"
        plugin_file.write_text('''
from tracelet.core.plugins import BackendPlugin, PluginMetadata, PluginType

class TestPlugin(BackendPlugin):
    @classmethod
    def get_metadata(cls):
        return PluginMetadata(
            name="file_test_plugin",
            version="1.0.0",
            type=PluginType.BACKEND
        )

    def initialize(self, config): pass
    def start(self): pass
    def stop(self): pass
    def get_status(self): return {}
    def log_metric(self, name, value, iteration): pass
    def log_params(self, params): pass
    def log_artifact(self, local_path, artifact_path=None): pass
    def save_experiment(self, data): pass
    def receive_metric(self, metric): pass
''')

        manager = PluginManager([str(plugin_file)], use_default_paths=False)
        discovered = manager.discover_plugins()

        assert len(discovered) == 1
        assert discovered[0].metadata.name == "file_test_plugin"
        assert discovered[0].state == PluginState.DISCOVERED

    def test_discover_plugins_in_directory(self, tmp_path):
        """Test plugin discovery in a directory"""
        # Create multiple plugin files
        for i in range(3):
            plugin_file = tmp_path / f"plugin_{i}.py"
            plugin_file.write_text(f'''
from tracelet.core.plugins import BackendPlugin, PluginMetadata, PluginType

class Plugin{i}(BackendPlugin):
    @classmethod
    def get_metadata(cls):
        return PluginMetadata(
            name="plugin_{i}",
            version="1.0.0",
            type=PluginType.BACKEND
        )

    def initialize(self, config): pass
    def start(self): pass
    def stop(self): pass
    def get_status(self): return {{}}
    def log_metric(self, name, value, iteration): pass
    def log_params(self, params): pass
    def log_artifact(self, local_path, artifact_path=None): pass
    def save_experiment(self, data): pass
    def receive_metric(self, metric): pass
''')

        manager = PluginManager([str(tmp_path)], use_default_paths=False)
        discovered = manager.discover_plugins()

        assert len(discovered) == 3
        names = {p.metadata.name for p in discovered}
        assert names == {"plugin_0", "plugin_1", "plugin_2"}

    def test_load_plugin(self, tmp_path):
        """Test plugin loading"""
        # Create a plugin file
        plugin_file = tmp_path / "loadable_plugin.py"
        plugin_file.write_text('''
from tracelet.core.plugins import BackendPlugin, PluginMetadata, PluginType

class LoadablePlugin(BackendPlugin):
    loaded = True

    @classmethod
    def get_metadata(cls):
        return PluginMetadata(
            name="loadable_plugin",
            version="1.0.0",
            type=PluginType.BACKEND
        )

    def initialize(self, config): pass
    def start(self): pass
    def stop(self): pass
    def get_status(self): return {}
    def log_metric(self, name, value, iteration): pass
    def log_params(self, params): pass
    def log_artifact(self, local_path, artifact_path=None): pass
    def save_experiment(self, data): pass
    def receive_metric(self, metric): pass
''')

        manager = PluginManager([str(plugin_file)])
        manager.discover_plugins()

        # Load the plugin
        success = manager.load_plugin("loadable_plugin")
        assert success

        plugin_info = manager.plugins["loadable_plugin"]
        assert plugin_info.state == PluginState.LOADED
        assert plugin_info.instance is not None
        assert hasattr(plugin_info.instance, "loaded")

    def test_validate_plugin(self):
        """Test plugin validation"""
        manager = PluginManager()

        # Manually add a test plugin
        plugin_info = PluginInfo(
            metadata=MockBackendPlugin.get_metadata(),
            module_path="test",
            class_name="MockBackendPlugin",
            instance=MockBackendPlugin
        )
        plugin_info.state = PluginState.LOADED
        manager.plugins["test_backend"] = plugin_info

        # Validate
        success = manager.validate_plugin("test_backend")
        assert success
        assert plugin_info.state == PluginState.VALIDATED

    def test_validate_plugin_missing_methods(self):
        """Test validation fails for incomplete plugin"""
        class IncompletePlugin:  # Not inheriting from PluginBase
            @classmethod
            def get_metadata(cls):
                return PluginMetadata(
                    name="incomplete",
                    version="1.0.0",
                    type=PluginType.BACKEND
                )
            # Missing required methods like initialize, start, stop, get_status

        manager = PluginManager()
        plugin_info = PluginInfo(
            metadata=IncompletePlugin.get_metadata(),
            module_path="test",
            class_name="IncompletePlugin",
            instance=IncompletePlugin
        )
        plugin_info.state = PluginState.LOADED
        manager.plugins["incomplete"] = plugin_info

        success = manager.validate_plugin("incomplete")
        assert not success
        assert plugin_info.state == PluginState.ERROR

    def test_initialize_plugin(self):
        """Test plugin initialization"""
        manager = PluginManager()

        # Add test plugin
        plugin_info = PluginInfo(
            metadata=MockBackendPlugin.get_metadata(),
            module_path="test",
            class_name="MockBackendPlugin",
            instance=MockBackendPlugin
        )
        plugin_info.state = PluginState.VALIDATED
        manager.plugins["test_backend"] = plugin_info

        # Initialize with config
        config = {"option1": "value1", "option2": 42}
        success = manager.initialize_plugin("test_backend", config)

        assert success
        assert plugin_info.state == PluginState.INITIALIZED
        assert "test_backend" in manager._plugin_instances

        instance = manager._plugin_instances["test_backend"]
        assert instance._config == config

    def test_plugin_lifecycle(self):
        """Test complete plugin lifecycle"""
        manager = PluginManager()

        # Add test plugin
        plugin_info = PluginInfo(
            metadata=MockBackendPlugin.get_metadata(),
            module_path="test",
            class_name="MockBackendPlugin",
            instance=MockBackendPlugin
        )
        plugin_info.state = PluginState.VALIDATED
        manager.plugins["test_backend"] = plugin_info

        # Initialize
        assert manager.initialize_plugin("test_backend", {})

        # Start
        assert manager.start_plugin("test_backend")
        assert plugin_info.state == PluginState.ACTIVE

        instance = manager.get_plugin_instance("test_backend")
        assert instance._active

        # Stop
        assert manager.stop_plugin("test_backend")
        assert plugin_info.state == PluginState.STOPPED
        assert not instance._active

    def test_dependency_resolution(self):
        """Test plugin dependency resolution"""
        manager = PluginManager()

        # Create plugins with dependencies
        # A depends on B, B depends on C
        for name, deps in [("A", ["B"]), ("B", ["C"]), ("C", [])]:
            metadata = PluginMetadata(
                name=name,
                version="1.0.0",
                type=PluginType.BACKEND,
                dependencies=deps
            )
            manager.plugins[name] = PluginInfo(
                metadata=metadata,
                module_path="test",
                class_name=f"Plugin{name}"
            )

        # Resolve dependencies
        order = manager.resolve_dependencies(["A", "B", "C"])
        assert order == ["C", "B", "A"]

    def test_circular_dependency_detection(self):
        """Test circular dependency detection"""
        manager = PluginManager()

        # Create circular dependency: A -> B -> C -> A
        for name, deps in [("A", ["B"]), ("B", ["C"]), ("C", ["A"])]:
            metadata = PluginMetadata(
                name=name,
                version="1.0.0",
                type=PluginType.BACKEND,
                dependencies=deps
            )
            manager.plugins[name] = PluginInfo(
                metadata=metadata,
                module_path="test",
                class_name=f"Plugin{name}"
            )

        # Should raise error for circular dependency
        with pytest.raises(ValueError, match="Circular dependency"):
            manager.resolve_dependencies(["A", "B", "C"])

    def test_get_plugins_by_type(self):
        """Test filtering plugins by type"""
        manager = PluginManager()

        # Add different types of plugins
        backend_metadata = PluginMetadata(
            name="backend1",
            version="1.0.0",
            type=PluginType.BACKEND
        )
        collector_metadata = PluginMetadata(
            name="collector1",
            version="1.0.0",
            type=PluginType.COLLECTOR
        )

        manager.plugins["backend1"] = PluginInfo(
            metadata=backend_metadata,
            module_path="test",
            class_name="Backend1"
        )
        manager.plugins["collector1"] = PluginInfo(
            metadata=collector_metadata,
            module_path="test",
            class_name="Collector1"
        )

        # Get by type
        backends = manager.get_plugins_by_type(PluginType.BACKEND)
        collectors = manager.get_plugins_by_type(PluginType.COLLECTOR)

        assert len(backends) == 1
        assert backends[0].metadata.name == "backend1"
        assert len(collectors) == 1
        assert collectors[0].metadata.name == "collector1"

    def test_load_config(self, tmp_path):
        """Test loading configuration from file"""
        manager = PluginManager()

        # Create config file
        config_file = tmp_path / "config.json"
        config_data = {
            "plugins": {
                "backend1": {"host": "localhost", "port": 8080},
                "collector1": {"interval": 30}
            }
        }
        config_file.write_text(json.dumps(config_data))

        # Load config
        loaded_config = manager.load_config(str(config_file))
        assert loaded_config == config_data

    def test_get_status(self):
        """Test getting plugin manager status"""
        manager = PluginManager()

        # Add plugins in different states
        for i, state in enumerate([PluginState.DISCOVERED, PluginState.ACTIVE, PluginState.ERROR]):
            metadata = PluginMetadata(
                name=f"plugin_{i}",
                version="1.0.0",
                type=PluginType.BACKEND
            )
            plugin_info = PluginInfo(
                metadata=metadata,
                module_path="test",
                class_name=f"Plugin{i}",
                state=state
            )
            if state == PluginState.ERROR:
                plugin_info.error = "Test error"
            manager.plugins[f"plugin_{i}"] = plugin_info

        # Get status
        status = manager.get_status()

        assert status["discovered"] == 3
        assert status["loaded"] == 2  # ACTIVE and ERROR are >= LOADED
        assert status["active"] == 1
        assert status["errors"] == 1
        assert len(status["plugins"]) == 3
        assert status["plugins"]["plugin_2"]["error"] == "Test error"

    def test_backend_plugin_integration(self):
        """Test BackendPlugin MetricSink integration"""
        plugin = MockBackendPlugin()

        # Test sink ID
        assert plugin.get_sink_id() == "backend_test_backend"

        # Test metric handling
        assert plugin.can_handle_type(MetricType.SCALAR)
        assert plugin.can_handle_type(MetricType.ARTIFACT)

        # Test receiving metrics
        plugin.initialize({})
        plugin.start()

        metric = MetricData(
            name="test_metric",
            value=42,
            type=MetricType.SCALAR
        )
        plugin.receive_metric(metric)

        status = plugin.get_status()
        assert status["metrics_received"] == 1
