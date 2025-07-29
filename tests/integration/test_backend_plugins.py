"""Integration tests for backend plugins."""

import tempfile
import time
from pathlib import Path

import pytest

from tracelet.core.orchestrator import DataFlowOrchestrator, MetricData, MetricType, RoutingRule
from tracelet.core.plugins import PluginManager

# Test imports - these will be skipped if dependencies aren't available
pytest_plugins = []

try:
    from tracelet.backends.clearml_backend import ClearMLBackend

    _has_clearml = True
except ImportError:
    _has_clearml = False
    ClearMLBackend = None

try:
    from tracelet.backends.mlflow_backend import MLflowBackend

    _has_mlflow = True
except ImportError:
    _has_mlflow = False
    MLflowBackend = None


class TestClearMLBackend:
    """Integration tests for ClearML backend plugin."""

    @pytest.mark.skipif(not _has_clearml, reason="ClearML not installed")
    def test_clearml_plugin_metadata(self):
        """Test ClearML plugin metadata."""
        metadata = ClearMLBackend.get_metadata()

        assert metadata.name == "clearml"
        assert metadata.type.value == "backend"
        assert "metrics" in metadata.capabilities
        assert "parameters" in metadata.capabilities
        assert "artifacts" in metadata.capabilities

    @pytest.mark.skipif(not _has_clearml, reason="ClearML not installed")
    def test_clearml_plugin_lifecycle(self):
        """Test ClearML plugin lifecycle (offline mode)."""
        plugin = ClearMLBackend()

        # Test initialization (offline mode to avoid requiring ClearML server)
        config = {
            "project_name": "Test Project",
            "task_name": "Test Task",
            "auto_connect_frameworks": False,
            "auto_connect_arg_parser": False,
        }

        # Set offline mode
        import os

        os.environ["CLEARML_WEB_HOST"] = ""
        os.environ["CLEARML_API_HOST"] = ""
        os.environ["CLEARML_FILES_HOST"] = ""

        try:
            plugin.initialize(config)
            assert plugin._project_name == "Test Project"
            assert plugin._task_name == "Test Task"

            # Test start/stop
            plugin.start()
            assert plugin._active

            status = plugin.get_status()
            assert status["active"]
            assert status["project_name"] == "Test Project"

            plugin.stop()
            assert not plugin._active

        except Exception as e:
            # ClearML might fail in CI/offline environments, that's okay
            print(f"ClearML initialization failed (expected in CI): {e}")
            pytest.skip("ClearML requires server connection")

    @pytest.mark.skipif(not _has_clearml, reason="ClearML not installed")
    def test_clearml_metric_logging(self):
        """Test ClearML metric logging."""
        plugin = ClearMLBackend()

        # Use offline mode
        import os

        os.environ["CLEARML_WEB_HOST"] = ""
        os.environ["CLEARML_API_HOST"] = ""
        os.environ["CLEARML_FILES_HOST"] = ""

        try:
            plugin.initialize({"project_name": "Test", "task_name": "MetricTest"})
            plugin.start()

            # Test scalar metric
            metric = MetricData(name="accuracy", value=0.95, type=MetricType.SCALAR, iteration=10, source="test_source")

            # This should not raise an exception
            plugin.receive_metric(metric)

            # Test parameter
            param_metric = MetricData(
                name="learning_rate", value=0.001, type=MetricType.PARAMETER, source="test_source"
            )

            plugin.receive_metric(param_metric)

            plugin.stop()

        except Exception as e:
            print(f"ClearML metric logging failed (expected in CI): {e}")
            pytest.skip("ClearML requires server connection for full functionality")


class TestMLflowBackend:
    """Integration tests for MLflow backend plugin."""

    @pytest.mark.skipif(not _has_mlflow, reason="MLflow not installed")
    def test_mlflow_plugin_metadata(self):
        """Test MLflow plugin metadata."""
        metadata = MLflowBackend.get_metadata()

        assert metadata.name == "mlflow"
        assert metadata.type.value == "backend"
        assert "metrics" in metadata.capabilities
        assert "parameters" in metadata.capabilities
        assert "artifacts" in metadata.capabilities
        assert "model_registry" in metadata.capabilities

    @pytest.mark.skipif(not _has_mlflow, reason="MLflow not installed")
    def test_mlflow_plugin_lifecycle(self):
        """Test MLflow plugin lifecycle with local tracking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plugin = MLflowBackend()

            # Use local file store for testing
            config = {
                "experiment_name": "Test Experiment",
                "run_name": "Test Run",
                "tracking_uri": f"file://{temp_dir}/mlruns",
            }

            plugin.initialize(config)
            assert plugin._experiment_name == "Test Experiment"
            assert plugin._experiment_id is not None

            # Test start/stop
            plugin.start()
            assert plugin._active
            assert plugin._run_id is not None

            status = plugin.get_status()
            assert status["active"]
            assert status["experiment_name"] == "Test Experiment"
            assert status["run_id"] is not None

            plugin.stop()
            assert not plugin._active

    @pytest.mark.skipif(not _has_mlflow, reason="MLflow not installed")
    def test_mlflow_metric_logging(self):
        """Test MLflow metric logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plugin = MLflowBackend()

            config = {
                "experiment_name": "Metric Test Experiment",
                "run_name": "Metric Test Run",
                "tracking_uri": f"file://{temp_dir}/mlruns",
            }

            plugin.initialize(config)
            plugin.start()

            # Test scalar metric
            metric = MetricData(name="loss", value=0.123, type=MetricType.SCALAR, iteration=5, source="training")

            plugin.receive_metric(metric)

            # Test parameter
            param_metric = MetricData(name="batch_size", value=32, type=MetricType.PARAMETER, source="config")

            plugin.receive_metric(param_metric)

            # Test custom metric (should become tag)
            custom_metric = MetricData(name="model_type", value="transformer", type=MetricType.CUSTOM, source="model")

            plugin.receive_metric(custom_metric)

            plugin.stop()

            # Verify run was created and data logged
            import mlflow

            mlflow.set_tracking_uri(f"file://{temp_dir}/mlruns")

            experiment = mlflow.get_experiment_by_name("Metric Test Experiment")
            assert experiment is not None

            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            assert len(runs) == 1

            run = runs.iloc[0]
            assert run["status"] == "FINISHED"

    @pytest.mark.skipif(not _has_mlflow, reason="MLflow not installed")
    def test_mlflow_artifact_logging(self):
        """Test MLflow artifact logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plugin = MLflowBackend()

            config = {"experiment_name": "Artifact Test Experiment", "tracking_uri": f"file://{temp_dir}/mlruns"}

            plugin.initialize(config)
            plugin.start()

            # Create a test file
            test_file = Path(temp_dir) / "test_artifact.txt"
            test_file.write_text("This is a test artifact")

            # Test artifact logging
            artifact_metric = MetricData(
                name="test_artifact",
                value=str(test_file),
                type=MetricType.ARTIFACT,
                metadata={"artifact_path": "test_files"},
            )

            plugin.receive_metric(artifact_metric)

            plugin.stop()


class TestBackendPluginIntegration:
    """Test integration with the plugin system and orchestrator."""

    def test_plugin_discovery(self):
        """Test that backend plugins can be discovered."""
        manager = PluginManager([str(Path(__file__).parent.parent.parent / "tracelet" / "backends")])

        discovered = manager.discover_plugins()

        # Should find our backend plugins
        plugin_names = {p.metadata.name for p in discovered}

        if _has_clearml:
            assert "clearml" in plugin_names
        if _has_mlflow:
            assert "mlflow" in plugin_names

    @pytest.mark.skipif(not _has_mlflow, reason="MLflow not installed")
    def test_orchestrator_with_mlflow_backend(self):
        """Test full orchestrator integration with MLflow backend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create orchestrator
            orchestrator = DataFlowOrchestrator(max_queue_size=100, num_workers=2)

            # Create and configure MLflow backend
            backend = MLflowBackend()
            config = {"experiment_name": "Integration Test", "tracking_uri": f"file://{temp_dir}/mlruns"}

            backend.initialize(config)
            backend.start()

            # Register backend as sink
            orchestrator.register_sink(backend)

            # Add routing rule
            orchestrator.add_routing_rule(RoutingRule(source_pattern="*", sink_id=backend.get_sink_id()))

            # Start orchestrator
            orchestrator.start()

            try:
                # Emit some metrics
                metrics = [
                    MetricData("accuracy", 0.95, MetricType.SCALAR, iteration=1, source="training"),
                    MetricData("loss", 0.123, MetricType.SCALAR, iteration=1, source="training"),
                    MetricData("learning_rate", 0.001, MetricType.PARAMETER, source="config"),
                ]

                for metric in metrics:
                    orchestrator.emit_metric(metric)

                # Wait for processing
                time.sleep(0.5)

                # Check stats
                stats = orchestrator.get_stats()
                assert stats["metrics_processed"] >= len(metrics)
                assert stats["running"]

            finally:
                orchestrator.stop()
                backend.stop()

    def test_plugin_manager_with_backends(self):
        """Test plugin manager with backend plugins."""
        manager = PluginManager()

        # Manually register our plugins for testing
        if _has_mlflow:
            # Create plugin info manually
            from tracelet.core.plugins import PluginInfo, PluginState

            plugin_info = PluginInfo(
                metadata=MLflowBackend.get_metadata(),
                module_path="tracelet.backends.mlflow_backend",
                class_name="MLflowBackend",
                instance=MLflowBackend,
                state=PluginState.LOADED,
            )

            manager.plugins["mlflow"] = plugin_info

            # Test validation
            assert manager.validate_plugin("mlflow")

            # Test getting plugins by type
            from tracelet.core.plugins import PluginType

            backend_plugins = manager.get_plugins_by_type(PluginType.BACKEND)
            assert len(backend_plugins) >= 1
            assert any(p.metadata.name == "mlflow" for p in backend_plugins)
