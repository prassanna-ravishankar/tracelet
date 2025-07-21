"""End-to-end integration tests for experiment tracking flow."""

import tempfile
import time
from pathlib import Path

import pytest

# Test imports
try:
    from tracelet.plugins.mlflow_backend import MLflowBackend
    _has_mlflow = True
except ImportError:
    _has_mlflow = False
    MLflowBackend = None

try:
    from tracelet.plugins.clearml_backend import ClearMLBackend
    _has_clearml = True
except ImportError:
    _has_clearml = False
    ClearMLBackend = None

from tracelet.core.experiment import Experiment, ExperimentConfig
from tracelet.core.plugins import PluginInfo, PluginManager, PluginState


class TestExperimentTrackingFlow:
    """Test complete experiment tracking workflows."""

    @pytest.mark.skipif(not _has_mlflow, reason="MLflow not installed")
    def test_complete_mlflow_experiment(self):
        """Test a complete experiment with MLflow backend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create experiment with MLflow backend
            config = ExperimentConfig(
                track_metrics=True,
                track_environment=True,
                track_args=True
            )

            # For this test, we'll manually set up the backend
            # In practice, this would be done via configuration
            experiment = Experiment(
                name="Complete MLflow Test",
                config=config,
                tags=["integration_test", "mlflow"]
            )

            # Manually add MLflow backend to plugin manager
            backend = MLflowBackend()
            backend_config = {
                "experiment_name": "Tracelet Integration Test",
                "run_name": "Complete Test Run",
                "tracking_uri": f"file://{temp_dir}/mlruns"
            }

            backend.initialize(backend_config)

            # Register backend with orchestrator
            experiment._orchestrator.register_sink(backend)

            # Add routing rule
            from tracelet.core.orchestrator import RoutingRule
            experiment._orchestrator.add_routing_rule(
                RoutingRule(source_pattern="*", sink_id=backend.get_sink_id())
            )

            # Start experiment and backend
            backend.start()
            experiment.start()

            try:
                # Log various types of data

                # 1. Parameters
                experiment.log_params({
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "model_type": "transformer",
                    "dataset": "custom_dataset"
                })

                # 2. Training loop simulation
                for epoch in range(5):
                    experiment.set_iteration(epoch)

                    # Training metrics
                    experiment.log_metric("train_loss", 1.0 - (epoch * 0.15), epoch)
                    experiment.log_metric("train_accuracy", 0.5 + (epoch * 0.08), epoch)

                    # Validation metrics
                    experiment.log_metric("val_loss", 1.1 - (epoch * 0.12), epoch)
                    experiment.log_metric("val_accuracy", 0.45 + (epoch * 0.09), epoch)

                # 3. Final metrics
                experiment.log_metric("final_accuracy", 0.95)
                experiment.log_metric("final_loss", 0.123)

                # 4. Create and log an artifact
                artifact_file = Path(temp_dir) / "model_summary.txt"
                artifact_file.write_text("""
Model Summary:
- Type: Transformer
- Parameters: 125M
- Final Accuracy: 95%
- Training Time: 2.5 hours
                """)

                experiment.log_artifact(str(artifact_file), "model/summary.txt")

                # Wait for processing
                time.sleep(1.0)

                # Check orchestrator stats
                stats = experiment._orchestrator.get_stats()
                assert stats["metrics_processed"] > 0
                assert stats["running"]

                # Check backend status
                backend_status = backend.get_status()
                assert backend_status["active"]
                assert backend_status["run_id"] is not None

            finally:
                experiment.stop()
                backend.stop()

            # Verify data was logged to MLflow
            import mlflow
            mlflow.set_tracking_uri(f"file://{temp_dir}/mlruns")

            exp = mlflow.get_experiment_by_name("Tracelet Integration Test")
            assert exp is not None

            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            assert len(runs) == 1

            run = runs.iloc[0]
            assert run["status"] == "FINISHED"

            # Check that parameters were logged (they'll have experiment ID prefix)
            param_cols = [col for col in run.index if "params." in col and "learning_rate" in col]
            assert len(param_cols) > 0
            assert run[param_cols[0]] == "0.001"

            # Check that metrics were logged
            metric_cols = [col for col in run.index if "metrics." in col and "final_accuracy" in col]
            assert len(metric_cols) > 0
            assert run[metric_cols[0]] == 0.95

    def test_experiment_with_plugin_manager_integration(self):
        """Test experiment with proper plugin manager integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a custom plugin manager with our backends
            plugin_manager = PluginManager()

            if _has_mlflow:
                # Manually register MLflow plugin with class
                mlflow_info = PluginInfo(
                    metadata=MLflowBackend.get_metadata(),
                    module_path="tracelet.plugins.mlflow_backend",
                    class_name="MLflowBackend",
                    instance=MLflowBackend,  # Pass the class, not instance
                    state=PluginState.LOADED   # Set to LOADED since we have class
                )
                plugin_manager.plugins["mlflow"] = mlflow_info

                # Test plugin validation (skip loading since we already have instance)
                assert plugin_manager.validate_plugin("mlflow")

                # Test plugin initialization
                config = {
                    "experiment_name": "Plugin Manager Test",
                    "tracking_uri": f"file://{temp_dir}/mlruns"
                }
                assert plugin_manager.initialize_plugin("mlflow", config)

                # Test plugin starting
                assert plugin_manager.start_plugin("mlflow")

                # Get plugin instance
                backend_instance = plugin_manager.get_plugin_instance("mlflow")
                assert backend_instance is not None
                assert backend_instance.get_status()["active"]

                # Test stopping
                assert plugin_manager.stop_plugin("mlflow")
                assert not backend_instance.get_status()["active"]

    def test_multiple_backend_experiment(self):
        """Test experiment with multiple backends (if available)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment = Experiment(
                name="Multi-Backend Test",
                config=ExperimentConfig(track_metrics=True),
                tags=["multi_backend"]
            )

            backends = []

            # Add MLflow backend if available
            if _has_mlflow:
                mlflow_backend = MLflowBackend()
                mlflow_config = {
                    "experiment_name": "Multi Backend MLflow",
                    "tracking_uri": f"file://{temp_dir}/mlruns"
                }
                mlflow_backend.initialize(mlflow_config)
                backends.append(mlflow_backend)

                # Register backend with orchestrator BEFORE starting
                experiment._orchestrator.register_sink(mlflow_backend)

                # Route all metrics to MLflow
                from tracelet.core.orchestrator import RoutingRule
                experiment._orchestrator.add_routing_rule(
                    RoutingRule(source_pattern="*", sink_id=mlflow_backend.get_sink_id())
                )

            # Add ClearML backend if available (in offline mode)
            if _has_clearml:
                try:
                    import os
                    os.environ["CLEARML_WEB_HOST"] = ""
                    os.environ["CLEARML_API_HOST"] = ""
                    os.environ["CLEARML_FILES_HOST"] = ""

                    clearml_backend = ClearMLBackend()
                    clearml_config = {
                        "project_name": "Multi Backend Test",
                        "task_name": "Multi Backend Task",
                        "auto_connect_frameworks": False
                    }
                    clearml_backend.initialize(clearml_config)
                    backends.append(clearml_backend)

                    # Register backend with orchestrator BEFORE starting
                    experiment._orchestrator.register_sink(clearml_backend)

                    # Route all metrics to ClearML as well
                    from tracelet.core.orchestrator import RoutingRule
                    experiment._orchestrator.add_routing_rule(
                        RoutingRule(source_pattern="*", sink_id=clearml_backend.get_sink_id())
                    )
                except Exception:
                    # ClearML might fail in CI environments
                    pass

            if not backends:
                pytest.skip("No backends available for testing")

            # Start backends first, then experiment
            for backend in backends:
                backend.start()

            experiment.start()

            try:
                # Log some metrics
                experiment.log_params({"backend_count": len(backends)})

                for i in range(3):
                    experiment.set_iteration(i)
                    experiment.log_metric("test_metric", i * 0.1, i)

                # Wait for processing - increase time for worker threads
                time.sleep(2.0)

                # Verify orchestrator processed metrics
                stats = experiment._orchestrator.get_stats()
                assert stats["metrics_processed"] > 0
                assert stats["num_sinks"] == len(backends)

            finally:
                experiment.stop()
                for backend in backends:
                    backend.stop()

    def test_error_handling_in_experiment_flow(self):
        """Test error handling during experiment tracking."""
        experiment = Experiment(
            name="Error Handling Test",
            config=ExperimentConfig(track_metrics=True)
        )

        # Create a mock backend that will fail
        class FailingBackend:
            def get_sink_id(self):
                return "failing_backend"

            def can_handle_type(self, metric_type):
                return True

            def receive_metric(self, metric):
                raise ValueError("Simulated backend failure")

        failing_backend = FailingBackend()
        experiment._orchestrator.register_sink(failing_backend)

        from tracelet.core.orchestrator import RoutingRule
        experiment._orchestrator.add_routing_rule(
            RoutingRule(source_pattern="*", sink_id="failing_backend")
        )

        experiment.start()

        try:
            # This should not crash the experiment despite backend failures
            experiment.log_metric("test_metric", 1.0)
            experiment.log_params({"test_param": "value"})

            # Wait for processing
            time.sleep(0.2)

            # Orchestrator should still be running despite backend errors
            stats = experiment._orchestrator.get_stats()
            assert stats["running"]
            assert stats["processing_errors"] > 0  # Should have recorded the errors

        finally:
            experiment.stop()
