"""Integration tests for AIM backend."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import aim  # noqa: F401

    _has_aim = True
except ImportError:
    _has_aim = False

from tracelet.backends.aim import AimBackend
from tracelet.core.orchestrator import MetricData, MetricType


@pytest.fixture
def temp_aim_repo():
    """Create a temporary AIM repository for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def aim_backend():
    """Create an AIM backend instance."""
    return AimBackend()


@pytest.mark.skipif(not _has_aim, reason="AIM not installed")
class TestAimBackendIntegration:
    """Integration tests for AIM backend."""

    def test_aim_backend_metadata(self, aim_backend):
        """Test AIM backend metadata."""
        metadata = aim_backend.get_metadata()
        assert metadata.name == "aim"
        assert metadata.version == "1.0.0"
        assert metadata.type.value == "backend"
        assert "AIM backend" in metadata.description
        assert "remote" in metadata.capabilities

    def test_aim_backend_initialization_local(self, aim_backend, temp_aim_repo):
        """Test AIM backend initialization with local repository."""
        config = {
            "repo_path": temp_aim_repo,
            "experiment_name": "test_experiment",
            "run_name": "test_run",
            "tags": {"env": "test", "framework": "pytorch"},
        }

        aim_backend.initialize(config)
        assert aim_backend._repo is not None
        assert aim_backend._experiment_name == "test_experiment"
        assert aim_backend._run_name == "test_run"
        assert aim_backend._tags == {"env": "test", "framework": "pytorch"}

    def test_aim_backend_initialization_without_aim(self):
        """Test AIM backend initialization when AIM is not available."""
        with patch("tracelet.backends.aim._has_aim", False):
            backend = AimBackend()
            config = {"repo_path": tempfile.mkdtemp()}

            with pytest.raises(ImportError, match="AIM is not installed"):
                backend.initialize(config)

    def test_aim_backend_lifecycle(self, aim_backend, temp_aim_repo):
        """Test complete AIM backend lifecycle."""
        # Initialize
        config = {"repo_path": temp_aim_repo, "experiment_name": "lifecycle_test", "run_name": "test_run"}
        aim_backend.initialize(config)

        # Start
        aim_backend.start()
        assert aim_backend._active is True
        assert aim_backend._run is not None
        assert aim_backend._run_hash is not None

        # Check status
        status = aim_backend.get_status()
        assert status["active"] is True
        assert status["experiment_name"] == "lifecycle_test"
        assert status["run_hash"] is not None
        assert status["has_run"] is True

        # Stop
        aim_backend.stop()
        assert aim_backend._active is False

    def test_aim_scalar_metric_logging(self, aim_backend, temp_aim_repo):
        """Test scalar metric logging."""
        config = {"repo_path": temp_aim_repo, "experiment_name": "scalar_test"}
        aim_backend.initialize(config)
        aim_backend.start()

        # Log scalar metrics
        for i in range(5):
            metric = MetricData(
                name="test_loss", value=1.0 / (i + 1), type=MetricType.SCALAR, iteration=i, source="experiment"
            )
            aim_backend.receive_metric(metric)

        # Verify metrics are logged
        run = aim_backend._run
        metric_names = run.get_metric_names()
        assert "test_loss" in metric_names

        # Check metric values
        metric_values = list(run.get_metric("test_loss").values.numpy())
        expected_values = [1.0 / (i + 1) for i in range(5)]
        np.testing.assert_array_almost_equal(metric_values, expected_values)

        aim_backend.stop()

    def test_aim_parameter_logging(self, aim_backend, temp_aim_repo):
        """Test parameter logging."""
        config = {"repo_path": temp_aim_repo, "experiment_name": "param_test"}
        aim_backend.initialize(config)
        aim_backend.start()

        # Log parameters
        params = [("learning_rate", 0.001), ("batch_size", 32), ("optimizer", "Adam"), ("epochs", 100)]

        for name, value in params:
            metric = MetricData(name=name, value=value, type=MetricType.PARAMETER, iteration=0, source="experiment")
            aim_backend.receive_metric(metric)

        # Verify parameters are logged
        run_params = dict(aim_backend._run.get_params())
        for name, value in params:
            assert name in run_params
            assert run_params[name] == value

        aim_backend.stop()

    def test_aim_distribution_logging(self, aim_backend, temp_aim_repo):
        """Test distribution/histogram logging."""
        config = {"repo_path": temp_aim_repo, "experiment_name": "dist_test"}
        aim_backend.initialize(config)
        aim_backend.start()

        # Log distribution
        distribution_data = np.random.normal(0, 1, 1000)
        metric = MetricData(
            name="weight_distribution", value=distribution_data, type=MetricType.HISTOGRAM, iteration=0, source="model"
        )

        aim_backend.receive_metric(metric)

        # Verify distribution is logged
        run = aim_backend._run
        metric_names = run.get_metric_names()
        assert "model/distributions/weight_distribution" in metric_names

        aim_backend.stop()

    def test_aim_text_logging(self, aim_backend, temp_aim_repo):
        """Test text logging."""
        config = {"repo_path": temp_aim_repo, "experiment_name": "text_test"}
        aim_backend.initialize(config)
        aim_backend.start()

        # Log text
        text_content = "This is a test summary of the training process."
        metric = MetricData(
            name="training_summary", value=text_content, type=MetricType.TEXT, iteration=0, source="experiment"
        )

        aim_backend.receive_metric(metric)

        # Verify text is logged
        run = aim_backend._run
        metric_names = run.get_metric_names()
        assert "experiment/text/training_summary" in metric_names

        aim_backend.stop()

    def test_aim_artifact_logging(self, aim_backend, temp_aim_repo):
        """Test artifact logging."""
        config = {"repo_path": temp_aim_repo, "experiment_name": "artifact_test"}
        aim_backend.initialize(config)
        aim_backend.start()

        # Create a temporary artifact file
        artifact_file = Path(temp_aim_repo) / "test_model.txt"
        artifact_file.write_text("This is a test model file.")

        # Log artifact
        metric = MetricData(
            name="model_file",
            value=str(artifact_file),
            type=MetricType.ARTIFACT,
            iteration=0,
            source="experiment",
            metadata={"artifact_path": "models/test_model.txt"},
        )

        aim_backend.receive_metric(metric)

        # Verify artifact reference is logged
        run_params = dict(aim_backend._run.get_params())
        assert "artifacts/models/test_model.txt" in run_params
        assert run_params["artifacts/models/test_model.txt"] == str(artifact_file)

        aim_backend.stop()

    def test_aim_image_logging(self, aim_backend, temp_aim_repo):
        """Test image logging."""
        config = {"repo_path": temp_aim_repo, "experiment_name": "image_test"}
        aim_backend.initialize(config)
        aim_backend.start()

        # Create sample image data (3x32x32 RGB image)
        image_data = np.random.rand(3, 32, 32)
        metric = MetricData(
            name="sample_image", value=image_data, type=MetricType.IMAGE, iteration=0, source="experiment"
        )

        aim_backend.receive_metric(metric)

        # Verify image is logged
        run = aim_backend._run
        metric_names = run.get_metric_names()
        assert "experiment/images/sample_image" in metric_names

        aim_backend.stop()

    def test_aim_audio_logging(self, aim_backend, temp_aim_repo):
        """Test audio logging."""
        config = {"repo_path": temp_aim_repo, "experiment_name": "audio_test"}
        aim_backend.initialize(config)
        aim_backend.start()

        # Create sample audio data
        sample_rate = 44100
        duration = 1.0  # 1 second
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))

        metric = MetricData(
            name="sample_audio",
            value=audio_data,
            type=MetricType.AUDIO,
            iteration=0,
            source="experiment",
            metadata={"sample_rate": sample_rate},
        )

        aim_backend.receive_metric(metric)

        # Verify audio is logged
        run = aim_backend._run
        metric_names = run.get_metric_names()
        assert "experiment/audio/sample_audio" in metric_names

        aim_backend.stop()

    def test_aim_backend_interface_methods(self, aim_backend, temp_aim_repo):
        """Test BackendInterface method implementations."""
        config = {"repo_path": temp_aim_repo, "experiment_name": "interface_test"}
        aim_backend.initialize(config)
        aim_backend.start()

        # Test log_metric
        aim_backend.log_metric("test_metric", 0.5, 10)

        # Test log_params
        params = {"lr": 0.001, "batch_size": 32}
        aim_backend.log_params(params)

        # Test log_artifact
        artifact_file = Path(temp_aim_repo) / "test_artifact.txt"
        artifact_file.write_text("Test artifact content")
        aim_backend.log_artifact(str(artifact_file), "artifacts/test.txt")

        # Test save_experiment
        experiment_data = {"total_epochs": 100, "dataset": "CIFAR-10"}
        aim_backend.save_experiment(experiment_data)

        # Verify all data is logged
        run = aim_backend._run

        # Check metric
        metric_names = run.get_metric_names()
        assert "test_metric" in metric_names

        # Check parameters
        run_params = dict(run.get_params())
        assert "lr" in run_params and run_params["lr"] == 0.001
        assert "batch_size" in run_params and run_params["batch_size"] == 32
        assert "artifacts/test.txt" in run_params
        assert "experiment/total_epochs" in run_params and run_params["experiment/total_epochs"] == 100
        assert "experiment/dataset" in run_params and run_params["experiment/dataset"] == "CIFAR-10"

        aim_backend.stop()

    def test_aim_metric_source_prefixing(self, aim_backend, temp_aim_repo):
        """Test that metrics are properly prefixed by source."""
        config = {"repo_path": temp_aim_repo, "experiment_name": "prefix_test"}
        aim_backend.initialize(config)
        aim_backend.start()

        # Log metrics from different sources
        sources_and_metrics = [
            ("tensorboard", "loss", MetricType.SCALAR),
            ("lightning", "accuracy", MetricType.SCALAR),
            ("experiment", "validation_loss", MetricType.SCALAR),
            ("model", "weight_norm", MetricType.HISTOGRAM),
        ]

        for source, name, metric_type in sources_and_metrics:
            metric = MetricData(
                name=name,
                value=np.random.random() if metric_type == MetricType.SCALAR else np.random.randn(100),
                type=metric_type,
                iteration=0,
                source=source,
            )
            aim_backend.receive_metric(metric)

        # Verify metrics have proper prefixes
        run = aim_backend._run
        metric_names = run.get_metric_names()

        assert "tensorboard/loss" in metric_names
        assert "lightning/accuracy" in metric_names
        assert "validation_loss" in metric_names  # experiment source doesn't get prefixed for scalars
        assert "model/distributions/weight_norm" in metric_names

        aim_backend.stop()

    def test_aim_error_handling(self, aim_backend, temp_aim_repo):
        """Test error handling in metric logging."""
        config = {"repo_path": temp_aim_repo, "experiment_name": "error_test"}
        aim_backend.initialize(config)
        aim_backend.start()

        # Test with invalid metric data
        with patch.object(aim_backend._run, "track", side_effect=Exception("AIM error")):
            metric = MetricData(
                name="problematic_metric",
                value=float("inf"),  # Might cause issues
                type=MetricType.SCALAR,
                iteration=0,
                source="experiment",
            )
            # Should not raise exception, just log the error
            aim_backend.receive_metric(metric)

        # Backend should still be active
        assert aim_backend._active is True

        aim_backend.stop()

    def test_aim_without_active_run(self, aim_backend, temp_aim_repo):
        """Test metric logging when no run is active."""
        config = {"repo_path": temp_aim_repo, "experiment_name": "inactive_test"}
        aim_backend.initialize(config)
        # Don't start the backend

        metric = MetricData(name="test_metric", value=0.5, type=MetricType.SCALAR, iteration=0, source="experiment")

        # Should handle gracefully
        aim_backend.receive_metric(metric)
        assert not aim_backend._active


@pytest.mark.skipif(not _has_aim, reason="AIM not installed")
class TestAimBackendRemote:
    """Tests for AIM backend remote functionality."""

    @patch("tracelet.backends.aim.aim.Repo.from_remote")
    def test_aim_remote_initialization(self, mock_from_remote, aim_backend):
        """Test AIM backend initialization with remote repository."""
        mock_repo = MagicMock()
        mock_from_remote.return_value = mock_repo

        config = {"remote_uri": "http://aim-server:53800", "experiment_name": "remote_test"}

        aim_backend.initialize(config)

        mock_from_remote.assert_called_once_with("http://aim-server:53800")
        assert aim_backend._remote_tracking_uri == "http://aim-server:53800"
        assert aim_backend._repo == mock_repo

    @patch("tracelet.backends.aim.aim.Repo.from_remote")
    def test_aim_remote_connection_failure(self, mock_from_remote, aim_backend):
        """Test handling of remote connection failures."""
        mock_from_remote.side_effect = Exception("Connection failed")

        config = {"remote_uri": "http://nonexistent-server:53800", "experiment_name": "connection_test"}

        with pytest.raises(ConnectionError):
            aim_backend.initialize(config)


@pytest.mark.skipif(_has_aim, reason="This test requires AIM to NOT be installed")
class TestAimBackendWithoutAim:
    """Tests for AIM backend when AIM is not installed."""

    def test_aim_backend_plugin_metadata_without_aim(self):
        """Test that plugin metadata works even without AIM installed."""
        # This should work even without AIM
        metadata = AimBackend.get_metadata()
        assert metadata.name == "aim"
        assert metadata.version == "1.0.0"

    def test_aim_backend_initialization_without_aim_installed(self):
        """Test initialization fails gracefully when AIM not installed."""
        backend = AimBackend()
        config = {"repo_path": tempfile.mkdtemp()}

        with pytest.raises(ImportError, match="AIM is not installed"):
            backend.initialize(config)
