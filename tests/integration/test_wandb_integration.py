from unittest.mock import Mock, patch

import pytest

try:
    import wandb  # noqa: F401

    _has_wandb = True
except ImportError:
    _has_wandb = False


@pytest.fixture
def mock_wandb_run():
    """Mock W&B run for testing without actual W&B API calls."""
    with patch("tracelet.backends.wandb_backend.wandb") as mock_wandb:
        # Create mock run instance
        mock_run = Mock()
        mock_wandb.init.return_value = mock_run

        # Configure the mocks
        mock_run.id = "test_run_123"
        mock_run.name = "test_experiment"
        mock_run.url = "https://wandb.ai/user/project/runs/test_run_123"
        mock_run.state = "running"
        mock_run.tags = ["test"]
        mock_run.config = {}
        mock_run.notes = None

        # Mock W&B classes
        mock_wandb.Histogram = Mock(return_value="mock_histogram")
        mock_wandb.Image = Mock(return_value="mock_image")
        mock_wandb.Artifact = Mock()

        yield {"wandb": mock_wandb, "run": mock_run}


@pytest.mark.skipif(not _has_wandb, reason="W&B not installed")
def test_wandb_backend_plugin_metadata():
    """Test W&B backend plugin metadata."""
    from tracelet.backends.wandb_backend import WandbBackend
    from tracelet.core.plugins import PluginType

    metadata = WandbBackend.get_metadata()

    assert metadata.name == "wandb"
    assert metadata.type == PluginType.BACKEND
    assert "weights & biases" in metadata.description.lower()
    assert "enhanced_visualizations" in metadata.capabilities
    assert "search" in metadata.capabilities


@pytest.mark.skipif(not _has_wandb, reason="W&B not installed")
def test_wandb_backend_initialization(mock_wandb_run):
    """Test W&B backend initialization."""
    from tracelet.backends.wandb_backend import WandbBackend

    backend = WandbBackend()

    # Test configuration
    config = {
        "project_name": "Test Project",
        "experiment_name": "test_experiment",
        "entity": "test_user",
        "tags": ["test", "integration"],
        "job_type": "train",
    }

    backend.initialize(config)
    backend.start()

    # Verify wandb.init was called with correct parameters
    mock_wandb_run["wandb"].init.assert_called_once()
    call_kwargs = mock_wandb_run["wandb"].init.call_args.kwargs

    assert call_kwargs["project"] == "Test Project"
    assert call_kwargs["name"] == "test_experiment"
    assert call_kwargs["entity"] == "test_user"
    assert call_kwargs["tags"] == ["test", "integration"]
    assert call_kwargs["job_type"] == "train"
    assert call_kwargs["reinit"] is True

    # Test backend status
    status = backend.get_status()
    assert status["active"] is True
    assert status["project_name"] == "Test Project"
    assert status["experiment_name"] == "test_experiment"
    assert status["entity"] == "test_user"


@pytest.mark.skipif(not _has_wandb, reason="W&B not installed")
def test_wandb_backend_scalar_logging(mock_wandb_run):
    """Test scalar metric logging."""
    from tracelet.backends.wandb_backend import WandbBackend
    from tracelet.core.orchestrator import MetricData, MetricType

    backend = WandbBackend()
    backend.initialize({})
    backend.start()

    # Test scalar metric
    metric = MetricData(name="train/loss", value=0.5, type=MetricType.SCALAR, iteration=10)

    backend.receive_metric(metric)

    # Verify run.log was called
    mock_wandb_run["run"].log.assert_called_with({"train/loss": 0.5}, step=10)


@pytest.mark.skipif(not _has_wandb, reason="W&B not installed")
def test_wandb_backend_parameter_logging(mock_wandb_run):
    """Test parameter logging."""
    from tracelet.backends.wandb_backend import WandbBackend
    from tracelet.core.orchestrator import MetricData, MetricType

    backend = WandbBackend()
    backend.initialize({})
    backend.start()

    # Test parameter
    metric = MetricData(name="learning_rate", value=0.001, type=MetricType.PARAMETER, iteration=0)

    backend.receive_metric(metric)

    # Verify config was updated
    assert mock_wandb_run["run"].config["learning_rate"] == 0.001


@pytest.mark.skipif(not _has_wandb, reason="W&B not installed")
def test_wandb_backend_enhanced_visualizations(mock_wandb_run):
    """Test enhanced visualization logging."""
    import numpy as np

    from tracelet.backends.wandb_backend import WandbBackend
    from tracelet.core.orchestrator import MetricData, MetricType

    backend = WandbBackend()
    backend.initialize({})
    backend.start()

    # Test histogram metric
    histogram_metric = MetricData(
        name="weights/histogram", value=np.random.randn(100), type=MetricType.HISTOGRAM, iteration=5
    )

    backend.receive_metric(histogram_metric)

    # Verify histogram logging
    mock_wandb_run["wandb"].Histogram.assert_called_once_with(histogram_metric.value)
    mock_wandb_run["run"].log.assert_called_with({"weights/histogram": "mock_histogram"}, step=5)

    # Test image metric
    fake_image = np.random.rand(64, 64, 3)
    image_metric = MetricData(name="samples/input", value=fake_image, type=MetricType.IMAGE, iteration=10)

    backend.receive_metric(image_metric)

    # Verify image logging
    mock_wandb_run["wandb"].Image.assert_called_with(fake_image, caption=None)


@pytest.mark.skipif(not _has_wandb, reason="W&B not installed")
def test_wandb_backend_artifact_logging(mock_wandb_run):
    """Test artifact logging."""
    from tracelet.backends.wandb_backend import WandbBackend
    from tracelet.core.orchestrator import MetricData, MetricType

    backend = WandbBackend()
    backend.initialize({})
    backend.start()

    # Test artifact
    artifact_metric = MetricData(
        name="model_checkpoint",
        value="/path/to/model.pth",
        type=MetricType.ARTIFACT,
        metadata={"artifact_name": "checkpoint", "artifact_type": "model"},
    )

    backend.receive_metric(artifact_metric)

    # Verify artifact creation and logging
    mock_wandb_run["wandb"].Artifact.assert_called_with(name="checkpoint", type="model")

    artifact_instance = mock_wandb_run["wandb"].Artifact.return_value
    artifact_instance.add_file.assert_called_with("/path/to/model.pth")
    mock_wandb_run["run"].log_artifact.assert_called_with(artifact_instance)


@pytest.mark.skipif(not _has_wandb, reason="W&B not installed")
def test_wandb_backend_experiment_metadata(mock_wandb_run):
    """Test experiment metadata saving."""
    from tracelet.backends.wandb_backend import WandbBackend

    backend = WandbBackend()
    backend.initialize({})
    backend.start()

    # Test experiment metadata
    experiment_data = {
        "git": {"commit_hash": "abc123def456", "branch": "main", "dirty": False},
        "system": {"python_version": "3.11.0", "platform": "linux"},
        "description": "Test experiment with W&B backend",
    }

    backend.save_experiment(experiment_data)

    # Verify git info was added to config
    assert mock_wandb_run["run"].config["git_commit_hash"] == "abc123def456"
    assert mock_wandb_run["run"].config["git_branch"] == "main"
    assert mock_wandb_run["run"].config["git_dirty"] is False

    # Verify system info was added to config
    assert mock_wandb_run["run"].config["system_python_version"] == "3.11.0"
    assert mock_wandb_run["run"].config["system_platform"] == "linux"

    # Verify additional tags were stored
    expected_tags = ["git:abc123de", "python:3.11.0"]
    assert mock_wandb_run["run"].config["additional_tags"] == expected_tags

    # Verify description/notes were set
    assert mock_wandb_run["run"].notes == "Test experiment with W&B backend"


@pytest.mark.skipif(not _has_wandb, reason="W&B not installed")
def test_wandb_backend_lifecycle(mock_wandb_run):
    """Test complete backend lifecycle."""
    from tracelet.backends.wandb_backend import WandbBackend

    backend = WandbBackend()

    # Initialize
    backend.initialize({"project_name": "Test Project"})
    assert not backend._active

    # Start
    backend.start()
    assert backend._active

    # Test logging works
    backend.log_metric("test_metric", 42.0, 1)
    mock_wandb_run["run"].log.assert_called()

    # Stop
    backend.stop()
    assert not backend._active
    mock_wandb_run["run"].finish.assert_called_once()


@pytest.mark.skipif(not _has_wandb, reason="W&B not installed")
def test_wandb_backend_batch_operations(mock_wandb_run):
    """Test batch parameter and metric operations."""
    from tracelet.backends.wandb_backend import WandbBackend

    backend = WandbBackend()
    backend.initialize({})
    backend.start()

    # Test batch parameter logging
    params = {"learning_rate": 0.001, "batch_size": 32, "epochs": 100}

    backend.log_params(params)

    # Verify all parameters were added to config
    for key, value in params.items():
        assert mock_wandb_run["run"].config[key] == value

    # Test multiple metric logging
    for i in range(5):
        backend.log_metric(f"metric_{i}", i * 0.1, i)

    # Verify multiple log calls
    assert mock_wandb_run["run"].log.call_count >= 5


def test_wandb_backend_without_wandb():
    """Test W&B backend behavior when W&B is not available."""
    with patch("tracelet.backends.wandb_backend._has_wandb", False):
        from tracelet.backends.wandb_backend import WandbBackend

        backend = WandbBackend()

        with pytest.raises(ImportError, match="Weights & Biases is not installed"):
            backend.initialize({})
