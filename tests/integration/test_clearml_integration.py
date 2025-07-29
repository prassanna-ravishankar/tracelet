from unittest.mock import Mock, patch

import pytest

try:
    import clearml  # noqa: F401

    _has_clearml = True
except ImportError:
    _has_clearml = False


@pytest.fixture
def mock_clearml_task():
    """Mock ClearML Task for testing without actual ClearML API calls."""
    with (
        patch("tracelet.backends.clearml_backend.Task") as mock_task_class,
        patch("tracelet.backends.clearml_backend.Logger") as mock_logger_class,
    ):
        # Create mock instances
        mock_task = Mock()
        mock_logger = Mock()

        # Configure the mocks
        mock_task_class.init.return_value = mock_task
        mock_task.get_logger.return_value = mock_logger
        mock_task.id = "test_task_123"
        mock_task.name = "test_experiment"
        mock_task.get_status.return_value = "running"
        mock_task.get_tags.return_value = ["test"]
        mock_task.get_output_log_web_page.return_value = (
            "https://app.clearml.ai/projects/test/experiments/test_task_123"
        )

        yield {
            "task_class": mock_task_class,
            "logger_class": mock_logger_class,
            "task": mock_task,
            "logger": mock_logger,
        }


@pytest.mark.skipif(not _has_clearml, reason="ClearML not installed")
def test_clearml_backend_plugin_metadata():
    """Test ClearML backend plugin metadata."""
    from tracelet.backends.clearml_backend import ClearMLBackend
    from tracelet.core.plugins import PluginType

    metadata = ClearMLBackend.get_metadata()

    assert metadata.name == "clearml"
    assert metadata.type == PluginType.BACKEND
    assert "experiment tracking" in metadata.description.lower()
    assert "enhanced_visualizations" in metadata.capabilities


@pytest.mark.skipif(not _has_clearml, reason="ClearML not installed")
def test_clearml_backend_initialization(mock_clearml_task):
    """Test ClearML backend initialization."""
    from tracelet.backends.clearml_backend import ClearMLBackend

    backend = ClearMLBackend()

    # Test configuration
    config = {
        "project_name": "Test Project",
        "task_name": "test_task",
        "task_type": "training",
        "tags": ["test", "integration"],
    }

    backend.initialize(config)
    backend.start()

    # Verify Task.init was called with correct parameters
    mock_clearml_task["task_class"].init.assert_called_once()
    call_kwargs = mock_clearml_task["task_class"].init.call_args.kwargs

    assert call_kwargs["project_name"] == "Test Project"
    assert call_kwargs["task_name"] == "test_task"
    assert call_kwargs["task_type"] == "training"
    assert call_kwargs["tags"] == ["test", "integration"]

    # Verify logger was obtained
    mock_clearml_task["task"].get_logger.assert_called_once()

    # Test backend status
    status = backend.get_status()
    assert status["active"] is True
    assert status["project_name"] == "Test Project"
    assert status["task_name"] == "test_task"


@pytest.mark.skipif(not _has_clearml, reason="ClearML not installed")
def test_clearml_backend_scalar_logging(mock_clearml_task):
    """Test scalar metric logging."""
    from tracelet.backends.clearml_backend import ClearMLBackend
    from tracelet.core.orchestrator import MetricData, MetricType

    backend = ClearMLBackend()
    backend.initialize({})
    backend.start()

    # Test scalar metric
    metric = MetricData(name="train/loss", value=0.5, type=MetricType.SCALAR, iteration=10)

    backend.receive_metric(metric)

    # Verify logger.report_scalar was called
    mock_clearml_task["logger"].report_scalar.assert_called_once_with(
        title="loss", series="train", value=0.5, iteration=10
    )


@pytest.mark.skipif(not _has_clearml, reason="ClearML not installed")
def test_clearml_backend_parameter_logging(mock_clearml_task):
    """Test parameter logging."""
    from tracelet.backends.clearml_backend import ClearMLBackend
    from tracelet.core.orchestrator import MetricData, MetricType

    backend = ClearMLBackend()
    backend.initialize({})
    backend.start()

    # Test parameter
    metric = MetricData(name="learning_rate", value=0.001, type=MetricType.PARAMETER, iteration=0)

    backend.receive_metric(metric)

    # Verify task.connect was called for parameters
    mock_clearml_task["task"].connect.assert_called()


@pytest.mark.skipif(not _has_clearml, reason="ClearML not installed")
def test_clearml_backend_enhanced_visualizations(mock_clearml_task):
    """Test enhanced visualization logging."""
    import numpy as np

    from tracelet.backends.clearml_backend import ClearMLBackend
    from tracelet.core.orchestrator import MetricData, MetricType

    backend = ClearMLBackend()
    backend.initialize({})
    backend.start()

    # Test histogram metric
    histogram_metric = MetricData(
        name="weights/histogram", value=np.random.randn(100), type=MetricType.HISTOGRAM, iteration=5
    )

    backend.receive_metric(histogram_metric)

    # Verify histogram logging
    mock_clearml_task["logger"].report_histogram.assert_called_once_with(
        title="histogram",
        series="weights",
        values=histogram_metric.value,
        iteration=5,
        xlabels=None,
        yaxis="Frequency",
        xaxis="Value",
    )

    # Test text metric
    text_metric = MetricData(
        name="debug_info", value="Model converged after 100 epochs", type=MetricType.TEXT, iteration=100
    )

    backend.receive_metric(text_metric)

    # Verify text logging
    mock_clearml_task["logger"].report_text.assert_called()


@pytest.mark.skipif(not _has_clearml, reason="ClearML not installed")
def test_clearml_backend_experiment_metadata(mock_clearml_task):
    """Test experiment metadata saving."""
    from tracelet.backends.clearml_backend import ClearMLBackend

    backend = ClearMLBackend()
    backend.initialize({})
    backend.start()

    # Test experiment metadata
    experiment_data = {
        "git": {"commit_hash": "abc123def456", "branch": "main", "dirty": False},
        "system": {"python_version": "3.11.0", "platform": "linux"},
        "description": "Test experiment with ClearML backend",
    }

    backend.save_experiment(experiment_data)

    # Verify git and system info were connected
    calls = mock_clearml_task["task"].connect.call_args_list
    call_names = [call[1]["name"] for call in calls if len(call) > 1 and "name" in call[1]]

    assert "git_info" in call_names
    assert "system_info" in call_names

    # Verify tags were set
    mock_clearml_task["task"].set_tags.assert_called()

    # Verify description was set
    mock_clearml_task["task"].set_description.assert_called_with("Test experiment with ClearML backend")


@pytest.mark.skipif(not _has_clearml, reason="ClearML not installed")
def test_clearml_backend_lifecycle(mock_clearml_task):
    """Test complete backend lifecycle."""
    from tracelet.backends.clearml_backend import ClearMLBackend

    backend = ClearMLBackend()

    # Initialize
    backend.initialize({"project_name": "Test Project"})
    assert not backend._active

    # Start
    backend.start()
    assert backend._active

    # Test logging works
    backend.log_metric("test_metric", 42.0, 1)
    mock_clearml_task["logger"].report_scalar.assert_called()

    # Stop
    backend.stop()
    assert not backend._active
    mock_clearml_task["task"].close.assert_called_once()


def test_clearml_backend_without_clearml():
    """Test ClearML backend behavior when ClearML is not available."""
    with patch("tracelet.backends.clearml_backend._has_clearml", False):
        from tracelet.backends.clearml_backend import ClearMLBackend

        backend = ClearMLBackend()

        with pytest.raises(ImportError, match="ClearML is not installed"):
            backend.initialize({})
