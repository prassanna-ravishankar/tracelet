from unittest.mock import MagicMock, patch

import pytest

from tracelet.frameworks.pytorch import PyTorchFramework

# Check if torch is available
try:
    import torch
    from torch.utils.tensorboard import SummaryWriter

    _has_torch = True
except ImportError:
    _has_torch = False
    torch = SummaryWriter = None


@pytest.fixture
def mock_experiment():
    mock = MagicMock()
    mock.log_metric = MagicMock()
    return mock


@pytest.fixture
def mock_tensorboard():
    with patch("tracelet.frameworks.pytorch.importlib.import_module") as mock_import:
        mock_import.return_value = MagicMock()
        yield mock_import


def test_pytorch_framework_init():
    """Test framework initialization without tensorboard"""
    with patch("tracelet.frameworks.pytorch.importlib.import_module") as mock_import:
        mock_import.side_effect = ImportError()
        framework = PyTorchFramework()

        assert not framework._tensorboard_available
        assert framework._experiment is None


def test_pytorch_framework_with_tensorboard(mock_tensorboard):
    """Test framework initialization with tensorboard"""
    framework = PyTorchFramework()
    assert framework._tensorboard_available


@pytest.mark.skipif(not _has_torch, reason="PyTorch not installed")
def test_pytorch_framework_patching(mock_tensorboard, mock_experiment):
    """Test tensorboard patching"""
    framework = PyTorchFramework()
    framework.initialize(mock_experiment)

    # Import SummaryWriter after initialization to get the patched version
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()

    # Test scalar logging
    writer.add_scalar("test/metric", 0.5, 1)
    mock_experiment.log_metric.assert_called_with("test/metric", 0.5, 1)

    # Test multiple scalars
    writer.add_scalars("test", {"a": 1.0, "b": 2.0}, 2)
    assert mock_experiment.log_metric.call_count == 3
    mock_experiment.log_metric.assert_any_call("test/a", 1.0, 2)
    mock_experiment.log_metric.assert_any_call("test/b", 2.0, 2)


@pytest.mark.skipif(not _has_torch, reason="PyTorch not installed")
def test_pytorch_framework_no_patching(mock_tensorboard, mock_experiment):
    """Test disabling tensorboard patching"""
    framework = PyTorchFramework(patch_tensorboard=False)
    framework.initialize(mock_experiment)

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()

    # Logging should not trigger experiment logging
    writer.add_scalar("test/metric", 0.5, 1)
    mock_experiment.log_metric.assert_not_called()


@pytest.mark.skipif(not _has_torch, reason="PyTorch not installed")
def test_pytorch_framework_cleanup(mock_tensorboard, mock_experiment):
    """Test cleanup on stop"""
    framework = PyTorchFramework()
    framework.initialize(mock_experiment)

    # Check that patching happened
    assert framework._original_add_scalar is not None

    # Stop tracking
    framework.stop_tracking()

    # Check that references are cleared after cleanup
    assert framework._original_add_scalar is None


def test_pytorch_framework_tensorboard_availability(mock_experiment):
    """Test framework behavior with current tensorboard availability"""
    framework = PyTorchFramework()
    framework.initialize(mock_experiment)

    # Check that availability matches actual installation status
    assert framework._tensorboard_available == _has_torch

    # Should not raise errors regardless of tensorboard availability
    framework.start_tracking()
    framework.stop_tracking()
