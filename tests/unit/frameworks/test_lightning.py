from unittest.mock import MagicMock, patch

import pytest

from tracelet.frameworks.lightning import LightningFramework

# Check if pytorch_lightning is available
try:
    import pytorch_lightning as pl

    _has_lightning = True
except ImportError:
    _has_lightning = False
    pl = None


@pytest.fixture
def mock_experiment():
    mock = MagicMock()
    mock.log_metric = MagicMock()
    return mock


@pytest.fixture
def mock_lightning():
    with patch("tracelet.frameworks.lightning.importlib.import_module") as mock_import:
        mock_pl = MagicMock()
        mock_pl.Trainer = MagicMock()
        mock_import.return_value = mock_pl
        yield mock_import


def test_lightning_framework_init():
    """Test framework initialization without lightning"""
    with patch("tracelet.frameworks.lightning.importlib.import_module") as mock_import:
        mock_import.side_effect = ImportError()
        framework = LightningFramework()

        assert not framework._lightning_available
        assert framework._experiment is None


def test_lightning_framework_with_lightning(mock_lightning):
    """Test framework initialization with lightning"""
    framework = LightningFramework()
    assert framework._lightning_available


@pytest.mark.skipif(not _has_lightning, reason="PyTorch Lightning not installed")
def test_lightning_framework_patching(mock_lightning, mock_experiment):
    """Test lightning logging patching"""
    framework = LightningFramework()
    framework.initialize(mock_experiment)

    # Get the patched Trainer
    import pytorch_lightning as pl

    trainer = pl.Trainer()

    # Test metric logging
    metrics = {"train/loss": 0.5, "val/accuracy": 0.95}
    trainer.log_metrics(metrics, step=10)

    assert mock_experiment.log_metric.call_count == 2
    mock_experiment.log_metric.assert_any_call("train/loss", 0.5, 10)
    mock_experiment.log_metric.assert_any_call("val/accuracy", 0.95, 10)


@pytest.mark.skipif(not _has_lightning, reason="PyTorch Lightning not installed")
def test_lightning_framework_cleanup(mock_lightning, mock_experiment):
    """Test cleanup on stop"""
    framework = LightningFramework()
    framework.initialize(mock_experiment)

    # Store original method
    import pytorch_lightning as pl

    original_log_metrics = pl.Trainer.log_metrics

    # Stop tracking
    framework.stop_tracking()

    # Method should be restored
    assert pl.Trainer.log_metrics == original_log_metrics


def test_lightning_framework_availability(mock_experiment):
    """Test framework behavior with current lightning availability"""
    framework = LightningFramework()
    framework.initialize(mock_experiment)

    # Check that availability matches actual installation status
    assert framework._lightning_available == _has_lightning

    # Should not raise errors regardless of lightning availability
    framework.start_tracking()
    framework.stop_tracking()


@pytest.mark.skipif(not _has_lightning, reason="PyTorch Lightning not installed")
def test_lightning_framework_epoch_handling(mock_lightning, mock_experiment):
    """Test handling of epoch vs step in logging"""
    framework = LightningFramework()
    framework.initialize(mock_experiment)

    import pytorch_lightning as pl

    trainer = pl.Trainer()
    trainer.current_epoch = 5

    # Test logging without step (should use epoch)
    metrics = {"train/loss": 0.5}
    trainer.log_metrics(metrics)

    mock_experiment.log_metric.assert_called_with("train/loss", 0.5, 5)

    # Test logging with step (should use step)
    trainer.log_metrics(metrics, step=10)
    mock_experiment.log_metric.assert_called_with("train/loss", 0.5, 10)
