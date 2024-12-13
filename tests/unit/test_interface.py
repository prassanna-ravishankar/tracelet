import os
import pytest
from unittest.mock import patch, MagicMock
import tracelet
from tracelet.settings import TraceletSettings

@pytest.fixture
def clean_env():
    """Remove any tracelet environment variables before tests"""
    tracelet_vars = [key for key in os.environ if key.startswith("TRACELET_")]
    for var in tracelet_vars:
        del os.environ[var]
    yield
    # Cleanup after test
    for var in tracelet_vars:
        del os.environ[var]

@pytest.fixture
def mock_experiment():
    with patch("tracelet.interface.Experiment") as mock:
        yield mock

@pytest.fixture
def mock_pytorch():
    with patch("tracelet.interface.PyTorchFramework") as mock:
        yield mock

@pytest.fixture
def mock_lightning():
    with patch("tracelet.interface.LightningFramework") as mock:
        yield mock

def test_start_logging_minimal(clean_env, mock_experiment, mock_pytorch, mock_lightning):
    """Test starting with minimal configuration"""
    exp = tracelet.start_logging("test_exp")
    
    # Check experiment creation
    mock_experiment.assert_called_once()
    assert mock_experiment.call_args[1]["name"] == "test_exp"
    
    # Check framework initialization
    mock_pytorch.assert_called_once()
    mock_pytorch.return_value.initialize.assert_called_once()
    mock_lightning.assert_called_once()
    mock_lightning.return_value.initialize.assert_called_once()

def test_start_logging_with_env_vars(clean_env, mock_experiment):
    """Test that environment variables are respected"""
    os.environ["TRACELET_PROJECT"] = "env_project"
    os.environ["TRACELET_BACKEND"] = "wandb"
    os.environ["TRACELET_API_KEY"] = "test_key"
    
    exp = tracelet.start_logging()
    
    assert mock_experiment.call_args[1]["tags"] == ["project:env_project"]
    assert mock_experiment.call_args[1]["backend"] == "wandb"

def test_start_logging_config_override(clean_env, mock_experiment, mock_pytorch):
    """Test that config parameters override environment variables"""
    os.environ["TRACELET_TRACK_TENSORBOARD"] = "true"
    
    exp = tracelet.start_logging(
        config={"track_tensorboard": False}
    )
    
    # PyTorch framework should not be initialized
    mock_pytorch.assert_not_called()

def test_stop_logging(clean_env, mock_experiment):
    """Test stopping experiment tracking"""
    exp = tracelet.start_logging("test_exp")
    mock_exp_instance = mock_experiment.return_value
    
    tracelet.stop_logging()
    
    mock_exp_instance.stop.assert_called_once()
    assert tracelet.get_active_experiment() is None

def test_get_active_experiment(clean_env, mock_experiment):
    """Test getting the active experiment"""
    assert tracelet.get_active_experiment() is None
    
    exp = tracelet.start_logging("test_exp")
    assert tracelet.get_active_experiment() == mock_experiment.return_value

def test_multiple_experiments(clean_env, mock_experiment):
    """Test that starting a new experiment stops the previous one"""
    exp1 = tracelet.start_logging("exp1")
    mock_exp1 = mock_experiment.return_value
    
    exp2 = tracelet.start_logging("exp2")
    
    mock_exp1.stop.assert_called_once()

@pytest.mark.parametrize("backend", ["mlflow", "wandb", "aim"])
def test_backend_validation(clean_env, backend):
    """Test that only valid backends are accepted"""
    exp = tracelet.start_logging(backend=backend)
    assert exp is not None

def test_invalid_backend(clean_env):
    """Test that invalid backend raises error"""
    with pytest.raises(ValueError):
        tracelet.start_logging(backend="invalid_backend") 