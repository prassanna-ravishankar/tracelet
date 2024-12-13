import os
import pytest
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tracelet

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        
    def forward(self, x):
        return self.linear(x)

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def mock_data():
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    return x, y

def test_pytorch_experiment_tracking(temp_dir, mock_data):
    """Test full experiment tracking workflow with PyTorch"""
    mlflow_tracking_uri = str(temp_dir / "mlruns")
    os.environ["TRACELET_BACKEND_URL"] = mlflow_tracking_uri
    
    # Start experiment tracking
    experiment = tracelet.start_logging(
        exp_name="test_pytorch",
        project="integration_test",
        backend="mlflow"
    )
    
    # Create model and optimizer
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Training loop with tensorboard logging
    writer = SummaryWriter(str(temp_dir / "runs"))
    x, y = mock_data
    
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # Log metrics - should be captured by tracelet
        writer.add_scalar("train/loss", loss.item(), epoch)
        writer.add_scalars("train/metrics", {
            "loss": loss.item(),
            "learning_rate": 0.01
        }, epoch)
    
    # Stop tracking
    tracelet.stop_logging()
    
    # Verify MLflow artifacts were created
    mlruns_dir = Path(mlflow_tracking_uri)
    assert mlruns_dir.exists()
    assert any(mlruns_dir.glob("**/metrics"))

@pytest.mark.skipif(not tracelet.frameworks.lightning.LightningFramework._check_lightning(),
                   reason="PyTorch Lightning not installed")
def test_lightning_experiment_tracking(temp_dir, mock_data):
    """Test full experiment tracking workflow with Lightning"""
    import pytorch_lightning as pl
    
    class LightningModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = SimpleModel()
            
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.model(x)
            loss = nn.MSELoss()(y_hat, y)
            self.log("train/loss", loss)
            return loss
            
        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)
    
    # Start experiment tracking
    experiment = tracelet.start_logging(
        exp_name="test_lightning",
        project="integration_test",
        backend="mlflow",
        config={"track_tensorboard": False}  # Use only Lightning logging
    )
    
    # Create data
    x, y = mock_data
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    # Train model
    model = LightningModel()
    trainer = pl.Trainer(
        max_epochs=5,
        enable_checkpointing=False,
        logger=False  # Disable default logger
    )
    trainer.fit(model, dataloader)
    
    # Stop tracking
    tracelet.stop_logging() 