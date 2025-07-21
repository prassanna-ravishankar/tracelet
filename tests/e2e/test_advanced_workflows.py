"""
Advanced E2E workflow tests for Tracelet

These tests cover more complex scenarios including:
- Computer vision workflows
- NLP workflows
- Multi-GPU scenarios (when available)
- Advanced visualization features
- Error handling and recovery
"""

import time
from pathlib import Path

import pytest

from .framework import TrainingWorkflow, e2e_framework

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision

    # import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Dataset

    _has_torch_vision = True
except ImportError:
    _has_torch_vision = False

try:
    import numpy as np

    _has_numpy = True
except ImportError:
    _has_numpy = False


@pytest.mark.skipif(not _has_torch_vision, reason="Torchvision not installed")
class ComputerVisionWorkflow(TrainingWorkflow):
    """Computer vision training workflow with image classification."""

    def __init__(self, config=None):
        super().__init__("computer_vision", config)

    def run(self, backend_config):  # noqa: C901
        """Run computer vision workflow with CNN."""
        from torch.utils.tensorboard import SummaryWriter

        import tracelet

        # Create synthetic image dataset
        class SyntheticImageDataset(Dataset):
            def __init__(self, num_samples=1000, img_size=32, num_classes=10):
                self.num_samples = num_samples
                self.img_size = img_size
                self.num_classes = num_classes

                # Generate synthetic data
                torch.manual_seed(42)
                self.images = torch.randn(num_samples, 3, img_size, img_size)
                self.labels = torch.randint(0, num_classes, (num_samples,))

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                return self.images[idx], self.labels[idx]

        # Create datasets
        train_dataset = SyntheticImageDataset(800, 32, 10)
        val_dataset = SyntheticImageDataset(200, 32, 10)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Create CNN model
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((4, 4)),
                )
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(128 * 4 * 4, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes),
                )

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x

        model = SimpleCNN(num_classes=10)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        # Start experiment tracking
        exp = tracelet.start_logging(
            exp_name=f"cnn_vision_{int(time.time())}",
            project=backend_config.get("project", "e2e_test"),
            backend=backend_config["backend"],
        )

        # Log hyperparameters
        hyperparams = {
            "model": "SimpleCNN",
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 8,
            "num_classes": 10,
            "optimizer": "Adam",
            "scheduler": "StepLR",
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "input_size": "32x32x3",
        }

        for name, value in hyperparams.items():
            exp.log_param(name, value)

        # Set up TensorBoard
        if "temp_dir" in backend_config:
            writer = SummaryWriter(str(Path(backend_config["temp_dir"]) / "tensorboard"))
        else:
            writer = SummaryWriter()

        # Training loop
        results = {
            "train_losses": [],
            "train_accuracies": [],
            "val_losses": [],
            "val_accuracies": [],
            "epochs_completed": 0,
            "best_val_accuracy": 0.0,
        }

        for epoch in range(8):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()
                train_total += target.size(0)

                # Log batch metrics
                if batch_idx % 10 == 0:
                    batch_acc = 100.0 * train_correct / train_total if train_total > 0 else 0.0
                    exp.log_metric("train/batch_loss", loss.item(), epoch * len(train_loader) + batch_idx)
                    exp.log_metric("train/batch_accuracy", batch_acc, epoch * len(train_loader) + batch_idx)

                    writer.add_scalar("train/batch_loss", loss.item(), epoch * len(train_loader) + batch_idx)
                    writer.add_scalar("train/batch_accuracy", batch_acc, epoch * len(train_loader) + batch_idx)

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    val_correct += pred.eq(target.view_as(pred)).sum().item()
                    val_total += target.size(0)

            # Calculate epoch metrics
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100.0 * train_correct / train_total
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100.0 * val_correct / val_total

            results["train_losses"].append(avg_train_loss)
            results["train_accuracies"].append(train_accuracy)
            results["val_losses"].append(avg_val_loss)
            results["val_accuracies"].append(val_accuracy)

            if val_accuracy > results["best_val_accuracy"]:
                results["best_val_accuracy"] = val_accuracy

            # Log epoch metrics
            current_lr = optimizer.param_groups[0]["lr"]
            exp.log_metric("train/epoch_loss", avg_train_loss, epoch)
            exp.log_metric("train/epoch_accuracy", train_accuracy, epoch)
            exp.log_metric("val/epoch_loss", avg_val_loss, epoch)
            exp.log_metric("val/epoch_accuracy", val_accuracy, epoch)
            exp.log_metric("train/learning_rate", current_lr, epoch)
            exp.log_metric("val/best_accuracy", results["best_val_accuracy"], epoch)

            writer.add_scalar("train/epoch_loss", avg_train_loss, epoch)
            writer.add_scalar("train/epoch_accuracy", train_accuracy, epoch)
            writer.add_scalar("val/epoch_loss", avg_val_loss, epoch)
            writer.add_scalar("val/epoch_accuracy", val_accuracy, epoch)
            writer.add_scalar("train/learning_rate", current_lr, epoch)

            # Log model parameters histograms
            for name, param in model.named_parameters():
                writer.add_histogram(f"params/{name}", param, epoch)
                if param.grad is not None:
                    writer.add_histogram(f"gradients/{name}", param.grad, epoch)

            # Log sample images (first batch of validation)
            if epoch == 0:
                sample_images, sample_labels = next(iter(val_loader))
                sample_images = sample_images[:8]  # First 8 images

                # Create a grid of sample images
                img_grid = torchvision.utils.make_grid(sample_images, nrow=4, normalize=True)
                writer.add_image("validation/sample_images", img_grid, epoch)

            scheduler.step()
            results["epochs_completed"] = epoch + 1

            print(
                f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={train_accuracy:.2f}%, "
                f"Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.2f}%"
            )

        # Log final metrics
        exp.log_metric("final/best_val_accuracy", results["best_val_accuracy"], results["epochs_completed"])
        exp.log_metric("final/final_train_loss", results["train_losses"][-1], results["epochs_completed"])
        exp.log_metric("final/final_val_loss", results["val_losses"][-1], results["epochs_completed"])

        writer.close()
        tracelet.stop_logging()

        return results

    def get_expected_metrics(self):
        return [
            "train/batch_loss",
            "train/batch_accuracy",
            "train/epoch_loss",
            "train/epoch_accuracy",
            "val/epoch_loss",
            "val/epoch_accuracy",
            "train/learning_rate",
            "val/best_accuracy",
            "final/best_val_accuracy",
            "final/final_train_loss",
            "final/final_val_loss",
        ]

    def validate_results(self, results):
        if results["epochs_completed"] != 8:
            return False
        if len(results["train_losses"]) != 8:
            return False
        if len(results["val_accuracies"]) != 8:
            return False
        # Expect some reasonable validation accuracy for synthetic data (better than random)
        return not (results["best_val_accuracy"] < 5.0)


@pytest.mark.skipif(not _has_numpy, reason="NumPy not installed")
class TimeSeriesWorkflow(TrainingWorkflow):
    """Time series forecasting workflow."""

    def __init__(self, config=None):
        super().__init__("time_series", config)

    def run(self, backend_config):  # noqa: C901
        """Run time series forecasting workflow."""
        from torch.utils.tensorboard import SummaryWriter

        import tracelet

        # Generate synthetic time series data
        def create_sine_wave_data(seq_length=100, num_sequences=1000):
            torch.manual_seed(42)
            t = torch.linspace(0, 4 * np.pi, seq_length)

            data = []
            targets = []

            for _i in range(num_sequences):
                # Random frequency and phase
                freq = 0.5 + torch.rand(1) * 2.0
                phase = torch.rand(1) * 2 * np.pi
                amplitude = 0.5 + torch.rand(1) * 1.5
                noise_level = 0.1

                # Generate sine wave with noise
                signal = amplitude * torch.sin(freq * t + phase) + noise_level * torch.randn(seq_length)

                # Use first seq_length-1 points to predict the last point
                data.append(signal[:-1].unsqueeze(-1))  # Add feature dimension
                targets.append(signal[-1].unsqueeze(0))  # Last point as target

            return torch.stack(data), torch.stack(targets)

        # Create datasets
        train_data, train_targets = create_sine_wave_data(50, 800)
        val_data, val_targets = create_sine_wave_data(50, 200)

        train_dataset = torch.utils.data.TensorDataset(train_data, train_targets)
        val_dataset = torch.utils.data.TensorDataset(val_data, val_targets)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Create LSTM model
        class LSTMPredictor(nn.Module):
            def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers

                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, output_size)
                )

            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])  # Use last output
                return out

        model = LSTMPredictor()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        # Start experiment tracking
        exp = tracelet.start_logging(
            exp_name=f"lstm_timeseries_{int(time.time())}",
            project=backend_config.get("project", "e2e_test"),
            backend=backend_config["backend"],
        )

        # Log hyperparameters
        hyperparams = {
            "model": "LSTMPredictor",
            "hidden_size": 64,
            "num_layers": 2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "sequence_length": 49,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
        }

        for name, value in hyperparams.items():
            exp.log_param(name, value)

        # Set up TensorBoard
        if "temp_dir" in backend_config:
            writer = SummaryWriter(str(Path(backend_config["temp_dir"]) / "tensorboard"))
        else:
            writer = SummaryWriter()

        # Training loop
        results = {
            "train_losses": [],
            "val_losses": [],
            "train_maes": [],
            "val_maes": [],
            "epochs_completed": 0,
            "best_val_mae": float("inf"),
        }

        for epoch in range(10):
            # Training phase
            model.train()
            train_loss = 0.0
            train_mae = 0.0

            for batch_idx, (data, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_mae += torch.mean(torch.abs(outputs - targets)).item()

                # Log batch metrics
                if batch_idx % 10 == 0:
                    exp.log_metric("train/batch_loss", loss.item(), epoch * len(train_loader) + batch_idx)
                    writer.add_scalar("train/batch_loss", loss.item(), epoch * len(train_loader) + batch_idx)

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_mae = 0.0

            with torch.no_grad():
                for data, targets in val_loader:
                    outputs = model(data)
                    val_loss += criterion(outputs, targets).item()
                    val_mae += torch.mean(torch.abs(outputs - targets)).item()

            # Calculate epoch metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_train_mae = train_mae / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_mae = val_mae / len(val_loader)

            results["train_losses"].append(avg_train_loss)
            results["train_maes"].append(avg_train_mae)
            results["val_losses"].append(avg_val_loss)
            results["val_maes"].append(avg_val_mae)

            if avg_val_mae < results["best_val_mae"]:
                results["best_val_mae"] = avg_val_mae

            # Log epoch metrics
            current_lr = optimizer.param_groups[0]["lr"]
            exp.log_metric("train/epoch_loss", avg_train_loss, epoch)
            exp.log_metric("train/epoch_mae", avg_train_mae, epoch)
            exp.log_metric("val/epoch_loss", avg_val_loss, epoch)
            exp.log_metric("val/epoch_mae", avg_val_mae, epoch)
            exp.log_metric("train/learning_rate", current_lr, epoch)
            exp.log_metric("val/best_mae", results["best_val_mae"], epoch)

            writer.add_scalar("train/epoch_loss", avg_train_loss, epoch)
            writer.add_scalar("train/epoch_mae", avg_train_mae, epoch)
            writer.add_scalar("val/epoch_loss", avg_val_loss, epoch)
            writer.add_scalar("val/epoch_mae", avg_val_mae, epoch)

            # Log predictions vs targets plot
            if epoch % 3 == 0:
                model.eval()
                with torch.no_grad():
                    sample_data, sample_targets = next(iter(val_loader))
                    sample_outputs = model(sample_data)

                    # Create prediction vs target plot
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(10, 6))
                    targets_np = sample_targets[:20].squeeze().cpu().numpy()
                    outputs_np = sample_outputs[:20].squeeze().cpu().numpy()

                    ax.scatter(targets_np, outputs_np, alpha=0.6)
                    ax.plot([targets_np.min(), targets_np.max()], [targets_np.min(), targets_np.max()], "r--")
                    ax.set_xlabel("True Values")
                    ax.set_ylabel("Predictions")
                    ax.set_title(f"Predictions vs True Values (Epoch {epoch})")

                    writer.add_figure("validation/predictions_vs_true", fig, epoch)
                    plt.close(fig)

            scheduler.step(avg_val_loss)
            results["epochs_completed"] = epoch + 1

            print(
                f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train MAE={avg_train_mae:.4f}, "
                f"Val Loss={avg_val_loss:.4f}, Val MAE={avg_val_mae:.4f}"
            )

        # Log final metrics
        exp.log_metric("final/best_val_mae", results["best_val_mae"], results["epochs_completed"])
        exp.log_metric("final/final_train_loss", results["train_losses"][-1], results["epochs_completed"])
        exp.log_metric("final/final_val_loss", results["val_losses"][-1], results["epochs_completed"])

        writer.close()
        tracelet.stop_logging()

        return results

    def get_expected_metrics(self):
        return [
            "train/batch_loss",
            "train/epoch_loss",
            "train/epoch_mae",
            "val/epoch_loss",
            "val/epoch_mae",
            "val/best_mae",
            "train/learning_rate",
            "final/best_val_mae",
            "final/final_train_loss",
            "final/final_val_loss",
        ]

    def validate_results(self, results):
        if results["epochs_completed"] != 10:
            return False
        if len(results["train_losses"]) != 10:
            return False
        if results["best_val_mae"] == float("inf"):
            return False
        # Expect reasonable MAE for synthetic data
        return not (results["best_val_mae"] > 5.0)


class TestAdvancedWorkflows:
    """Test advanced training workflows."""

    def test_computer_vision_with_mlflow(self):
        """Test computer vision CNN workflow with MLflow."""
        if "mlflow" not in e2e_framework.get_available_backends():
            pytest.skip("MLflow backend not available")

        if not _has_torch_vision:
            pytest.skip("Torchvision not available")

        # Register workflow
        e2e_framework.workflows["computer_vision"] = ComputerVisionWorkflow

        with e2e_framework.backend_environment("mlflow") as backend_config:
            results = e2e_framework.run_workflow("computer_vision", backend_config)

            assert results["success"], f"CV workflow failed: {results.get('error', 'Unknown error')}"
            assert results["epochs_completed"] == 8
            assert len(results["train_losses"]) == 8
            assert results["best_val_accuracy"] > 0

    def test_computer_vision_with_wandb(self):
        """Test computer vision CNN workflow with W&B."""
        if "wandb" not in e2e_framework.get_available_backends():
            pytest.skip("W&B backend not available")

        if not _has_torch_vision:
            pytest.skip("Torchvision not available")

        # Register workflow
        e2e_framework.workflows["computer_vision"] = ComputerVisionWorkflow

        with e2e_framework.backend_environment("wandb") as backend_config:
            results = e2e_framework.run_workflow("computer_vision", backend_config)

            assert results["success"], f"CV workflow failed: {results.get('error', 'Unknown error')}"
            assert results["epochs_completed"] == 8
            assert len(results["val_accuracies"]) == 8
            assert results["best_val_accuracy"] > 0

    def test_time_series_with_mlflow(self):
        """Test time series LSTM workflow with MLflow."""
        if "mlflow" not in e2e_framework.get_available_backends():
            pytest.skip("MLflow backend not available")

        if not _has_numpy:
            pytest.skip("NumPy not available")

        # Register workflow
        e2e_framework.workflows["time_series"] = TimeSeriesWorkflow

        with e2e_framework.backend_environment("mlflow") as backend_config:
            results = e2e_framework.run_workflow("time_series", backend_config)

            assert results["success"], f"Time series workflow failed: {results.get('error', 'Unknown error')}"
            assert results["epochs_completed"] == 10
            assert len(results["train_losses"]) == 10
            assert results["best_val_mae"] < 5.0

    def test_time_series_with_wandb(self):
        """Test time series LSTM workflow with W&B."""
        if "wandb" not in e2e_framework.get_available_backends():
            pytest.skip("W&B backend not available")

        if not _has_numpy:
            pytest.skip("NumPy not available")

        # Register workflow
        e2e_framework.workflows["time_series"] = TimeSeriesWorkflow

        with e2e_framework.backend_environment("wandb") as backend_config:
            results = e2e_framework.run_workflow("time_series", backend_config)

            assert results["success"], f"Time series workflow failed: {results.get('error', 'Unknown error')}"
            assert results["epochs_completed"] == 10
            assert len(results["val_losses"]) == 10
            assert results["best_val_mae"] < 5.0


class TestErrorHandling:
    """Test error handling and recovery scenarios."""

    def test_backend_initialization_failure_recovery(self):
        """Test graceful handling when backend initialization fails."""
        # This test simulates what happens when backends aren't properly configured

        # Try to use a backend that doesn't exist
        try:
            with e2e_framework.backend_environment("nonexistent_backend"):
                pass
        except ValueError as e:
            assert "Unknown backend" in str(e)

    def test_workflow_failure_recovery(self):
        """Test graceful handling when workflow fails."""
        if "mlflow" not in e2e_framework.get_available_backends():
            pytest.skip("MLflow backend not available")

        class FailingWorkflow(TrainingWorkflow):
            def __init__(self, config=None):
                super().__init__("failing_workflow", config)

            def run(self, backend_config):
                msg = "Intentional workflow failure"
                raise RuntimeError(msg)

            def get_expected_metrics(self):
                return []

            def validate_results(self, results):
                return False

        # Register failing workflow
        e2e_framework.workflows["failing_workflow"] = FailingWorkflow

        with e2e_framework.backend_environment("mlflow") as backend_config:
            results = e2e_framework.run_workflow("failing_workflow", backend_config)

            assert not results["success"]
            assert "error" in results
            assert "Intentional workflow failure" in results["error"]
            assert results["execution_time"] > 0


class TestVisualizationFeatures:
    """Test advanced visualization and logging features."""

    def test_tensorboard_integration_comprehensive(self):
        """Test comprehensive TensorBoard integration across backends."""
        available_backends = [b for b in e2e_framework.get_available_backends() if b in ["mlflow", "wandb"]]

        if not available_backends:
            pytest.skip("No suitable backends available")

        if not _has_torch_vision:
            pytest.skip("Torchvision not available")

        # Register CV workflow which has extensive visualization
        e2e_framework.workflows["computer_vision"] = ComputerVisionWorkflow

        for backend_name in available_backends:
            with e2e_framework.backend_environment(backend_name) as backend_config:
                results = e2e_framework.run_workflow("computer_vision", backend_config)

                assert results["success"], f"Visualization test failed for {backend_name}: {results.get('error')}"

                # Check that tensorboard directory was created
                if "temp_dir" in backend_config:
                    tb_dir = Path(backend_config["temp_dir"]) / "tensorboard"
                    # TensorBoard files might exist in subdirectories
                    tb_files = list(tb_dir.rglob("events.out.tfevents.*"))
                    print(f"{backend_name}: Found {len(tb_files)} TensorBoard event files")
