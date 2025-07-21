"""
E2E Test Framework for Tracelet

This module provides a comprehensive framework for end-to-end testing
of Tracelet across all supported backends with realistic PyTorch workflows.
"""

import contextlib
import logging
import os
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import pytest

try:
    import docker

    _has_docker = True
except ImportError:
    docker = None
    _has_docker = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    _has_torch = True
except ImportError:
    torch = nn = optim = DataLoader = TensorDataset = None
    _has_torch = False

try:
    import pytorch_lightning as pl

    _has_lightning = True
except ImportError:
    pl = None
    _has_lightning = False

try:
    import numpy as np

    _has_numpy = True
except ImportError:
    np = None
    _has_numpy = False

logger = logging.getLogger(__name__)


class BackendEnvironment(ABC):
    """Abstract base class for backend test environments."""

    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.config = config
        self._temp_dir: Optional[Path] = None
        self._docker_containers: list[Any] = []

    @abstractmethod
    def setup(self) -> dict[str, Any]:
        """Set up the backend environment and return connection config."""
        pass

    @abstractmethod
    def teardown(self):
        """Clean up the backend environment."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend dependencies are available."""
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """Get the backend name for Tracelet initialization."""
        pass

    def _create_temp_dir(self) -> Path:
        """Create a temporary directory for the test."""
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix=f"tracelet_e2e_{self.name}_"))
        return self._temp_dir

    def _cleanup_temp_dir(self):
        """Clean up temporary directory."""
        if self._temp_dir and self._temp_dir.exists():
            import shutil

            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None


class MLflowEnvironment(BackendEnvironment):
    """MLflow backend test environment."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__("mlflow", config or {})

    def is_available(self) -> bool:
        try:
            import mlflow  # noqa: F401
        except ImportError:
            return False
        else:
            return True

    def get_backend_name(self) -> str:
        return "mlflow"

    def setup(self) -> dict[str, Any]:
        """Set up MLflow with local file store."""
        temp_dir = self._create_temp_dir()
        tracking_uri = f"file://{temp_dir}/mlruns"

        # Set environment variables
        os.environ["TRACELET_BACKEND_URL"] = tracking_uri

        return {"backend": "mlflow", "project": "e2e_test", "tracking_uri": tracking_uri, "temp_dir": str(temp_dir)}

    def teardown(self):
        """Clean up MLflow environment."""
        # Clear environment variables
        os.environ.pop("TRACELET_BACKEND_URL", None)
        self._cleanup_temp_dir()


class ClearMLEnvironment(BackendEnvironment):
    """ClearML backend test environment."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__("clearml", config or {})

    def is_available(self) -> bool:
        try:
            import clearml  # noqa: F401
        except ImportError:
            return False
        else:
            return True

    def get_backend_name(self) -> str:
        return "clearml"

    def setup(self) -> dict[str, Any]:
        """Set up ClearML in offline mode."""
        temp_dir = self._create_temp_dir()

        # Set offline mode environment variables
        os.environ["CLEARML_WEB_HOST"] = ""
        os.environ["CLEARML_API_HOST"] = ""
        os.environ["CLEARML_FILES_HOST"] = ""
        os.environ["CLEARML_OFFLINE_MODE"] = "1"

        return {"backend": "clearml", "project": "e2e_test", "temp_dir": str(temp_dir)}

    def teardown(self):
        """Clean up ClearML environment."""
        # Clear environment variables
        for key in ["CLEARML_WEB_HOST", "CLEARML_API_HOST", "CLEARML_FILES_HOST", "CLEARML_OFFLINE_MODE"]:
            os.environ.pop(key, None)
        self._cleanup_temp_dir()


class WandbEnvironment(BackendEnvironment):
    """Weights & Biases backend test environment."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__("wandb", config or {})

    def is_available(self) -> bool:
        try:
            import wandb  # noqa: F401
        except ImportError:
            return False
        else:
            return True

    def get_backend_name(self) -> str:
        return "wandb"

    def setup(self) -> dict[str, Any]:
        """Set up W&B in offline mode."""
        temp_dir = self._create_temp_dir()

        # Set offline mode environment variables
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = str(temp_dir)
        os.environ["WANDB_CACHE_DIR"] = str(temp_dir / "cache")

        return {"backend": "wandb", "project": "e2e_test", "temp_dir": str(temp_dir)}

    def teardown(self):
        """Clean up W&B environment."""
        # Clear environment variables
        for key in ["WANDB_MODE", "WANDB_DIR", "WANDB_CACHE_DIR"]:
            os.environ.pop(key, None)
        self._cleanup_temp_dir()


class TrainingWorkflow(ABC):
    """Abstract base class for training workflows."""

    def __init__(self, name: str, config: Optional[dict[str, Any]] = None):
        self.name = name
        self.config = config or {}

    @abstractmethod
    def run(self, backend_config: dict[str, Any]) -> dict[str, Any]:
        """Run the training workflow and return metrics."""
        pass

    @abstractmethod
    def get_expected_metrics(self) -> list[str]:
        """Get list of expected metric names."""
        pass

    @abstractmethod
    def validate_results(self, results: dict[str, Any]) -> bool:
        """Validate that the training produced expected results."""
        pass


@pytest.mark.skipif(not _has_torch, reason="PyTorch not installed")
class SimplePyTorchWorkflow(TrainingWorkflow):
    """Simple PyTorch training workflow for testing."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__("simple_pytorch", config)

    def run(self, backend_config: dict[str, Any]) -> dict[str, Any]:
        """Run a simple PyTorch training loop."""
        from torch.utils.tensorboard import SummaryWriter

        import tracelet

        # Generate synthetic data
        torch.manual_seed(42)
        X = torch.randn(1000, 10)
        y = torch.sum(X[:, :3], dim=1, keepdim=True) + 0.1 * torch.randn(1000, 1)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Create model
        model = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Start experiment tracking
        exp = tracelet.start_logging(
            exp_name=f"pytorch_{self.name}_{int(time.time())}",
            project=backend_config.get("project", "e2e_test"),
            backend=backend_config["backend"],
        )

        # Log hyperparameters
        hyperparams = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "model_layers": 3,
            "dropout": 0.1,
            "optimizer": "adam",
        }

        exp.log_params(hyperparams)

        # Set up TensorBoard logging
        if "temp_dir" in backend_config:
            writer = SummaryWriter(str(Path(backend_config["temp_dir"]) / "tensorboard"))
        else:
            writer = SummaryWriter()

        # Training loop
        results = {"losses": [], "accuracies": [], "epochs_completed": 0}

        model.train()
        for epoch in range(10):
            epoch_loss = 0.0
            epoch_samples = 0

            for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
                optimizer.zero_grad()

                # Forward pass
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)
                epoch_samples += batch_x.size(0)

                # Log batch metrics
                if batch_idx % 10 == 0:
                    exp.log_metric("train/batch_loss", loss.item(), epoch * len(dataloader) + batch_idx)
                    writer.add_scalar("train/batch_loss", loss.item(), epoch * len(dataloader) + batch_idx)

            avg_loss = epoch_loss / epoch_samples
            results["losses"].append(avg_loss)

            # Calculate R² as accuracy metric
            model.eval()
            with torch.no_grad():
                predictions = model(X)
                ss_tot = torch.sum((y - torch.mean(y)) ** 2)
                ss_res = torch.sum((y - predictions) ** 2)
                r2_score = 1 - ss_res / ss_tot
                accuracy = float(r2_score)
            model.train()

            results["accuracies"].append(accuracy)

            # Log epoch metrics
            exp.log_metric("train/epoch_loss", avg_loss, epoch)
            exp.log_metric("train/accuracy_r2", accuracy, epoch)
            exp.log_metric("train/learning_rate", optimizer.param_groups[0]["lr"], epoch)

            writer.add_scalar("train/epoch_loss", avg_loss, epoch)
            writer.add_scalar("train/accuracy_r2", accuracy, epoch)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch)

            # Log histograms of model parameters
            for name, param in model.named_parameters():
                writer.add_histogram(f"params/{name}", param, epoch)
                writer.add_histogram(f"gradients/{name}", param.grad, epoch)

            results["epochs_completed"] = epoch + 1

            logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, R²={accuracy:.4f}")

        # Log final metrics
        exp.log_metric("final/best_accuracy", max(results["accuracies"]), results["epochs_completed"])
        exp.log_metric("final/final_loss", results["losses"][-1], results["epochs_completed"])

        # Log model artifact (save model)
        if "temp_dir" in backend_config:
            model_path = Path(backend_config["temp_dir"]) / "model.pth"
            torch.save(model.state_dict(), model_path)
            exp.log_artifact(str(model_path), "model.pth")

        writer.close()
        tracelet.stop_logging()

        return results

    def get_expected_metrics(self) -> list[str]:
        return [
            "train/batch_loss",
            "train/epoch_loss",
            "train/accuracy_r2",
            "train/learning_rate",
            "final/best_accuracy",
            "final/final_loss",
        ]

    def validate_results(self, results: dict[str, Any]) -> bool:
        """Validate that training completed successfully."""
        if results["epochs_completed"] != 10:
            return False

        if len(results["losses"]) != 10:
            return False

        if len(results["accuracies"]) != 10:
            return False

        # Check that loss generally decreased
        if results["losses"][-1] > results["losses"][0] * 1.5:  # Allow some variance
            return False

        # Check that accuracy generally improved (allow some variance)
        return not (results["accuracies"][-1] < results["accuracies"][0] - 0.1)


@pytest.mark.skipif(not _has_lightning, reason="PyTorch Lightning not installed")
class LightningWorkflow(TrainingWorkflow):
    """PyTorch Lightning training workflow for testing."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__("lightning", config)

    def run(self, backend_config: dict[str, Any]) -> dict[str, Any]:
        """Run a PyTorch Lightning training workflow."""
        import tracelet

        class LightningModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(10, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
                )
                self.criterion = nn.MSELoss()
                self.training_losses = []

            def forward(self, x):
                return self.model(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self.forward(x)
                loss = self.criterion(y_hat, y)

                # Log metrics
                self.log("train/batch_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
                self.log("train/epoch_loss", loss, on_step=False, on_epoch=True)

                return loss

            def validation_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self.forward(x)
                loss = self.criterion(y_hat, y)

                # Calculate R² score
                predictions = y_hat
                ss_tot = torch.sum((y - torch.mean(y)) ** 2)
                ss_res = torch.sum((y - predictions) ** 2)
                r2_score = 1 - ss_res / ss_tot

                self.log("val/loss", loss, on_epoch=True, prog_bar=True)
                self.log("val/accuracy_r2", r2_score, on_epoch=True, prog_bar=True)

                return loss

            def configure_optimizers(self):
                optimizer = optim.Adam(self.parameters(), lr=0.001)
                return optimizer

        # Generate synthetic data
        torch.manual_seed(42)
        X = torch.randn(1000, 10)
        y = torch.sum(X[:, :3], dim=1, keepdim=True) + 0.1 * torch.randn(1000, 1)

        # Split data
        train_size = int(0.8 * len(X))
        train_X, val_X = X[:train_size], X[train_size:]
        train_y, val_y = y[:train_size], y[train_size:]

        train_dataset = TensorDataset(train_X, train_y)
        val_dataset = TensorDataset(val_X, val_y)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        # Start experiment tracking
        exp = tracelet.start_logging(
            exp_name=f"lightning_{self.name}_{int(time.time())}",
            project=backend_config.get("project", "e2e_test"),
            backend=backend_config["backend"],
        )

        # Log hyperparameters
        hyperparams = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "max_epochs": 10,
            "model_layers": 3,
            "dropout": 0.1,
            "optimizer": "adam",
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        }

        exp.log_params(hyperparams)

        # Create model and trainer
        model = LightningModel()

        trainer = pl.Trainer(
            max_epochs=10,
            enable_checkpointing=True,
            logger=False,  # We use Tracelet for logging
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        # Train model
        trainer.fit(model, train_loader, val_loader)

        # Extract results
        results = {
            "epochs_completed": trainer.current_epoch + 1,
            "training_completed": True,
            "final_train_loss": float(trainer.logged_metrics.get("train/epoch_loss", 0.0)),
            "final_val_loss": float(trainer.logged_metrics.get("val/loss", 0.0)),
            "final_val_accuracy": float(trainer.logged_metrics.get("val/accuracy_r2", 0.0)),
        }

        # Log final metrics
        exp.log_metric("final/train_loss", results["final_train_loss"], results["epochs_completed"])
        exp.log_metric("final/val_loss", results["final_val_loss"], results["epochs_completed"])
        exp.log_metric("final/val_accuracy", results["final_val_accuracy"], results["epochs_completed"])

        tracelet.stop_logging()

        return results

    def get_expected_metrics(self) -> list[str]:
        return [
            "train/batch_loss",
            "train/epoch_loss",
            "val/loss",
            "val/accuracy_r2",
            "final/train_loss",
            "final/val_loss",
            "final/val_accuracy",
        ]

    def validate_results(self, results: dict[str, Any]) -> bool:
        """Validate Lightning training results."""
        if not results.get("training_completed", False):
            return False

        if results["epochs_completed"] != 10:
            return False

        # Check that we have reasonable final metrics (R² should be reasonable)
        return not (results["final_val_accuracy"] < 0)


class E2ETestFramework:
    """Main E2E test framework coordinator."""

    def __init__(self):
        self.environments = {"mlflow": MLflowEnvironment, "clearml": ClearMLEnvironment, "wandb": WandbEnvironment}

        self.workflows = {"simple_pytorch": SimplePyTorchWorkflow, "lightning": LightningWorkflow}

    def get_available_backends(self) -> list[str]:
        """Get list of available backend environments."""
        available = []
        for name, env_class in self.environments.items():
            env = env_class()
            if env.is_available():
                available.append(name)
        return available

    def get_available_workflows(self) -> list[str]:
        """Get list of available training workflows."""
        available = []
        for name, _workflow_class in self.workflows.items():
            # Check workflow dependencies
            if name == "simple_pytorch" and not _has_torch:
                continue
            if name == "lightning" and (not _has_torch or not _has_lightning):
                continue
            available.append(name)
        return available

    @contextlib.contextmanager
    def backend_environment(self, backend_name: str, config: Optional[dict[str, Any]] = None):
        """Context manager for backend environment setup/teardown."""
        if backend_name not in self.environments:
            msg = f"Unknown backend: {backend_name}"
            raise ValueError(msg)

        env_class = self.environments[backend_name]
        env = env_class(config)

        if not env.is_available():
            pytest.skip(f"{backend_name} backend not available")

        try:
            backend_config = env.setup()
            logger.info(f"Set up {backend_name} environment")
            yield backend_config
        finally:
            env.teardown()
            logger.info(f"Tore down {backend_name} environment")

    def run_workflow(
        self, workflow_name: str, backend_config: dict[str, Any], workflow_config: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Run a specific workflow with the given backend."""
        if workflow_name not in self.workflows:
            msg = f"Unknown workflow: {workflow_name}"
            raise ValueError(msg)

        workflow_class = self.workflows[workflow_name]
        workflow = workflow_class(workflow_config)

        logger.info(f"Running {workflow_name} workflow with {backend_config['backend']} backend")

        start_time = time.time()
        try:
            results = workflow.run(backend_config)
            results["execution_time"] = time.time() - start_time
            results["workflow_name"] = workflow_name
            results["backend_name"] = backend_config["backend"]
            results["success"] = workflow.validate_results(results)

            logger.info(
                f"Workflow {workflow_name} completed in {results['execution_time']:.2f}s: "
                f"{'SUCCESS' if results['success'] else 'FAILED'}"
            )
        except Exception as e:
            logger.exception(f"Workflow {workflow_name} failed")
            return {
                "execution_time": time.time() - start_time,
                "workflow_name": workflow_name,
                "backend_name": backend_config["backend"],
                "success": False,
                "error": str(e),
            }
        else:
            return results

    def run_comprehensive_test(
        self, backends: Optional[list[str]] = None, workflows: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """Run comprehensive E2E tests across multiple backends and workflows."""
        if backends is None:
            backends = self.get_available_backends()

        if workflows is None:
            workflows = self.get_available_workflows()

        results = {
            "test_summary": {
                "total_tests": len(backends) * len(workflows),
                "passed_tests": 0,
                "failed_tests": 0,
                "execution_time": 0,
            },
            "backend_results": {},
            "workflow_results": {},
        }

        start_time = time.time()

        for backend_name in backends:
            results["backend_results"][backend_name] = {}

            with self.backend_environment(backend_name) as backend_config:
                for workflow_name in workflows:
                    test_result = self.run_workflow(workflow_name, backend_config)

                    results["backend_results"][backend_name][workflow_name] = test_result

                    if workflow_name not in results["workflow_results"]:
                        results["workflow_results"][workflow_name] = {}
                    results["workflow_results"][workflow_name][backend_name] = test_result

                    if test_result["success"]:
                        results["test_summary"]["passed_tests"] += 1
                    else:
                        results["test_summary"]["failed_tests"] += 1

        results["test_summary"]["execution_time"] = time.time() - start_time

        logger.info(
            f"Comprehensive E2E tests completed: "
            f"{results['test_summary']['passed_tests']}/{results['test_summary']['total_tests']} passed "
            f"in {results['test_summary']['execution_time']:.2f}s"
        )

        return results


# Global framework instance
e2e_framework = E2ETestFramework()
