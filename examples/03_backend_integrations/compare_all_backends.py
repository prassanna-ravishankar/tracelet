#!/usr/bin/env python3
"""
Comprehensive ML Tracking Test Suite
=====================================

This script tests all three ML tracking backends (ClearML, W&B, MLflow/Databricks)
with real PyTorch Lightning training runs to verify end-to-end functionality.

Features tested:
- PyTorch Lightning integration
- Hyperparameter logging
- Metric tracking (train/val/test)
- Model checkpointing
- System metrics
- TensorBoard integration
- Automagic instrumentation

Requirements:
- ClearML configured via ~/clearml.conf
- W&B API key in .env
- Databricks/MLflow configured in .env
- PyTorch Lightning installed

Usage:
    uv run python test_comprehensive_ml_tracking.py [--backend all|clearml|wandb|mlflow]
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset, random_split

import tracelet


class SimpleClassificationModel(LightningModule):
    """
    PyTorch Lightning model for classification tasks.
    Includes comprehensive logging and realistic training dynamics.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: Optional[list[int]] = None,
        num_classes: int = 10,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.2,
        weight_decay: float = 1e-4,
        optimizer_name: str = "adam",
        scheduler_name: str = "cosine",
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model architecture
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]

        layers = []
        prev_size = input_size

        for _i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics tracking
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        # For logging purposes
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, y)

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/accuracy", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)

        # Log gradient norms periodically
        if batch_idx % 100 == 0:
            total_norm = 0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1.0 / 2)
            self.log("train/grad_norm", total_norm, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)

        # Store for epoch-end logging
        self.validation_step_outputs.append({"val_loss": loss, "val_accuracy": acc, "batch_size": x.size(0)})

        return loss

    def on_validation_epoch_end(self):
        # Calculate epoch metrics
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in self.validation_step_outputs]).mean()

        # Log epoch metrics
        self.log("val/loss", avg_loss, prog_bar=True)
        self.log("val/accuracy", avg_acc, prog_bar=True)

        # Clear for next epoch
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.test_accuracy(preds, y)

        self.test_step_outputs.append({"test_loss": loss, "test_accuracy": acc})

        return loss

    def on_test_epoch_end(self):
        avg_loss = torch.stack([x["test_loss"] for x in self.test_step_outputs]).mean()
        avg_acc = torch.stack([x["test_accuracy"] for x in self.test_step_outputs]).mean()

        self.log("test/loss", avg_loss)
        self.log("test/accuracy", avg_acc)

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        # Optimizer
        if self.hparams.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.hparams.learning_rate, momentum=0.9, weight_decay=self.hparams.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer_name}")

        # Scheduler
        if self.hparams.scheduler_name.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"}}
        elif self.hparams.scheduler_name.lower() == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, verbose=True
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"}}
        else:
            return optimizer


class SyntheticDataModule(LightningDataModule):
    """
    Lightning DataModule for synthetic classification data.
    Creates realistic multi-class classification dataset.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        input_size: int = 784,
        num_classes: int = 10,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Generate synthetic data
        torch.manual_seed(42)
        np.random.seed(42)

        # Create class-specific patterns
        data = []
        labels = []

        for class_idx in range(num_classes):
            samples_per_class = num_samples // num_classes

            # Create class-specific mean and covariance
            class_mean = torch.randn(input_size) * 0.5
            class_mean[class_idx * (input_size // num_classes) : (class_idx + 1) * (input_size // num_classes)] += 2.0

            # Generate samples for this class
            class_data = torch.randn(samples_per_class, input_size) * 0.8 + class_mean
            class_labels = torch.full((samples_per_class,), class_idx, dtype=torch.long)

            data.append(class_data)
            labels.append(class_labels)

        # Combine and shuffle
        self.data = torch.cat(data, dim=0)
        self.labels = torch.cat(labels, dim=0)

        # Shuffle
        indices = torch.randperm(len(self.data))
        self.data = self.data[indices]
        self.labels = self.labels[indices]

    def setup(self, stage: Optional[str] = None):
        # Create dataset
        dataset = TensorDataset(self.data, self.labels)

        # Split into train/val/test
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.2 * total_size)
        test_size = total_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
        )


def setup_environment_for_backend(backend_name: str):
    """Configure environment variables for specific backends."""
    if backend_name == "wandb":
        # W&B should use the API key from .env
        if not os.environ.get("WANDB_API_KEY"):
            print("⚠️  Warning: WANDB_API_KEY not found in environment")
            # Don't set to offline mode - let it fail if no key

    elif backend_name == "clearml":
        # ClearML should use ~/clearml.conf
        clearml_conf = Path.home() / "clearml.conf"
        if not clearml_conf.exists():
            print("⚠️  Warning: ~/clearml.conf not found")

    elif backend_name == "mlflow":
        # MLflow/Databricks should use environment variables from .env
        required_vars = ["DATABRICKS_TOKEN", "DATABRICKS_HOST", "MLFLOW_TRACKING_URI"]
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        if missing_vars:
            print(f"⚠️  Warning: Missing MLflow environment variables: {missing_vars}")


def run_training_experiment(
    backend_name: str, experiment_name: str, max_epochs: int = 15, fast_dev_run: bool = False
) -> dict:
    """
    Run a complete training experiment with specified backend.

    Returns:
        Dictionary with experiment results and metrics
    """
    print(f"\n{'=' * 80}")
    print(f"🚀 Starting {backend_name.upper()} Integration Test")
    print(f"{'=' * 80}")

    # Setup environment
    setup_environment_for_backend(backend_name)

    # Hyperparameters for this experiment
    hparams = {
        "backend": backend_name,
        "input_size": 784,
        "hidden_sizes": [512, 256, 128],
        "num_classes": 10,
        "learning_rate": 1e-3,
        "dropout_rate": 0.3,
        "weight_decay": 1e-4,
        "optimizer": "adam",
        "scheduler": "cosine",
        "batch_size": 64,
        "max_epochs": max_epochs,
        "num_samples": 10000,
        "model_type": "feedforward_nn",
        "train_samples": 7000,
        "val_samples": 2000,
        "test_samples": 1000,
    }

    # Start Tracelet experiment tracking
    try:
        experiment = tracelet.start_logging(
            exp_name=experiment_name,
            project="ML-Tracking-Tests",
            backend=[backend_name],
            config={
                "track_system": True,
                "track_git": True,
                "track_tensorboard": True,
                "track_lightning": True,
                "metrics_interval": 5.0,
            },
        )
        print(f"✅ Started Tracelet experiment: {experiment.name}")

        # Log hyperparameters
        experiment.log_params(hparams)
        print("✅ Logged hyperparameters")

    except Exception as e:
        print(f"❌ Failed to start Tracelet experiment: {e}")
        return {"success": False, "error": str(e)}

    # Create data module
    data_module = SyntheticDataModule(
        num_samples=hparams["num_samples"],
        input_size=hparams["input_size"],
        num_classes=hparams["num_classes"],
        batch_size=hparams["batch_size"],
        num_workers=2,  # Reduced for stability
    )

    # Create model
    model = SimpleClassificationModel(
        input_size=hparams["input_size"],
        hidden_sizes=hparams["hidden_sizes"],
        num_classes=hparams["num_classes"],
        learning_rate=hparams["learning_rate"],
        dropout_rate=hparams["dropout_rate"],
        weight_decay=hparams["weight_decay"],
        optimizer_name=hparams["optimizer"],
        scheduler_name=hparams["scheduler"],
    )

    # Create checkpoint directory
    checkpoint_dir = Path(f"./checkpoints/{backend_name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{experiment_name}-{{epoch:02d}}-{{val/loss:.2f}}",
            save_top_k=3,
            monitor="val/loss",
            mode="min",
            save_last=True,
        ),
        EarlyStopping(monitor="val/loss", patience=7, mode="min", min_delta=0.001),
    ]

    # TensorBoard logger
    tb_logger = TensorBoardLogger(save_dir="./logs", name=f"{backend_name}_experiment", version=experiment_name)

    # Create trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=tb_logger,
        accelerator="auto",
        devices=1,
        fast_dev_run=fast_dev_run,
        log_every_n_steps=20,
        val_check_interval=0.5,  # Check validation every half epoch
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    print("✅ Created PyTorch Lightning trainer")
    print(f"   - Max epochs: {max_epochs}")
    print(f"   - Accelerator: {trainer.accelerator}")
    print(f"   - Fast dev run: {fast_dev_run}")

    # Start training
    start_time = time.time()
    try:
        print("\n🏃‍♂️ Starting training...")
        trainer.fit(model, data_module)
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f} seconds")

        # Run test
        print("🧪 Running test evaluation...")
        test_results = trainer.test(model, data_module)
        test_metrics = test_results[0] if test_results else {}

        # Log final metrics to Tracelet
        final_metrics = {
            "final/training_time": training_time,
            "final/total_epochs": trainer.current_epoch + 1,
            "final/test_accuracy": test_metrics.get("test/accuracy", 0.0),
            "final/test_loss": test_metrics.get("test/loss", 0.0),
            "final/best_val_loss": trainer.callback_metrics.get("val/loss", 0.0),
            "final/best_val_accuracy": trainer.callback_metrics.get("val/accuracy", 0.0),
        }

        for metric_name, metric_value in final_metrics.items():
            experiment.log_metric(metric_name, metric_value, trainer.current_epoch)

        print("✅ Logged final metrics")

        # Custom tags for organization
        experiment.add_tag(f"backend:{backend_name}")
        experiment.add_tag("framework:pytorch_lightning")
        experiment.add_tag(f"epochs:{trainer.current_epoch + 1}")

        results = {
            "success": True,
            "backend": backend_name,
            "training_time": training_time,
            "epochs_completed": trainer.current_epoch + 1,
            "test_accuracy": test_metrics.get("test/accuracy", 0.0),
            "test_loss": test_metrics.get("test/loss", 0.0),
            "val_accuracy": trainer.callback_metrics.get("val/accuracy", 0.0),
            "val_loss": trainer.callback_metrics.get("val/loss", 0.0),
            "model_checkpoint": str(checkpoint_dir / f"{experiment_name}-last.ckpt"),
        }

    except Exception as e:
        training_time = time.time() - start_time
        print(f"❌ Training failed after {training_time:.2f} seconds: {e}")
        results = {"success": False, "backend": backend_name, "error": str(e), "training_time": training_time}

    finally:
        # Stop Tracelet tracking
        try:
            tracelet.stop_logging()
            print("✅ Stopped Tracelet experiment tracking")
        except Exception as e:
            print(f"⚠️  Warning: Failed to stop Tracelet tracking: {e}")

    return results


def print_results_summary(all_results: list[dict]):
    """Print a comprehensive summary of all test results."""
    print(f"\n{'=' * 80}")
    print("📊 COMPREHENSIVE ML TRACKING TEST RESULTS")
    print(f"{'=' * 80}")

    successful_runs = [r for r in all_results if r.get("success", False)]
    failed_runs = [r for r in all_results if not r.get("success", False)]

    print(f"\n✅ Successful runs: {len(successful_runs)}/{len(all_results)}")
    print(f"❌ Failed runs: {len(failed_runs)}/{len(all_results)}")

    if successful_runs:
        print("\n🏆 PERFORMANCE SUMMARY:")
        print(f"{'Backend':<12} {'Epochs':<8} {'Time(s)':<10} {'Test Acc':<10} {'Val Acc':<10}")
        print("-" * 60)

        for result in successful_runs:
            print(
                f"{result['backend'].upper():<12} "
                f"{result.get('epochs_completed', 0):<8} "
                f"{result.get('training_time', 0):<10.1f} "
                f"{result.get('test_accuracy', 0):<10.3f} "
                f"{result.get('val_accuracy', 0):<10.3f}"
            )

        # Best performers
        best_accuracy = max(successful_runs, key=lambda x: x.get("test_accuracy", 0))
        fastest_training = min(successful_runs, key=lambda x: x.get("training_time", float("inf")))

        print(
            f"\n🎯 Best Test Accuracy: {best_accuracy['backend'].upper()} ({best_accuracy.get('test_accuracy', 0):.3f})"
        )
        print(
            f"⚡ Fastest Training: {fastest_training['backend'].upper()} ({fastest_training.get('training_time', 0):.1f}s)"
        )

    if failed_runs:
        print("\n❌ FAILED RUNS:")
        for result in failed_runs:
            backend = result.get("backend", "Unknown")
            error = result.get("error", "Unknown error")
            print(f"   {backend.upper()}: {error}")

    print(f"\n{'=' * 80}")

    # Where to view results
    print("🔍 VIEW YOUR RESULTS:")
    if any(r["backend"] == "clearml" and r.get("success") for r in all_results):
        print("   ClearML: https://app.clearml.ai/")
    if any(r["backend"] == "wandb" and r.get("success") for r in all_results):
        print("   W&B: https://wandb.ai/")
    if any(r["backend"] == "mlflow" and r.get("success") for r in all_results):
        print("   MLflow: Check your Databricks workspace or local MLflow UI")


def main():
    """Main function to run comprehensive ML tracking tests."""
    parser = argparse.ArgumentParser(description="Comprehensive ML Tracking Test Suite")
    parser.add_argument(
        "--backend", choices=["all", "clearml", "wandb", "mlflow"], default="all", help="Which backend(s) to test"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Maximum epochs for training")
    parser.add_argument("--fast-dev-run", action="store_true", help="Run in fast development mode (1 batch per epoch)")

    args = parser.parse_args()

    print("🌟 Comprehensive ML Tracking Test Suite")
    print("Testing ClearML, W&B, and MLflow/Databricks integrations")
    print("with PyTorch Lightning on macOS")

    # Determine which backends to test
    backends_to_test = ["clearml", "wandb", "mlflow"] if args.backend == "all" else [args.backend]

    print(f"\n🎯 Testing backends: {', '.join(b.upper() for b in backends_to_test)}")
    print(f"📊 Max epochs per experiment: {args.epochs}")
    print(f"⚡ Fast dev run: {args.fast_dev_run}")

    # Run experiments
    all_results = []
    total_start_time = time.time()

    for backend in backends_to_test:
        experiment_name = f"{backend}_lightning_test_{int(time.time())}"

        try:
            result = run_training_experiment(
                backend_name=backend,
                experiment_name=experiment_name,
                max_epochs=args.epochs,
                fast_dev_run=args.fast_dev_run,
            )
            all_results.append(result)

            # Brief pause between experiments
            if len(backends_to_test) > 1:
                print("\n⏳ Pausing 5 seconds before next experiment...")
                time.sleep(5)

        except Exception as e:
            print(f"❌ Unexpected error testing {backend}: {e}")
            all_results.append({"success": False, "backend": backend, "error": f"Unexpected error: {e!s}"})

    total_time = time.time() - total_start_time
    print(f"\n⏱️  Total test suite execution time: {total_time:.1f} seconds")

    # Print comprehensive results
    print_results_summary(all_results)

    # Return exit code based on results
    successful_count = sum(1 for r in all_results if r.get("success", False))
    if successful_count == len(all_results):
        print(f"\n🎉 All {len(all_results)} experiments completed successfully!")
        return 0
    else:
        print(f"\n⚠️  {len(all_results) - successful_count} out of {len(all_results)} experiments failed")
        return 1


if __name__ == "__main__":
    exit(main())
