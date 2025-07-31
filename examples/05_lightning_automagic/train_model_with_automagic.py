#!/usr/bin/env python3
"""
Train a PyTorch Lightning Model with Automatic Metric Tracking
==============================================================

This example shows how to use Tracelet's automagic feature with PyTorch Lightning.
When automagic=True, all your model.log() calls are automatically captured and
sent to your chosen backend (W&B, MLflow, ClearML, etc.) without any extra code!

Key Features:
- Zero-code metric tracking - just use self.log() as normal
- Works with any backend or multiple backends simultaneously
- Captures train/val metrics, learning rates, and more
- No need to modify your existing Lightning code

Usage:
    python train_model_with_automagic.py
"""

import contextlib
import os
import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

# Load environment variables
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset, random_split

from tracelet.core.experiment import Experiment, ExperimentConfig

load_dotenv()

# Force online mode
os.environ["WANDB_MODE"] = "online"


class SyntheticDataset(Dataset):
    """Create a synthetic dataset for testing."""

    def __init__(self, num_samples=1000, input_dim=20, num_classes=5):
        super().__init__()
        # Create synthetic data with some structure
        self.data = torch.randn(num_samples, input_dim)

        # Create labels with some pattern
        self.labels = torch.zeros(num_samples, dtype=torch.long)
        for i in range(num_samples):
            # Create class clusters
            class_idx = i % num_classes
            self.data[i] += torch.randn(input_dim) * 0.3 + class_idx * 0.5
            self.labels[i] = class_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class AutomagicModel(pl.LightningModule):
    """PyTorch Lightning model that should be captured by automagic."""

    def __init__(self, input_dim=20, hidden_dim=64, num_classes=5, learning_rate=0.01):
        super().__init__()
        self.save_hyperparameters()

        # Model architecture
        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes))

        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, y)

        # These should be captured by automagic!
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/accuracy", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)

        # These should be captured by automagic!
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


def test_automagic_lightning(backend_name: str):
    """Test Lightning with automagic using Experiment constructor."""
    print(f"\n{'=' * 70}")
    print(f"🔮 Testing {backend_name.upper()} with Automagic Lightning")
    print(f"{'=' * 70}")

    try:
        # Use Experiment constructor directly with automagic=True
        print("1. Creating experiment with automagic=True...")

        config = ExperimentConfig(
            track_metrics=True,
            track_environment=True,
            track_args=True,
            track_stdout=True,
            track_checkpoints=True,
        )

        exp = Experiment(
            name=f"automagic_{backend_name}_{int(time.time())}",
            config=config,
            backend=[backend_name],  # Backend expects a list of strings!
            tags=["project:Automagic-Lightning-Test"],
            automagic=True,  # ✨ THE MAGIC LINE! ✨
        )

        exp.start()
        print(f"   ✅ Automagic experiment started: {exp.name}")

        # Log some manual hyperparameters too
        print("2. Logging hyperparameters...")
        hparams = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 3,
            "optimizer": "adam",
            "model_type": "feedforward",
            "hidden_dim": 64,
            "num_classes": 5,
            "dataset_size": 1000,
            "automagic_enabled": True,
        }
        exp.log_params(hparams)
        print("   ✅ Parameters logged")

        # Create dataset and dataloaders
        print("3. Creating dataset and dataloaders...")
        dataset = SyntheticDataset(num_samples=1000)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

        print(f"   Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        # Create model
        print("4. Creating model...")
        model = AutomagicModel()
        print("   ✅ Model created")

        # Create trainer
        print("5. Training with Lightning (automagic should capture metrics)...")
        trainer = pl.Trainer(
            max_epochs=3,
            logger=False,  # Disable default logger, let automagic handle it
            enable_progress_bar=True,
            enable_checkpointing=False,
            log_every_n_steps=5,
        )

        # Train the model - automagic should capture all self.log() calls
        start_time = time.time()
        trainer.fit(model, train_loader, val_loader)
        training_time = time.time() - start_time

        print(f"   ✅ Training completed in {training_time:.2f}s")
        print("   📊 Automagic should have captured all Lightning metrics!")

        # Log final summary metrics manually
        print("6. Logging summary metrics...")
        exp.log_metric("summary/total_epochs", 3, trainer.global_step)
        exp.log_metric("summary/training_time", training_time, trainer.global_step)
        exp.log_metric("summary/total_steps", trainer.global_step, trainer.global_step)
        exp.log_metric("summary/automagic_test", 1.0, trainer.global_step)

        # Wait for metrics to flush
        print("7. Waiting for metrics to flush...")
        time.sleep(3)

        # Stop experiment
        print("8. Stopping experiment...")
        exp.stop()
        print("   ✅ Experiment stopped")

        return {
            "success": True,
            "backend": backend_name,
            "training_time": training_time,
            "total_steps": trainer.global_step,
        }

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback

        traceback.print_exc()
        with contextlib.suppress(Exception):
            exp.stop()
        return {"success": False, "backend": backend_name, "error": str(e)}


def main():
    """Test automagic Lightning with all backends."""
    print("🔮 AUTOMAGIC LIGHTNING TEST")
    print("Testing automagic=True with Lightning self.log() calls")

    backends = ["clearml", "wandb", "mlflow"]
    results = []

    for backend in backends:
        result = test_automagic_lightning(backend)
        results.append(result)

        # Pause between backends
        time.sleep(3)

    # Summary
    print(f"\n{'=' * 70}")
    print("📊 AUTOMAGIC LIGHTNING RESULTS")
    print(f"{'=' * 70}")

    print(f"\n{'Backend':<12} {'Status':<10} {'Time':<10} {'Steps'}")
    print("-" * 50)

    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        time_str = f"{result.get('training_time', 0):.1f}s" if result["success"] else "N/A"
        steps = f"{result.get('total_steps', 0)} steps" if result["success"] else result.get("error", "")[:20]

        print(f"{result['backend'].upper():<12} {status:<10} {time_str:<10} {steps}")

    if all(r["success"] for r in results):
        print("\n✅ All backends successfully tested with automagic!")
        print("\n📈 You should now see in each platform:")
        print("   - train/loss and train/accuracy (from Lightning self.log())")
        print("   - val/loss and val/accuracy (from Lightning self.log())")
        print("   - summary metrics (manual)")
        print("   - Clean parameter names")
        print("   - 🔮 ALL captured automatically by automagic!")

    print("\n🔍 View your results:")
    print("   ClearML: https://app.clear.ml/ → Automagic-Lightning-Test → Scalars")
    print("   W&B: https://wandb.ai/ → Automagic-Lightning-Test project → Charts")
    print("   MLflow: Databricks workspace → Automagic-Lightning-Test → Metrics")


if __name__ == "__main__":
    main()
