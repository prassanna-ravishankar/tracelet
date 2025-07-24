#!/usr/bin/env python3
"""
Automagic PyTorch Lightning Training Example

This example demonstrates how Tracelet's automagic instrumentation works with
PyTorch Lightning, automatically capturing hyperparameters, model architecture,
and training metrics with minimal manual intervention.

Works great on Mac (CPU-only) without NVIDIA dependencies!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    import pytorch_lightning as lightning
    import pytorch_lightning as pl

    HAS_LIGHTNING = True
except ImportError:
    print("PyTorch Lightning not installed. Install with: pip install pytorch-lightning")
    HAS_LIGHTNING = False

from tracelet import Experiment
from tracelet.automagic import AutomagicConfig


class SimpleLightningModel(pl.LightningModule):
    """Simple PyTorch Lightning model for demonstration."""

    def __init__(self, input_size, hidden_size, num_classes, learning_rate, dropout_rate):
        super().__init__()

        # These will be captured by automagic as model attributes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        # Model architecture
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_size // 2, num_classes),
        )

        # Save hyperparameters (Lightning feature + automagic will capture these)
        self.save_hyperparameters()

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # Lightning logs these automatically, automagic captures them
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_loss_step", loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # Calculate accuracy
        acc = (y_hat.argmax(dim=1) == y).float().mean()

        # Lightning logs, automagic captures
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        # Automagic will capture optimizer config through hooks
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


def create_dummy_data(num_samples, input_size, num_classes):
    """Create dummy classification data for demonstration."""
    torch.manual_seed(42)  # For reproducibility

    # Generate random features
    X = torch.randn(num_samples, input_size)

    # Generate labels with some structure (not completely random)
    # Create clusters to make the classification task learnable
    cluster_centers = torch.randn(num_classes, input_size) * 2
    y = torch.zeros(num_samples, dtype=torch.long)

    for i in range(num_samples):
        # Assign to closest cluster with some noise
        distances = torch.norm(X[i : i + 1] - cluster_centers, dim=1)
        y[i] = distances.argmin()

        # Add some noise to make it more realistic
        if torch.rand(1) < 0.1:  # 10% noise
            y[i] = torch.randint(0, num_classes, (1,))

    return X, y


def automagic_lightning_training():
    """Demonstrate automagic tracking with PyTorch Lightning."""
    print("⚡ AUTOMAGIC PYTORCH LIGHTNING EXAMPLE")
    print("=" * 60)
    print("Mac-friendly CPU-only training with automagic tracking!")
    print()

    if not HAS_LIGHTNING:
        print("❌ PyTorch Lightning not available. Please install:")
        print("   pip install pytorch-lightning")
        return

    # Training hyperparameters (automatically captured by automagic!)
    input_size = 20
    hidden_size = 64
    num_classes = 5
    learning_rate = 0.001
    batch_size = 32
    max_epochs = 5
    dropout_rate = 0.2

    # Data parameters (also captured)
    num_train_samples = 1000
    num_val_samples = 200

    # Training configuration
    accelerator = "cpu"  # Perfect for Mac!
    precision = 32
    gradient_clip_val = 1.0

    print("🏗️  Lightning Configuration:")
    print(f"   Input Size: {input_size}, Hidden: {hidden_size}, Classes: {num_classes}")
    print(f"   Learning Rate: {learning_rate}, Batch Size: {batch_size}")
    print(f"   Max Epochs: {max_epochs}, Dropout: {dropout_rate}")
    print(f"   Accelerator: {accelerator} (Mac-friendly!)")
    print(f"   Data: {num_train_samples} train, {num_val_samples} val samples")
    print()

    # 🔮 THE AUTOMAGIC LINE - captures ALL hyperparameters above!
    automagic_config = AutomagicConfig(
        detect_function_args=True,
        detect_class_attributes=True,
        track_model_architecture=True,
        track_model_gradients=False,  # Keep it lightweight
        monitor_gpu_memory=False,  # Not needed on Mac CPU
        monitor_cpu_usage=True,  # Monitor CPU instead
        frameworks={"pytorch", "lightning"},
    )

    experiment = Experiment(
        name="automagic_lightning_training", backend=["clearml"], automagic=True, automagic_config=automagic_config
    )

    print("🔮 Starting experiment with automagic=True...")
    experiment.start()
    print("✅ All hyperparameters automatically captured!")

    # Create dummy data
    print("\n📊 Creating dummy classification data...")
    X_train, y_train = create_dummy_data(num_train_samples, input_size, num_classes)
    X_val, y_val = create_dummy_data(num_val_samples, input_size, num_classes)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create Lightning model (automagic captures architecture)
    print("\n🏗️  Creating Lightning model...")
    model = SimpleLightningModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
    )

    # Capture model info with automagic
    model_info = experiment.capture_model(model)
    print(f"📊 Model captured: {model_info.get('parameter_count', 'N/A')} parameters")

    # Configure Lightning trainer (Mac-friendly settings)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
        enable_checkpointing=False,  # Skip checkpointing for demo
        logger=False,  # Use tracelet instead of Lightning's default logger
    )

    print(f"\n⚡ Starting Lightning training for {max_epochs} epochs...")
    print("   (All metrics automatically captured via Lightning + automagic hooks)")

    # Train the model - automagic captures everything!
    trainer.fit(model, train_loader, val_loader)

    print("\n📊 Training completed! Getting final metrics...")

    # Get final validation metrics
    final_results = trainer.validate(model, val_loader, verbose=False)
    if final_results and len(final_results) > 0:
        final_val_loss = final_results[0].get("val_loss", 0)
        final_val_acc = final_results[0].get("val_acc", 0)

        experiment.log_metric("final_val_loss", final_val_loss)
        experiment.log_metric("final_val_accuracy", final_val_acc)

        print("📈 Final Results:")
        print(f"   Validation Loss: {final_val_loss:.4f}")
        print(f"   Validation Accuracy: {final_val_acc:.4f}")

    # Capture final model state
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    experiment.log_params({
        "training_completed": True,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "lightning_version": lightning.__version__,
        "pytorch_version": torch.__version__,
    })

    experiment.end()

    print("\n✅ Automagic Lightning training completed!")
    print("\n" + "=" * 60)
    print("⚡ LIGHTNING + AUTOMAGIC BENEFITS:")
    print("   🔮 Automatic hyperparameter capture from function scope")
    print("   🏗️  Model architecture automatically logged")
    print("   📊 Lightning metrics captured via automagic hooks")
    print("   💻 Mac-friendly CPU training")
    print("   🚀 Minimal manual logging (just final results)")
    print("   ⚡ Lightning's built-in logging + Tracelet's automagic")
    print("   🎯 Focus on model development, not experiment tracking")


def show_lightning_integration_benefits():
    """Show the benefits of Lightning + Automagic integration."""
    print("\n⚡ PYTORCH LIGHTNING + AUTOMAGIC INTEGRATION")
    print("=" * 70)

    print("🎯 PERFECT COMBINATION:")
    print("   Lightning: Simplifies PyTorch training loops")
    print("   Automagic: Simplifies experiment tracking")
    print("   Together: Maximum productivity!")

    print("\n🔮 WHAT AUTOMAGIC CAPTURES FROM LIGHTNING:")
    print("   📊 All hyperparameters from local variables")
    print("   🏗️  Model architecture via model hooks")
    print("   📈 Training/validation metrics via Lightning logs")
    print("   ⚡ Optimizer configuration via Lightning hooks")
    print("   📋 Lightning trainer configuration")
    print("   🖥️  System resources (CPU on Mac)")
    print("   📦 Framework versions (Lightning + PyTorch)")

    print("\n💻 MAC-FRIENDLY FEATURES:")
    print("   🍎 CPU-only training (perfect for MacBook)")
    print("   🚫 No NVIDIA dependencies required")
    print("   📊 CPU monitoring instead of GPU")
    print("   ⚡ Fast training on modern Mac CPUs")
    print("   🔋 Battery-efficient for laptop development")

    print("\n🏆 DEVELOPMENT WORKFLOW:")
    print("   1. Define hyperparameters normally")
    print("   2. Create Lightning module normally")
    print("   3. Add ONE line: Experiment(automagic=True)")
    print("   4. Train with Lightning as usual")
    print("   5. Everything tracked automatically!")

    print("\n📊 WHAT YOU GET:")
    print("   🎯 Complete experiment reproducibility")
    print("   📈 Automatic metric visualization")
    print("   🔍 Hyperparameter comparison across runs")
    print("   📱 Web dashboard via ClearML/WandB")
    print("   🚀 Zero overhead experiment tracking")


if __name__ == "__main__":
    automagic_lightning_training()
    show_lightning_integration_benefits()
