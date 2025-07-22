#!/usr/bin/env python3
"""
Example: Using Tracelet with ClearML Backend

This example demonstrates how to use Tracelet with the ClearML backend
for experiment tracking. ClearML offers a free SaaS platform at clearml.allegro.ai.

Prerequisites:
1. Install ClearML: pip install clearml
2. Set up ClearML credentials: clearml-init
   - Visit https://app.clearml.ai/settings/webapp-configuration
   - Copy your credentials to ~/.clearml.conf

Usage:
    python examples/clearml_example.py
"""

import time

import numpy as np

import tracelet


def train_model():
    """Simulate a simple training loop."""

    # Start logging with ClearML backend
    exp = tracelet.start_logging(exp_name="clearml_example", project="Tracelet Examples", backend="clearml")

    print("Started ClearML experiment tracking...")
    print(f"Experiment: {exp.name}")

    # Log hyperparameters
    hyperparams = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "optimizer": "adam",
        "model_type": "simple_cnn",
    }

    for name, value in hyperparams.items():
        exp.log_param(name, value)

    print("Logged hyperparameters")

    # Simulate training loop
    for epoch in range(5):  # Short demo
        # Simulate training metrics
        train_loss = 1.0 * np.exp(-epoch * 0.1) + 0.1 * np.random.random()
        train_acc = (1 - np.exp(-epoch * 0.2)) * 0.95 + 0.05 * np.random.random()

        # Simulate validation metrics
        val_loss = train_loss * 1.1 + 0.05 * np.random.random()
        val_acc = train_acc * 0.95 + 0.02 * np.random.random()

        # Log metrics
        exp.log_metric("train/loss", train_loss, epoch)
        exp.log_metric("train/accuracy", train_acc, epoch)
        exp.log_metric("val/loss", val_loss, epoch)
        exp.log_metric("val/accuracy", val_acc, epoch)

        # Log learning rate decay
        lr = hyperparams["learning_rate"] * (0.9**epoch)
        exp.log_metric("train/learning_rate", lr, epoch)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

        # Add some delay to simulate training time
        time.sleep(0.5)

    # Log some custom metrics
    exp.log_metric("final/best_val_accuracy", val_acc, epoch)
    exp.log_metric("final/total_epochs", epoch + 1, epoch)

    # Demonstrate enhanced visualization with fake data
    if hasattr(exp, "_framework") and exp._framework:
        import tempfile

        import matplotlib.pyplot as plt
        from torch.utils.tensorboard import SummaryWriter

        # Create a temporary tensorboard writer to trigger enhanced logging
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SummaryWriter(tmpdir)

            # Log histogram (simulated weights)
            weights = np.random.normal(0, 0.1, 1000)
            writer.add_histogram("model/weights", weights, epoch)

            # Log image (simulated training sample)
            fake_image = np.random.rand(3, 64, 64)
            writer.add_image("samples/input", fake_image, epoch)

            # Log text
            writer.add_text("training/notes", f"Completed training at epoch {epoch}", epoch)

            # Create and log a matplotlib figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

            # Plot loss curves
            epochs_range = list(range(epoch + 1))
            train_losses = [1.0 * np.exp(-i * 0.1) + 0.1 * np.random.random() for i in epochs_range]
            ax1.plot(epochs_range, train_losses, label="Training Loss")
            ax1.set_title("Training Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()

            # Plot accuracy curves
            train_accs = [(1 - np.exp(-i * 0.2)) * 0.95 + 0.05 * np.random.random() for i in epochs_range]
            ax2.plot(epochs_range, train_accs, label="Training Accuracy")
            ax2.set_title("Training Accuracy")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.legend()

            writer.add_figure("training/curves", fig, epoch)
            plt.close(fig)

            writer.close()

    print("Training completed!")

    # Stop logging
    tracelet.stop_logging()
    print("Experiment tracking stopped.")
    print("Check your ClearML dashboard at https://app.clearml.ai/ to view results!")


def main():
    """Main function."""
    print("Tracelet ClearML Integration Example")
    print("=" * 40)

    try:
        train_model()
    except ImportError as e:
        if "clearml" in str(e).lower():
            print("ClearML is not installed. Install with: pip install clearml")
            print("Then set up credentials with: clearml-init")
        else:
            raise
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have set up ClearML credentials with: clearml-init")


if __name__ == "__main__":
    main()
