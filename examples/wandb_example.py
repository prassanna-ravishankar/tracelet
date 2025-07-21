#!/usr/bin/env python3
"""
Example: Using Tracelet with Weights & Biases Backend

This example demonstrates how to use Tracelet with the W&B backend
for experiment tracking. W&B offers a generous free tier for individual users.

Prerequisites:
1. Install W&B: pip install wandb
2. Set up W&B account: wandb login
   - Visit https://wandb.ai/authorize
   - Copy your API key and run: wandb login

Usage:
    python examples/wandb_example.py
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import tracelet


def train_model():
    """Simulate a simple training loop."""
    import os

    # Set W&B to offline mode if no API key is configured
    if not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_MODE"] = "offline"
        print("W&B API key not found, running in offline mode...")

    # Start logging with W&B backend
    exp = tracelet.start_logging(exp_name="wandb_example", project="Tracelet Examples", backend="wandb")

    print("Started W&B experiment tracking...")
    print(f"Experiment: {exp.name}")

    # Log hyperparameters
    hyperparams = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "optimizer": "adam",
        "model_type": "simple_cnn",
        "dropout": 0.2,
    }

    exp.log_params(hyperparams)

    print("Logged hyperparameters")

    # Simulate training loop with more realistic metrics
    best_val_acc = 0.0
    for epoch in range(10):  # Longer demo than other examples
        # Simulate training metrics with some realistic patterns
        base_loss = 2.0 * np.exp(-epoch * 0.15)  # Decay faster initially
        train_loss = base_loss + 0.1 * np.random.random()

        # Accuracy that improves over time with some noise
        base_acc = 0.95 * (1 - np.exp(-epoch * 0.25))
        train_acc = base_acc + 0.05 * np.random.random()

        # Validation metrics (slightly worse than training)
        val_loss = train_loss * 1.05 + 0.03 * np.random.random()
        val_acc = train_acc * 0.98 + 0.02 * np.random.random()

        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Log metrics with W&B
        exp.log_metric("train/loss", train_loss, epoch)
        exp.log_metric("train/accuracy", train_acc, epoch)
        exp.log_metric("val/loss", val_loss, epoch)
        exp.log_metric("val/accuracy", val_acc, epoch)
        exp.log_metric("val/best_accuracy", best_val_acc, epoch)

        # Log learning rate decay
        lr = hyperparams["learning_rate"] * (0.95**epoch)
        exp.log_metric("train/learning_rate", lr, epoch)

        # Log gradient norm (simulated)
        grad_norm = 1.5 * np.exp(-epoch * 0.1) + 0.2 * np.random.random()
        exp.log_metric("train/grad_norm", grad_norm, epoch)

        print(f"Epoch {epoch:2d}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}, lr={lr:.6f}")

        # Add some delay to simulate training time
        time.sleep(0.3)

    # Log final metrics
    exp.log_metric("final/best_val_accuracy", best_val_acc, epoch)
    exp.log_metric("final/total_epochs", epoch + 1, epoch)
    exp.log_metric("final/final_train_loss", train_loss, epoch)

    # Demonstrate enhanced visualization with fake data
    if hasattr(exp, "_framework") and exp._framework:
        import tempfile

        from torch.utils.tensorboard import SummaryWriter

        # Create a temporary tensorboard writer to trigger enhanced logging
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SummaryWriter(tmpdir)

            # Log histogram (simulated layer weights)
            layer_weights = np.random.normal(0, 0.1, 1000)
            writer.add_histogram("model/layer1_weights", layer_weights, epoch)

            # Log another histogram (gradients)
            gradients = np.random.normal(0, 0.02, 500)
            writer.add_histogram("gradients/layer1_grad", gradients, epoch)

            # Log image (simulated training sample)
            fake_image = np.random.rand(3, 64, 64)
            writer.add_image("samples/training_input", fake_image, epoch)

            # Log text summary
            summary_text = f"""
            Training Summary - Epoch {epoch}:
            - Best Validation Accuracy: {best_val_acc:.4f}
            - Final Learning Rate: {lr:.6f}
            - Training converged successfully
            - Model ready for evaluation
            """
            writer.add_text("training/summary", summary_text, epoch)

            # Create and log training curves figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

            # Plot loss curves
            epochs_range = list(range(epoch + 1))
            train_losses = [2.0 * np.exp(-i * 0.15) + 0.1 * np.random.random() for i in epochs_range]
            val_losses = [loss * 1.05 + 0.03 * np.random.random() for loss in train_losses]

            ax1.plot(epochs_range, train_losses, label="Training Loss", color="blue")
            ax1.plot(epochs_range, val_losses, label="Validation Loss", color="red")
            ax1.set_title("Loss Curves")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True)

            # Plot accuracy curves
            train_accs = [0.95 * (1 - np.exp(-i * 0.25)) + 0.05 * np.random.random() for i in epochs_range]
            val_accs = [acc * 0.98 + 0.02 * np.random.random() for acc in train_accs]

            ax2.plot(epochs_range, train_accs, label="Training Accuracy", color="blue")
            ax2.plot(epochs_range, val_accs, label="Validation Accuracy", color="red")
            ax2.set_title("Accuracy Curves")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.legend()
            ax2.grid(True)

            # Plot learning rate decay
            lrs = [hyperparams["learning_rate"] * (0.95**i) for i in epochs_range]
            ax3.plot(epochs_range, lrs, label="Learning Rate", color="green")
            ax3.set_title("Learning Rate Schedule")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Learning Rate")
            ax3.legend()
            ax3.grid(True)
            ax3.set_yscale("log")

            # Plot gradient norms
            grad_norms = [1.5 * np.exp(-i * 0.1) + 0.2 * np.random.random() for i in epochs_range]
            ax4.plot(epochs_range, grad_norms, label="Gradient Norm", color="purple")
            ax4.set_title("Gradient Norms")
            ax4.set_xlabel("Epoch")
            ax4.set_ylabel("Gradient Norm")
            ax4.legend()
            ax4.grid(True)

            plt.tight_layout()
            writer.add_figure("training/comprehensive_curves", fig, epoch)
            plt.close(fig)

            # Log hyperparameters with metrics for W&B sweep compatibility
            hparam_dict = hyperparams
            metric_dict = {"best_val_accuracy": best_val_acc, "final_train_loss": train_loss, "total_epochs": epoch + 1}
            writer.add_hparams(hparam_dict, metric_dict)

            writer.close()

    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Stop logging
    tracelet.stop_logging()
    print("Experiment tracking stopped.")
    print("Check your W&B dashboard at https://wandb.ai/ to view results!")


def main():
    """Main function."""
    print("Tracelet Weights & Biases Integration Example")
    print("=" * 45)

    try:
        train_model()
    except ImportError as e:
        if "wandb" in str(e).lower():
            print("W&B is not installed. Install with: pip install wandb")
            print("Then authenticate with: wandb login")
        else:
            raise
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have authenticated with W&B using: wandb login")


if __name__ == "__main__":
    main()
