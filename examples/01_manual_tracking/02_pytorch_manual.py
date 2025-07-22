#!/usr/bin/env python3
"""
Manual PyTorch Training Example

This example shows how traditional PyTorch training requires extensive manual
logging for experiment tracking. Every metric, parameter, and model detail
must be explicitly logged.
"""

import random
import time

from tracelet import Experiment


def simulate_pytorch_training():
    """Simulate PyTorch training with manual experiment tracking."""
    print("🔧 MANUAL PYTORCH TRAINING EXAMPLE")
    print("=" * 60)
    print("This shows traditional PyTorch training with manual experiment tracking.")
    print("Every metric, parameter, and model detail requires explicit logging.")
    print()

    # Model hyperparameters
    learning_rate = 0.001
    batch_size = 64
    epochs = 10
    weight_decay = 1e-4
    dropout_rate = 0.3
    hidden_layers = [256, 128, 64]
    activation = "relu"
    optimizer_type = "adam"

    print("🏗️  Model Configuration:")
    print(f"   Architecture: {hidden_layers}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Dropout: {dropout_rate}")
    print(f"   Optimizer: {optimizer_type}")
    print()

    # Create experiment with manual tracking
    experiment = Experiment(
        name="manual_pytorch_training",
        backend=[],
        automagic=False,  # Manual tracking only
    )
    experiment.start()

    # MANUAL: Log all model hyperparameters
    print("📝 Manually logging model hyperparameters...")
    experiment.log_params({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "weight_decay": weight_decay,
        "dropout_rate": dropout_rate,
        "optimizer": optimizer_type,
        "activation": activation,
        "hidden_layer_1": hidden_layers[0],
        "hidden_layer_2": hidden_layers[1],
        "hidden_layer_3": hidden_layers[2],
        "total_layers": len(hidden_layers),
    })

    # MANUAL: Log model architecture details
    total_params = sum(hidden_layers) * 2 + 1000  # Simulated parameter count
    experiment.log_params({
        "model_type": "feedforward_neural_network",
        "total_parameters": total_params,
        "trainable_parameters": total_params,
        "input_size": 784,  # MNIST-like
        "output_size": 10,
        "parameter_count": total_params,
    })
    print("   ✅ Model architecture logged")

    # MANUAL: Log dataset information
    experiment.log_params({
        "dataset": "synthetic_classification",
        "train_samples": 50000,
        "val_samples": 10000,
        "test_samples": 10000,
        "num_classes": 10,
        "input_shape": [28, 28, 1],
    })
    print("   ✅ Dataset information logged")

    # Training loop with manual logging
    print("\n🚀 Starting manual PyTorch training simulation:")

    best_val_acc = 0
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        print(f"\n📊 Epoch {epoch + 1}/{epochs}")

        # Simulate training phase
        epoch_train_loss = 2.3 - (epoch * 0.15) + random.uniform(-0.1, 0.1)  # noqa: S311
        epoch_train_acc = 0.1 + (epoch * 0.08) + random.uniform(-0.02, 0.02)  # noqa: S311

        # MANUAL: Log training metrics
        experiment.log_metric("train_loss", epoch_train_loss, iteration=epoch)
        experiment.log_metric("train_accuracy", epoch_train_acc, iteration=epoch)

        # Simulate validation phase
        epoch_val_loss = epoch_train_loss + random.uniform(0.05, 0.15)  # noqa: S311
        epoch_val_acc = epoch_train_acc - random.uniform(0.01, 0.05)  # noqa: S311

        # MANUAL: Log validation metrics
        experiment.log_metric("val_loss", epoch_val_loss, iteration=epoch)
        experiment.log_metric("val_accuracy", epoch_val_acc, iteration=epoch)

        # MANUAL: Log learning rate (might change with schedulers)
        current_lr = learning_rate * (0.95**epoch)  # Simulated decay
        experiment.log_metric("learning_rate", current_lr, iteration=epoch)

        # MANUAL: Log gradient norms (simulated)
        grad_norm = random.uniform(0.5, 2.0)  # noqa: S311
        experiment.log_metric("gradient_norm", grad_norm, iteration=epoch)

        # MANUAL: Log system metrics
        gpu_memory = random.uniform(2.1, 2.5)  # GB  # noqa: S311
        epoch_time = random.uniform(45, 65)  # seconds  # noqa: S311
        experiment.log_metric("gpu_memory_gb", gpu_memory, iteration=epoch)
        experiment.log_metric("epoch_time_seconds", epoch_time, iteration=epoch)

        # MANUAL: Track best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            experiment.log_metric("best_val_accuracy", best_val_acc, iteration=epoch)
            # MANUAL: Log checkpoint info
            experiment.log_params({"best_model_epoch": epoch + 1})

        # MANUAL: Log additional training details
        experiment.log_metric("train_loss_std", random.uniform(0.1, 0.3), iteration=epoch)  # noqa: S311
        experiment.log_metric("batch_time_ms", random.uniform(120, 180), iteration=epoch)  # noqa: S311

        train_losses.append(epoch_train_loss)
        val_accuracies.append(epoch_val_acc)

        print(f"   Train: loss={epoch_train_loss:.4f}, acc={epoch_train_acc:.4f}")
        print(f"   Val:   loss={epoch_val_loss:.4f}, acc={epoch_val_acc:.4f}")

        time.sleep(0.1)  # Simulate training time

    # MANUAL: Log final training summary
    print("\n📊 Manually logging final training results...")
    experiment.log_params({
        "final_train_loss": train_losses[-1],
        "final_val_accuracy": val_accuracies[-1],
        "best_val_accuracy": best_val_acc,
        "convergence_epoch": len([acc for acc in val_accuracies if acc > 0.7]),
        "training_completed": True,
    })

    # MANUAL: Log model artifacts
    print("📁 Manually logging model artifacts...")
    experiment.log_artifact("best_model.pth", "path/to/best_model.pth")
    experiment.log_artifact("final_model.pth", "path/to/final_model.pth")
    experiment.log_artifact("training_history.json", "path/to/history.json")
    experiment.log_artifact("model_architecture.txt", "path/to/architecture.txt")

    # MANUAL: Log optimizer state
    experiment.log_params({
        "optimizer_state_dict": "saved",
        "scheduler_state_dict": "saved",
        "random_seed": 42,
        "torch_version": "2.0.0",
        "cuda_version": "11.8",
    })

    experiment.end()

    print("\n✅ Manual PyTorch training completed!")
    print("\n" + "=" * 60)
    print("📝 MANUAL PYTORCH TRACKING SUMMARY:")
    print("   ✅ 11 model hyperparameters logged manually")
    print("   ✅ 7 architecture parameters logged manually")
    print("   ✅ 6 dataset parameters logged manually")
    print("   ✅ 8 metrics x 10 epochs = 80 metric calls")
    print("   ✅ 8 final results logged manually")
    print("   ✅ 4 model artifacts logged manually")
    print("   ✅ 6 environment parameters logged manually")
    print("   📊 Total: ~120+ manual logging calls required!")
    print("\n💡 Compare with automagic_pytorch.py - same functionality, 1 line!")


if __name__ == "__main__":
    simulate_pytorch_training()
