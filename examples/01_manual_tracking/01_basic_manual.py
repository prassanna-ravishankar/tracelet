#!/usr/bin/env python3
"""
Basic Manual Experiment Tracking Example

This example shows traditional experiment tracking where you manually log
all parameters, metrics, and artifacts. This gives you complete control
but requires explicit logging calls for everything.
"""

import random
import time

from tracelet import Experiment


def manual_experiment():
    """Traditional manual experiment tracking approach."""
    print("🔧 MANUAL TRACKING EXAMPLE")
    print("=" * 50)
    print("This example shows traditional experiment tracking where")
    print("you must manually log every parameter and metric.")
    print()

    # 1. Define your hyperparameters
    learning_rate = 0.001
    batch_size = 32
    epochs = 5
    dropout_rate = 0.2
    hidden_size = 128
    optimizer_name = "adam"

    print("📋 Experiment Configuration:")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Dropout Rate: {dropout_rate}")
    print(f"   Hidden Size: {hidden_size}")
    print(f"   Optimizer: {optimizer_name}")
    print()

    # 2. Create experiment (no automagic)
    experiment = Experiment(
        name="manual_tracking_demo",
        backend=[],  # No backend for this demo
        automagic=False,  # Explicitly disable automagic
    )

    # 3. Start experiment
    experiment.start()
    print("✅ Experiment started")

    # 4. MANUAL: Log all hyperparameters explicitly
    print("📝 Manually logging hyperparameters...")
    experiment.log_params({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "dropout_rate": dropout_rate,
        "hidden_size": hidden_size,
        "optimizer": optimizer_name,
    })
    print("   ✅ Hyperparameters logged")

    # 5. MANUAL: Log additional configuration
    experiment.log_params({
        "model_type": "neural_network",
        "dataset": "synthetic",
        "experiment_type": "manual_tracking_demo",
    })
    print("   ✅ Configuration logged")

    # 6. MANUAL: Simulate training with explicit metric logging
    print("\n🚀 Training with manual metric logging:")

    best_accuracy = 0
    for epoch in range(epochs):
        # Simulate training
        epoch_loss = 1.0 - (epoch * 0.15) + random.uniform(-0.1, 0.1)  # noqa: S311
        epoch_accuracy = 0.5 + (epoch * 0.12) + random.uniform(-0.05, 0.05)  # noqa: S311

        # MANUAL: Log metrics for each epoch
        experiment.log_metric("loss", epoch_loss, iteration=epoch)
        experiment.log_metric("accuracy", epoch_accuracy, iteration=epoch)

        # MANUAL: Log learning rate (in real scenarios, this might change)
        experiment.log_metric("learning_rate", learning_rate, iteration=epoch)

        # MANUAL: Track best accuracy
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            experiment.log_metric("best_accuracy", best_accuracy, iteration=epoch)

        print(f"   Epoch {epoch + 1}/{epochs}: " f"loss={epoch_loss:.4f}, acc={epoch_accuracy:.4f}")

        # MANUAL: Log system metrics if needed
        experiment.log_metric("epoch_duration", random.uniform(2.5, 3.5), iteration=epoch)  # noqa: S311

        time.sleep(0.1)  # Simulate training time

    # 7. MANUAL: Log final results and artifacts
    print("\n📊 Manually logging final results...")
    experiment.log_params({
        "final_loss": epoch_loss,
        "final_accuracy": epoch_accuracy,
        "best_accuracy": best_accuracy,
        "total_epochs_completed": epochs,
    })

    # MANUAL: Log model artifacts (simulated)
    print("📁 Manually logging artifacts...")
    experiment.log_artifact("model.pth", "path/to/saved/model.pth")
    experiment.log_artifact("training_history.json", "path/to/history.json")

    # 8. End experiment
    experiment.end()
    print("\n✅ Manual tracking experiment completed!")

    # 9. Summary of manual effort
    print("\n" + "=" * 50)
    print("📝 MANUAL TRACKING SUMMARY:")
    print("   ✅ 6 hyperparameters logged manually")
    print("   ✅ 3 configuration parameters logged manually")
    print("   ✅ 4 metrics x 5 epochs = 20 metric calls")
    print("   ✅ 4 final results logged manually")
    print("   ✅ 2 artifacts logged manually")
    print("   📊 Total: ~30+ manual logging calls required")
    print("\n💡 Compare this with automagic_basic.py to see the difference!")


if __name__ == "__main__":
    manual_experiment()
