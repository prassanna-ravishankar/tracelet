"""Example of using AIM backend with Tracelet.

This example demonstrates:
1. Local and remote AIM repository usage
2. Metric logging with contexts
3. Distribution/histogram tracking
4. Image and text logging
5. Artifact management
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import tracelet


def create_sample_data(n_samples=1000):
    """Create sample classification data."""
    X = torch.randn(n_samples, 10)
    y = (X.sum(dim=1) > 0).long()
    return X, y


def create_model():
    """Create a simple neural network."""
    return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))


def train_with_aim_local():
    """Example using local AIM repository."""
    print("🎯 AIM Local Repository Example")

    # Start experiment tracking with local AIM backend
    exp = tracelet.start_logging(
        backend="aim",
        exp_name="aim_local_example",
        project="tracelet_demos",
        config={
            "repo_path": ".aim",  # Local repository path
            "tags": {"framework": "pytorch", "task": "classification", "environment": "local"},
        },
    )

    # Log hyperparameters
    hyperparams = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "optimizer": "Adam",
        "loss_function": "CrossEntropy",
    }
    for key, value in hyperparams.items():
        exp.log_param(key, value)

    # Create data and model
    X, y = create_sample_data()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = create_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop with comprehensive logging
    for epoch in range(10):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Log batch metrics with context
            if batch_idx % 5 == 0:
                exp.log_metric("batch_loss", loss.item(), epoch * len(dataloader) + batch_idx)

        # Log epoch metrics
        avg_loss = epoch_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        exp.log_metric("train/loss", avg_loss, epoch)
        exp.log_metric("train/accuracy", accuracy, epoch)

        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")

        # Log model weights as distributions (histograms)
        if epoch % 3 == 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Through TensorBoard integration, these will be logged as distributions
                    exp._framework.log_enhanced_metric(
                        name=f"weights/{name}",
                        value=param.data.cpu().numpy(),
                        metric_type=tracelet.core.orchestrator.MetricType.HISTOGRAM,
                        iteration=epoch,
                    )

    # Create and log a confusion matrix as an image
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Create confusion matrix plot
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    # Save and log the plot
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Log as artifact
    exp.log_artifact("confusion_matrix.png", "visualizations/confusion_matrix.png")

    # Log text summary
    summary_text = f"""
    Training completed successfully!
    Final metrics:
    - Loss: {avg_loss:.4f}
    - Accuracy: {accuracy:.2f}%
    - Total epochs: 10
    - Model parameters: {sum(p.numel() for p in model.parameters())}
    """

    # Log through enhanced metric system
    exp._framework.log_enhanced_metric(
        name="training_summary", value=summary_text, metric_type=tracelet.core.orchestrator.MetricType.TEXT, iteration=0
    )

    # Save model
    torch.save(model.state_dict(), "model.pth")
    exp.log_artifact("model.pth", "models/final_model.pth")

    # Stop tracking
    exp.stop()

    # Clean up
    os.remove("confusion_matrix.png")
    os.remove("model.pth")

    print("✅ Local AIM tracking completed! Check .aim directory for results.")
    print("   Run 'aim up' to view in AIM UI")


def train_with_aim_remote():
    """Example using remote AIM server."""
    print("\n🌐 AIM Remote Server Example")

    # Check if remote URI is configured
    remote_uri = os.getenv("AIM_REMOTE_URI", "http://localhost:53800")

    try:
        # Start experiment tracking with remote AIM backend
        exp = tracelet.start_logging(
            backend="aim",
            exp_name="aim_remote_example",
            project="tracelet_demos",
            config={
                "remote_uri": remote_uri,
                "run_name": "remote_training_run",
                "tags": {
                    "framework": "pytorch",
                    "task": "classification",
                    "environment": "remote",
                    "node": os.uname().nodename if hasattr(os, "uname") else "unknown",
                },
            },
        )

        # Simple training loop
        for i in range(20):
            loss = 1.0 / (i + 1) + np.random.normal(0, 0.01)
            accuracy = min(99.5, i * 5 + np.random.normal(0, 1))

            exp.log_metric("loss", loss, i)
            exp.log_metric("accuracy", accuracy, i)

            # Log learning rate schedule
            lr = 0.001 * (0.9 ** (i // 5))
            exp.log_metric("learning_rate", lr, i)

        # Log final parameters
        exp.log_param("final_loss", loss)
        exp.log_param("final_accuracy", accuracy)
        exp.log_param("total_steps", 20)

        exp.stop()
        print(f"✅ Remote AIM tracking completed! Check {remote_uri} for results.")

    except Exception as e:
        print(f"⚠️  Remote AIM example failed: {e}")
        print("   Make sure AIM server is running at", remote_uri)
        print("   Start it with: aim server --host 0.0.0.0 --port 53800")


if __name__ == "__main__":
    print("=" * 60)
    print("Tracelet AIM Backend Examples")
    print("=" * 60)

    # Run local example
    train_with_aim_local()

    # Run remote example (will fail gracefully if no server)
    train_with_aim_remote()

    print("\n" + "=" * 60)
    print("Examples completed!")
