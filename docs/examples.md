# Examples

This page contains examples of how to use Tracelet.

## Basic Usage

```python
import tracelet
import torch
from torch.utils.tensorboard import SummaryWriter

# Start experiment tracking with your preferred backend
tracelet.start_logging(
    exp_name="my_experiment",
    project="my_project",
    backend="mlflow"  # or "clearml", "wandb", "aim"
)

# Use TensorBoard as usual - metrics are automatically captured
writer = SummaryWriter()
for epoch in range(100):
    loss = 0.9**epoch
    writer.add_scalar('Loss/train', loss, epoch)
    # Metrics are automatically sent to MLflow!

# Stop tracking when done
tracelet.stop_logging()
```

## Multi-Backend Comparison

```python
#!/usr/bin/env python3
"""
Example: Multi-Backend Comparison with Tracelet

This example demonstrates how to run the same experiment across multiple backends
(MLflow, ClearML, and W&B) to compare their capabilities and performance.

Prerequisites:
- MLflow: pip install mlflow (works out of the box)
- ClearML: pip install clearml (works in offline mode)
- W&B: pip install wandb (works in offline mode if no API key)

Usage:
    python examples/multi_backend_comparison.py
"""

import importlib.util
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import tracelet


class SimpleNN(nn.Module):
    """Simple neural network for demonstration."""

    def __init__(self, input_size=20, hidden_size=64, output_size=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        return self.network(x)


def generate_synthetic_data(num_samples=2000, input_size=20, num_classes=3):
    """Generate synthetic classification data."""
    torch.manual_seed(42)
    data, labels = [], []
    for class_idx in range(num_classes):
        samples_per_class = num_samples // num_classes
        base_values = torch.randn(input_size) * 0.5 + (class_idx - 1) * 2
        class_data = (
            torch.randn(samples_per_class, input_size) * 0.8 + base_values
        )
        class_labels = torch.full(
            (samples_per_class,), class_idx, dtype=torch.long
        )
        data.append(class_data)
        labels.append(class_labels)

    X, y = torch.cat(data, dim=0), torch.cat(labels, dim=0)
    indices = torch.randperm(len(X))
    return X[indices], y[indices]


def _setup_backend_environment(backend_name: str):
    """Configure environment for a specific backend (e.g., offline mode)."""
    if backend_name == "wandb" and not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_MODE"] = "offline"
        print("📝 W&B: Running in offline mode (no API key found)")
    elif backend_name == "clearml":
        os.environ.update(
            {
                "CLEARML_WEB_HOST": "",
                "CLEARML_API_HOST": "",
                "CLEARML_FILES_HOST": "",
                "CLEARML_OFFLINE_MODE": "1",
            }
        )
        print("📝 ClearML: Running in offline mode")


def _get_dataloaders(batch_size=64):
    """Prepare and return train/validation data loaders."""
    X, y = generate_synthetic_data()
    split_idx = int(0.8 * len(X))
    train_X, val_X = X[:split_idx], X[split_idx:]
    train_y, val_y = y[:split_idx], y[split_idx:]

    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, len(train_dataset), len(val_dataset)


def _train_epoch(model, loader, criterion, optimizer, exp, writer, epoch):
    """Run a single training epoch."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for i, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if i % 10 == 0:
            step = epoch * len(loader) + i
            exp.log_metric("train/batch_loss", loss.item(), step)
            writer.add_scalar("train/batch_loss", loss.item(), step)

    return total_loss / len(loader), 100.0 * correct / total


def _validate_epoch(model, loader, criterion):
    """Run a single validation epoch."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def _log_epoch_metrics(
    exp, writer, metrics: dict, epoch: int, model: nn.Module
):
    """Log all metrics for a given epoch."""
    for key, value in metrics.items():
        exp.log_metric(key, value, epoch)
        writer.add_scalar(key, value, epoch)

    if epoch % 3 == 0:
        for name, param in model.named_parameters():
            writer.add_histogram(f"params/{name}", param, epoch)
            if param.grad is not None:
                writer.add_histogram(f"gradients/{name}", param.grad, epoch)


def run_experiment_with_backend(
    backend_name: str, project_name: str = "Multi-Backend Comparison"
):
    """Run the same experiment with a specific backend."""
    print(
        f"\n{'='*60}\n🚀 Running experiment with {backend_name.upper()} backend\n{'='*60}"
    )
    _setup_backend_environment(backend_name)

    train_loader, val_loader, num_train, num_val = _get_dataloaders()
    model = SimpleNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    exp = tracelet.start_logging(
        exp_name=f"{backend_name}_classification_experiment",
        project=project_name,
        backend=backend_name,
    )
    print(f"✅ Started {backend_name} experiment: {exp.name}")

    hyperparams = {
        "learning_rate": 0.001, "batch_size": 64, "epochs": 12
    }
    exp.log_params(hyperparams)

    temp_dir = Path(f"./demo_results/{backend_name}_tensorboard")
    temp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(temp_dir))

    start_time = time.time()
    best_val_accuracy = 0.0
    for epoch in range(12):
        train_loss, train_acc = _train_epoch(
            model, train_loader, criterion, optimizer, exp, writer, epoch
        )
        val_loss, val_acc = _validate_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc

        metrics = {
            "train/epoch_loss": train_loss,
            "train/epoch_accuracy": train_acc,
            "val/epoch_loss": val_loss,
            "val/epoch_accuracy": val_acc,
            "val/best_accuracy": best_val_accuracy,
            "train/learning_rate": optimizer.param_groups[0]["lr"],
        }
        _log_epoch_metrics(exp, writer, metrics, epoch, model)

        print(
            f"Epoch {epoch:2d}/11: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
            f"Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%"
        )

    execution_time = time.time() - start_time
    exp.log_metric("final/best_val_accuracy", best_val_accuracy, epoch)
    exp.log_metric("final/execution_time", execution_time, epoch)

    writer.close()
    tracelet.stop_logging()
    return {
        "backend": backend_name,
        "best_val_accuracy": best_val_accuracy,
        "execution_time": execution_time,
    }


def _is_backend_available(backend_name: str) -> bool:
    """Check if a backend's library is installed."""
    return importlib.util.find_spec(backend_name) is not None


def _get_available_backends(backends_to_check: list[str]) -> list[str]:
    """Get a list of available backends from a given list."""
    available = []
    for backend in backends_to_check:
        if _is_backend_available(backend):
            print(f"✅ {backend.upper()}: Available")
            available.append(backend)
        else:
            print(f"❌ {backend.upper()}: Not installed")
    return available


def _print_comparison_report(results: list[dict], total_time: float):
    """Print a formatted report comparing backend performance."""
    if not results or len(results) <= 1:
        return

    print(f"\n{'='*80}\n📊 MULTI-BACKEND COMPARISON RESULTS\n{'='*80}")
    print(f"{'Backend':<12} {'Best Val Acc':<15} {'Time (s)':<10}")
    print("-" * 80)

    for res in results:
        print(
            f"{res['backend'].upper():<12} {res['best_val_accuracy']:<15.2f}% "
            f"{res['execution_time']:<10.2f}"
        )

    best_accuracy = max(results, key=lambda x: x["best_val_accuracy"])
    fastest_time = min(results, key=lambda x: x["execution_time"])
    accuracies = [r["best_val_accuracy"] for r in results]
    accuracy_std = np.std(accuracies)

    print("\n🏆 PERFORMANCE HIGHLIGHTS:")
    print(
        f"   🎯 Best Accuracy: {best_accuracy['backend'].upper()} "
        f"({best_accuracy['best_val_accuracy']:.2f}%)"
    )
    print(
        f"   ⚡ Fastest Training: {fastest_time['backend'].upper()} "
        f"({fastest_time['execution_time']:.2f}s)"
    )
    print(f"   📊 Total Execution Time: {total_time:.2f}s")
    print(
        f"   📈 Accuracy Consistency: ±{accuracy_std:.2f}% (lower is more consistent)"
    )


def main():
    """Run multi-backend comparison experiment."""
    print("🌟 Tracelet Multi-Backend Comparison Example")
    backends_to_test = ["mlflow", "clearml", "wandb"]
    available_backends = _get_available_backends(backends_to_test)

    if not available_backends:
        print("\n❌ No backends available! Please install at least one.")
        return

    print(f"\n🚀 Running experiments with: {', '.join(available_backends)}")
    results = []
    overall_start_time = time.time()

    for backend in available_backends:
        try:
            result = run_experiment_with_backend(backend)
            results.append(result)
        except Exception as e:
            print(f"❌ Failed to run experiment with {backend}: {e}")

    overall_execution_time = time.time() - overall_start_time
    _print_comparison_report(results, overall_execution_time)

    print("\n🎉 Multi-backend comparison completed successfully!")


if __name__ == "__main__":
    main()


```

## AIM Backend Example

```python
import tracelet
import torch
from torch.utils.tensorboard import SummaryWriter

# Start experiment tracking with your preferred backend
tracelet.start_logging(
    exp_name="my_aim_experiment",
    project="my_aim_project",
    backend="aim"  # Use the AIM backend
)

# Use TensorBoard as usual - metrics are automatically captured
writer = SummaryWriter()
for epoch in range(10):
    loss = 0.9**epoch  # Example loss
    writer.add_scalar('Loss/train', loss, epoch)
    # Metrics are automatically sent to AIM!

# Log some parameters
tracelet.get_active_experiment().log_params({
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10
})

# Log an artifact (e.g., a dummy model file)
with open("dummy_model.txt", "w") as f:
    f.write("This is a dummy model file.")
tracelet.get_active_experiment().log_artifact("dummy_model.txt")

# Stop tracking when done
tracelet.stop_logging()
```
