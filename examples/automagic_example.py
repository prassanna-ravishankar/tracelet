#!/usr/bin/env python3
"""
Example demonstrating automagic instrumentation in Tracelet.

This example shows how tracelet can automatically capture:
- Hyperparameters from function arguments and variables
- Model architecture and training metrics
- Dataset information
- System resource usage
- Training progress

Run this example to see automagic instrumentation in action!
"""

import argparse
import time
from dataclasses import dataclass

# Import tracelet
from tracelet import Experiment
from tracelet.automagic import AutomagicConfig, automagic, capture_hyperparams

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available, using simplified example")

try:
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Scikit-learn not available, skipping sklearn example")


@dataclass
class TrainingConfig:
    """Training configuration - will be automatically captured."""

    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10
    hidden_size: int = 128
    dropout: float = 0.2
    weight_decay: float = 1e-4


def create_dummy_model(input_size: int, hidden_size: int, num_classes: int, dropout: float = 0.2):
    """Create a simple neural network - architecture will be auto-captured."""
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size // 2, num_classes),
    )


def pytorch_training_example():
    """Example of automagic instrumentation with PyTorch."""
    if not HAS_TORCH:
        print("Skipping PyTorch example - PyTorch not available")
        return

    print("=== PyTorch Automagic Example ===")

    # Create experiment with automagic configuration
    automagic_config = AutomagicConfig(
        detect_function_args=True,
        detect_class_attributes=True,
        track_model_architecture=True,
        track_model_checkpoints=True,
        monitor_training_loop=True,
        monitor_gpu_memory=True,
    )

    # Initialize experiment
    experiment = Experiment(
        name="automagic_pytorch_demo",
        backend=["mlflow"],  # You can use any backend
        tags=["automagic", "pytorch", "demo"],
    )

    # Enable automagic instrumentation
    instrumentor = automagic(experiment, automagic_config)

    # Training configuration - will be automatically captured
    config = TrainingConfig(
        learning_rate=0.001,
        batch_size=64,
        epochs=5,
        hidden_size=256,
        dropout=0.3,
    )

    # Capture hyperparameters from local variables
    captured_hyperparams = capture_hyperparams(experiment)
    print(f"Auto-captured hyperparameters: {list(captured_hyperparams.keys())}")

    # Create dummy data - dataset info will be auto-captured
    X = torch.randn(1000, 20)  # 1000 samples, 20 features
    y = torch.randint(0, 3, (1000,))  # 3 classes
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Auto-capture dataset information
    instrumentor.capture_dataset_info(dataset, experiment)

    # Create model - architecture will be auto-captured
    model = create_dummy_model(20, config.hidden_size, 3, config.dropout)

    # Auto-capture model information
    instrumentor.capture_model_info(model, experiment)

    # Create optimizer - will be automatically hooked for LR tracking
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()  # Loss will be automatically logged

    print("Starting training with automagic instrumentation...")

    # Training loop - metrics will be automatically captured
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)  # Loss automatically logged by hooks
            loss.backward()
            optimizer.step()  # LR automatically logged by hooks

            epoch_loss += loss.item()

            # Manual logging still works alongside automagic
            if batch_idx % 10 == 0:
                experiment.log_metric("batch_loss", loss.item())

        avg_loss = epoch_loss / len(dataloader)
        experiment.log_metric("epoch_loss", avg_loss)
        experiment.log_metric("epoch", epoch)

        print(f"Epoch {epoch + 1}/{config.epochs}, Loss: {avg_loss:.4f}")

        # Simulate saving checkpoint - will be auto-captured
        if epoch % 2 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

    print("Training completed! Check your ML tracking backend for automagically captured metrics.")

    # Clean up
    instrumentor.detach_experiment(experiment.id)
    experiment.end()


def sklearn_training_example():
    """Example of automagic instrumentation with scikit-learn."""
    if not HAS_SKLEARN:
        print("Skipping sklearn example - scikit-learn not available")
        return

    print("\n=== Scikit-learn Automagic Example ===")

    # Create experiment
    experiment = Experiment(name="automagic_sklearn_demo", backend=["mlflow"], tags=["automagic", "sklearn", "demo"])

    # Enable automagic instrumentation
    instrumentor = automagic(experiment)

    # Hyperparameters - will be automatically captured
    n_estimators = 100
    max_depth = 10
    random_state = 42
    test_size = 0.2

    # Capture hyperparameters
    captured_hyperparams = capture_hyperparams(experiment)
    print(f"Auto-captured hyperparameters: {list(captured_hyperparams.keys())}")

    # Create dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Auto-capture dataset info
    instrumentor.capture_dataset_info(X_train, experiment)

    # Create and train model - hyperparameters and training will be auto-captured
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    print("Training sklearn model with automagic instrumentation...")

    # Fit model - hyperparameters will be automatically logged
    model.fit(X_train, y_train)

    # Auto-capture model info
    instrumentor.capture_model_info(model, experiment)

    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    experiment.log_metric("train_accuracy", train_score)
    experiment.log_metric("test_accuracy", test_score)

    print(f"Train accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")

    # Clean up
    instrumentor.detach_experiment(experiment.id)
    experiment.end()


def simple_function_example():
    """Example showing hyperparameter capture from function arguments."""
    print("\n=== Function Argument Automagic Example ===")

    def train_model(learning_rate=0.01, batch_size=32, epochs=10, model_type="cnn"):
        """A training function - arguments will be auto-captured."""

        # Create experiment inside function
        experiment = Experiment(
            name="automagic_function_demo", backend=["mlflow"], tags=["automagic", "function", "demo"]
        )

        # Enable automagic - will automatically capture function arguments
        instrumentor = automagic(experiment)

        # Capture hyperparameters from this function's arguments
        captured_hyperparams = capture_hyperparams(experiment)
        print(f"Auto-captured function arguments: {list(captured_hyperparams.keys())}")

        # Additional variables will also be captured if they look like hyperparameters
        dropout_rate = 0.2  # noqa: F841
        weight_decay = 1e-4  # noqa: F841
        optimizer_name = "adam"  # noqa: F841

        # These will be automatically detected as hyperparameters
        additional_hyperparams = capture_hyperparams(experiment)
        print(f"Auto-captured additional hyperparameters: {list(additional_hyperparams.keys())}")

        # Simulate training
        for epoch in range(epochs):
            # Simulate loss decreasing
            fake_loss = 1.0 * (0.9**epoch) + 0.1
            experiment.log_metric("loss", fake_loss)
            time.sleep(0.1)  # Brief pause to simulate training

        experiment.end()
        instrumentor.detach_experiment(experiment.id)

    # Call function with different hyperparameters
    train_model(learning_rate=0.001, batch_size=64, epochs=5, model_type="resnet")


def main():
    """Run all automagic examples."""
    print("Tracelet Automagic Instrumentation Demo")
    print("=" * 50)
    print()
    print("This demo shows how Tracelet can automatically capture:")
    print("- Hyperparameters from function arguments and variables")
    print("- Model architectures and training metrics")
    print("- Dataset information")
    print("- System resources (if monitoring libraries are available)")
    print("- Training progress and patterns")
    print()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Tracelet Automagic Demo")
    parser.add_argument("--pytorch", action="store_true", help="Run PyTorch example")
    parser.add_argument("--sklearn", action="store_true", help="Run sklearn example")
    parser.add_argument("--function", action="store_true", help="Run function example")
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    # If no specific examples selected, run all
    if not any([args.pytorch, args.sklearn, args.function]):
        args.all = True

    # Run selected examples
    if args.all or args.function:
        simple_function_example()

    if args.all or args.pytorch:
        pytorch_training_example()

    if args.all or args.sklearn:
        sklearn_training_example()

    print("\n" + "=" * 50)
    print("Demo completed! Check your MLflow UI (or other backend) to see the")
    print("automatically captured experiments with hyperparameters, metrics,")
    print("model information, and system resources.")
    print()
    print("Key benefits of automagic instrumentation:")
    print("- No need to manually log hyperparameters")
    print("- Automatic model architecture capture")
    print("- Training progress monitoring without explicit logging")
    print("- System resource tracking")
    print("- Works alongside manual logging")


if __name__ == "__main__":
    main()
