#!/usr/bin/env python3
"""
Simple demonstration of Tracelet's automagic instrumentation.

This shows the minimal code needed to get automatic ML experiment tracking
similar to ClearML's Task.init() approach.
"""

from tracelet import Experiment


def train_model():
    """A simple training function that demonstrates automagic capture."""

    # Hyperparameters - will be automatically captured
    learning_rate = 0.001  # noqa: F841
    batch_size = 32  # noqa: F841
    epochs = 10  # noqa: F841 
    dropout = 0.2  # noqa: F841
    hidden_size = 128  # noqa: F841

    # Create experiment with automagic enabled
    experiment = Experiment(
        name="simple_automagic_demo",
        backend=["mlflow"],  # Use your preferred backend
        automagic=True,  # Enable automagic instrumentation!
    )

    # Start the experiment
    experiment.start()

    # Capture hyperparameters from local variables automatically
    hyperparams = experiment.capture_hyperparams()
    print(f"✨ Automagically captured: {list(hyperparams.keys())}")

    # Simulate training loop
    print("🚀 Training with automagic instrumentation...")

    for epoch in range(epochs):
        # Simulate training metrics
        loss = 1.0 * (0.9**epoch) + 0.1  # Decreasing loss
        accuracy = 0.5 + 0.4 * (1 - 0.9**epoch)  # Increasing accuracy

        # Log metrics (manual logging still works alongside automagic)
        experiment.log_metric("loss", loss)
        experiment.log_metric("accuracy", accuracy)
        experiment.log_metric("epoch", epoch)

        print(f"Epoch {epoch + 1}/{epochs}: loss={loss:.4f}, acc={accuracy:.4f}")

    print("✅ Training completed!")
    print("📊 Check your MLflow UI (or other backend) to see:")
    print("   • Automatically captured hyperparameters")
    print("   • Training metrics and progress")
    print("   • System information")

    # Clean up
    experiment.end()


def train_with_sklearn():
    """Demonstrate automagic with scikit-learn."""

    try:
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("⚠️  Scikit-learn not available, skipping sklearn demo")
        return

    print("\n🔬 Scikit-learn Automagic Demo")
    print("=" * 40)

    # Hyperparameters
    n_estimators = 100
    max_depth = 10
    random_state = 42
    test_size = 0.3

    # Create experiment with automagic
    experiment = Experiment(name="sklearn_automagic_demo", backend=["mlflow"], automagic=True)
    experiment.start()

    # Auto-capture hyperparameters
    hyperparams = experiment.capture_hyperparams()
    print(f"✨ Auto-captured: {list(hyperparams.keys())}")

    # Create synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Auto-capture dataset info
    dataset_info = experiment.capture_dataset(X_train)
    print(f"📊 Dataset info: {dataset_info.get('shape', 'N/A')}")

    # Create and train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    print("🌲 Training Random Forest...")
    model.fit(X_train, y_train)

    # Auto-capture model info
    model_info = experiment.capture_model(model)
    print(f"🤖 Model type: {model_info.get('type', 'Unknown')}")

    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    experiment.log_metric("train_accuracy", train_acc)
    experiment.log_metric("test_accuracy", test_acc)

    print(f"📈 Train accuracy: {train_acc:.4f}")
    print(f"📈 Test accuracy: {test_acc:.4f}")

    experiment.end()


def main():
    """Run the automagic demonstrations."""
    print("Tracelet Automagic Instrumentation Demo")
    print("=" * 50)
    print()
    print("This demo shows how Tracelet can automatically capture")
    print("experiment information with minimal code changes.")
    print()
    print("🔮 Just add automagic=True to your Experiment!")
    print()

    # Run basic demo
    print("🎯 Basic Automagic Demo")
    print("=" * 30)
    train_model()

    # Run sklearn demo if available
    train_with_sklearn()

    print("\n" + "=" * 50)
    print("🎉 Demos completed!")
    print()
    print("Key benefits of Tracelet's automagic instrumentation:")
    print("✅ Zero-code hyperparameter capture")
    print("✅ Automatic model architecture logging")
    print("✅ Framework-agnostic design")
    print("✅ Works alongside manual logging")
    print("✅ Easy integration: just add automagic=True")
    print()
    print("💡 For more advanced features, see examples/automagic_example.py")


if __name__ == "__main__":
    main()
