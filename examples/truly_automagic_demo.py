#!/usr/bin/env python3
"""
TRULY Automagic Tracelet Demo - Zero Manual Instrumentation!

This demonstrates the "ClearML-style" automagic approach where simply
creating an Experiment with automagic=True captures EVERYTHING automatically:

- Hyperparameters from local variables
- Training metrics from framework hooks
- Model architecture when models are created
- Dataset info when data is loaded
- System resources during training

NO MANUAL CALLS NEEDED - Just create the experiment and code normally!
"""

# Only needed import
from tracelet import Experiment

# Optional imports for ML
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def train_neural_network():
    """Train a neural network with ZERO manual instrumentation!"""
    if not HAS_TORCH:
        print("⚠️  PyTorch not available, skipping neural network demo")
        return

    print("🧠 Neural Network Training - Truly Automagic!")
    print("=" * 50)

    # 1. Define hyperparameters (will be automatically captured!)
    learning_rate = 0.001
    batch_size = 64
    epochs = 5
    hidden_size = 256
    dropout_rate = 0.3
    weight_decay = 1e-4
    optimizer_type = "adam"  # noqa: F841

    # 2. Create experiment - THE ONLY TRACELET LINE NEEDED!
    experiment = Experiment(
        name="truly_automagic_neural_network",
        backend=["mlflow"],
        automagic=True,  # ← MAGIC HAPPENS HERE!
    )
    experiment.start()  # Start tracking

    print("🔮 Hyperparameters automatically captured from local variables!")

    # 3. Create data (will be automatically detected and logged!)
    X = torch.randn(1000, 20)
    y = torch.randint(0, 3, (1000,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("📊 Dataset info automatically captured!")

    # 4. Create model (architecture automatically captured!)
    model = nn.Sequential(
        nn.Linear(20, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_size // 2, 3),
    )

    print("🏗️  Model architecture automatically captured!")

    # 5. Create optimizer and loss (will be automatically hooked!)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    print("⚡ Optimizer and loss function automatically hooked!")
    print("\n🚀 Starting training - all metrics will be automatically logged!")

    # 6. Train model (everything logged automatically!)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for _batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()  # ← Learning rate automatically logged here!

            output = model(data)
            loss = criterion(output, target)  # ← Loss automatically logged here!
            loss.backward()

            optimizer.step()  # ← Gradient norms automatically logged here!

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}")

    print("\n✅ Training completed!")
    print("🔍 Check your MLflow UI - everything was logged automatically:")
    print("   • All local variables identified as hyperparameters")
    print("   • Learning rates logged at every optimizer step")
    print("   • Losses logged at every forward pass")
    print("   • Gradient norms logged during backprop")
    print("   • Model architecture and parameter counts")
    print("   • Dataset shape and type information")
    print("   • System resource usage during training")

    experiment.end()


def train_sklearn_model():
    """Train scikit-learn model with zero instrumentation!"""
    if not HAS_SKLEARN:
        print("⚠️  Scikit-learn not available, skipping sklearn demo")
        return

    print("\n🌲 Random Forest Training - Truly Automagic!")
    print("=" * 50)

    # 1. Hyperparameters (automatically captured!)
    n_estimators = 200
    max_depth = 15
    min_samples_split = 5
    min_samples_leaf = 2
    random_state = 42
    test_size = 0.25
    n_samples = 2000
    n_features = 25
    n_classes = 4

    # 2. Create experiment - ONLY TRACELET LINE!
    experiment = Experiment(
        name="truly_automagic_random_forest",
        backend=["mlflow"],
        automagic=True,  # ← ALL THE MAGIC!
    )
    experiment.start()

    print("🔮 All hyperparameters automatically captured!")

    # 3. Generate data (info automatically captured!)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=2,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print("📊 Dataset automatically analyzed and logged!")

    # 4. Train model (hyperparameters automatically captured via hooks!)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )

    print("🌲 Training Random Forest (model hyperparameters auto-captured)...")
    model.fit(X_train, y_train)  # ← All model params automatically logged via hooks!

    # 5. Evaluate (predictions automatically tracked!)
    train_score = model.score(X_train, y_train)  # ← Inference automatically logged!
    test_score = model.score(X_test, y_test)  # ← Inference automatically logged!

    print(f"📈 Train Accuracy: {train_score:.4f}")
    print(f"📈 Test Accuracy: {test_score:.4f}")

    print("\n✅ Training completed!")
    print("🔍 Everything automatically logged:")
    print("   • All local variables as hyperparameters")
    print("   • Model hyperparameters from RandomForestClassifier")
    print("   • Dataset shape, features, and classes")
    print("   • Training completion and inference sample counts")
    print("   • Model type and available methods")

    experiment.end()


def simple_computation():
    """Even simple computational tasks get automatic tracking!"""
    print("\n🧮 Simple Computation - Still Automagic!")
    print("=" * 50)

    # Hyperparameters for a simple computation
    iterations = 1000
    step_size = 0.01
    target_value = 3.14159
    tolerance = 1e-6
    algorithm = "gradient_descent"  # noqa: F841

    # Create automagic experiment
    experiment = Experiment(name="simple_automagic_computation", backend=["mlflow"], automagic=True)
    experiment.start()

    print("🔮 Computation parameters automatically captured!")

    # Simple iterative computation
    current_value = 1.0
    for i in range(iterations):
        error = target_value - current_value
        if abs(error) < tolerance:
            print(f"✅ Converged after {i} iterations!")
            break
        current_value += step_size * error

        # Even manual logging works alongside automagic
        if i % 100 == 0:
            experiment.log_metric("current_value", current_value)
            experiment.log_metric("error", abs(error))

    print(f"🎯 Final value: {current_value:.6f}")
    print(f"🎯 Target value: {target_value:.6f}")

    experiment.end()


def main():
    """Run truly automagic demonstrations."""
    print("🔮 TRACELET TRULY AUTOMAGIC DEMO 🔮")
    print("=" * 60)
    print()
    print("This demo shows how Tracelet provides ClearML-style automagic")
    print("instrumentation where creating an Experiment automatically captures:")
    print()
    print("✨ Hyperparameters from local variables")
    print("✨ Training metrics via framework hooks")
    print("✨ Model architectures when created")
    print("✨ Dataset info when loaded")
    print("✨ System resources during execution")
    print()
    print("🎯 ZERO manual instrumentation calls needed!")
    print("🎯 Just create Experiment(automagic=True) and code normally!")
    print()

    # Run demos
    simple_computation()
    train_sklearn_model()
    train_neural_network()

    print("\n" + "=" * 60)
    print("🎉 ALL DEMOS COMPLETED!")
    print()
    print("🔍 Check your MLflow UI to see all the automatically captured:")
    print("   📊 Hyperparameters from local variables")
    print("   📈 Training metrics from framework hooks")
    print("   🏗️  Model architectures and metadata")
    print("   💾 Dataset information and statistics")
    print("   ⚡ System resource usage")
    print()
    print("💡 This is true 'automagic' - no manual instrumentation needed!")
    print("💡 Just add automagic=True and everything is tracked automatically!")


if __name__ == "__main__":
    main()
