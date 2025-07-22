# 🔮 Tracelet Examples

Welcome to the Tracelet examples! This collection demonstrates the evolution from manual experiment tracking to truly automagic instrumentation.

## 📁 Example Structure

### 🔧 01. Manual Tracking

Traditional experiment tracking requiring explicit logging calls.

- **`01_basic_manual.py`** - Basic manual logging of metrics and parameters
- **`02_pytorch_manual.py`** - Manual PyTorch model training with explicit tracking
- **`03_sklearn_manual.py`** - Manual scikit-learn experiment tracking

### 🔮 02. Automagic Tracking

Zero-code experiment tracking that captures everything automatically.

- **`01_basic_automagic.py`** - Minimal automagic setup with automatic hyperparameter capture
- **`02_pytorch_automagic.py`** - PyTorch training with full automagic instrumentation
- **`03_sklearn_automagic.py`** - Scikit-learn with automatic parameter and model capture
- **`04_comprehensive_automagic.py`** - Advanced automagic features showcase

### 🔌 03. Backend Integrations

Examples showing integration with different ML backends.

- **`mlflow_integration.py`** - MLflow backend integration
- **`wandb_integration.py`** - Weights & Biases integration
- **`clearml_integration.py`** - ClearML backend integration
- **`multi_backend_comparison.py`** - Using multiple backends simultaneously

### 🚀 04. Advanced Features

Advanced Tracelet capabilities and use cases.

- **`e2e_ml_pipeline.py`** - End-to-end ML pipeline with comprehensive tracking
- **`custom_collectors.py`** - Creating custom data collectors
- **`plugin_development.py`** - Developing custom plugins

## 🚀 Quick Start

👆 **New to Tracelet?** Check out [`QUICKSTART.md`](./QUICKSTART.md) for a 2-minute guide!

### 🔮 Automagic Tracking (Recommended)

```python
from tracelet import Experiment

# Define your hyperparameters normally
learning_rate = 0.001
batch_size = 32
epochs = 10

# THE ONLY TRACELET LINE NEEDED!
experiment = Experiment("my_experiment", automagic=True)

# Train model normally - everything tracked automatically!
for epoch in range(epochs):
    loss = train_epoch()  # Loss automatically captured via hooks!
    # No manual logging needed!

experiment.end()
```

### 📝 Manual Tracking (Full Control)

```python
from tracelet import Experiment

# Create experiment
experiment = Experiment("my_experiment")
experiment.start()

# Manual logging required
experiment.log_params({
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10
})

# Train model...
for epoch in range(epochs):
    loss = train_epoch()
    experiment.log_metric("loss", loss, epoch)  # Manual logging

experiment.end()
```

## 🔄 Migration Path

1. **Start with Manual** (`01_manual_tracking/`) - Understand basic concepts
2. **Try Basic Automagic** (`02_automagic_tracking/01_basic_automagic.py`) - See the magic
3. **Compare Approaches** - Run equivalent manual vs automagic examples
4. **Go Full Automagic** - Use comprehensive automagic features
5. **Integrate Backends** - Connect to your preferred ML platform

## 🎉 Key Benefits of Automagic

| Feature                | Manual Tracking                | 🔮 Automagic Tracking             |
| ---------------------- | ------------------------------ | --------------------------------- |
| Hyperparameter Logging | `experiment.log_params({...})` | ✨ **Automatic**                  |
| Training Metrics       | `experiment.log_metric(...)`   | ✨ **Automatic**                  |
| Model Architecture     | `experiment.log_artifact(...)` | ✨ **Automatic**                  |
| System Resources       | Custom collectors needed       | ✨ **Automatic**                  |
| Framework Integration  | Manual hooks required          | ✨ **Automatic**                  |
| Code Changes           | Extensive modifications        | **Single line: `automagic=True`** |

## 🚀 Getting Started

1. **Install Tracelet**:

   ```bash
   uv add tracelet
   # or
   pip install tracelet
   ```

2. **See the Difference** (Recommended first step):

   ```bash
   # Run side-by-side comparison to see the dramatic difference
   python examples/comparison_manual_vs_automagic.py
   ```

3. **Try Individual Examples**:

   ```bash
   # Manual tracking approach
   python examples/01_manual_tracking/01_basic_manual.py

   # Automagic tracking approach
   python examples/02_automagic_tracking/01_basic_automagic.py

   # See all automagic features
   python examples/02_automagic_tracking/04_comprehensive_automagic.py
   ```

4. **Choose your preferred approach** and integrate into your workflow!

## 🎯 Choose Your Path

- **🔧 Manual Control**: Perfect for custom logging needs and fine-grained control
- **🔮 Automagic Simplicity**: Ideal for rapid prototyping and zero-overhead tracking
- **🔄 Hybrid Approach**: Combine both for maximum flexibility

Start with automagic for the easiest experience, then add manual logging where needed!
