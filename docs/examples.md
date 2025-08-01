# Examples

Learn Tracelet through practical examples in the `/examples` directory. Each category demonstrates different aspects of experiment tracking.

## Quick Examples

### Basic Tracking

```python
from tracelet import Experiment

# Create experiment with automatic detection
exp = Experiment(name="my_model", backend=["wandb"], automagic=True)
exp.start()

# Your existing training code - metrics logged automatically
for epoch in range(10):
    loss = model.train_one_epoch()
    print(f"Loss: {loss}")

exp.stop()
```

### Artifact Tracking

```python
from tracelet.core.artifacts import ArtifactType

# Log a trained model
model_artifact = exp.create_artifact(
    name="classifier",
    artifact_type=ArtifactType.MODEL
).add_file("model.pth", "model/classifier.pth")
exp.log_artifact(model_artifact)
```

## Detailed Examples

### [Manual Tracking](examples/basic.md)

Step-by-step introduction to manual metric logging and parameter tracking.

### [Multi-Backend](examples/multi-backend.md)

Compare MLflow, ClearML, and W&B backends with the same experiment.

### [Notebooks](examples/notebooks.md)

Jupyter notebook examples with visualizations and interactive exploration.

## Repository Examples

The `/examples` directory contains runnable examples organized by complexity:

- **`01_manual_tracking/`** - Basic manual tracking examples
- **`02_automagic_tracking/`** - Zero-config automatic tracking
- **`03_backend_integrations/`** - Backend-specific features
- **`04_advanced_features/`** - Production-ready patterns
- **`05_lightning_automagic/`** - PyTorch Lightning integration
- **`06_artifacts/`** - Artifact management examples

### Quick Start Path

1. **New to tracking**: `examples/01_manual_tracking/01_basic_manual.py`
2. **Want zero config**: `examples/02_automagic_tracking/01_basic_automagic.py`
3. **Use PyTorch Lightning**: `examples/05_lightning_automagic/simple_lightning_example.py`
4. **Compare backends**: `examples/03_backend_integrations/compare_all_backends.py`

## Running Examples

All examples use synthetic data and can be run immediately:

```bash
cd examples
python 01_manual_tracking/01_basic_manual.py
```

For multi-backend examples, install the required backends:

```bash
pip install tracelet[mlflow,wandb,clearml]
```
