# Automagic Instrumentation

Tracelet's automagic instrumentation provides automatic detection and logging of machine learning hyperparameters with zero configuration. Enable automagic mode and Tracelet intelligently captures your experiment parameters using advanced heuristics.

## Overview

Traditional experiment tracking requires manual logging of every hyperparameter:

```python
# Traditional approach - tedious and error-prone
experiment.log_params({
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 100,
    "dropout": 0.3,
    "hidden_layers": [256, 128, 64],
    "optimizer": "adam",
    # ... 20+ more parameters
})
```

With automagic instrumentation, this becomes:

```python
# Automagic approach - zero configuration
learning_rate = 0.001
batch_size = 64
epochs = 100
dropout = 0.3
hidden_layers = [256, 128, 64]
optimizer = "adam"
# All parameters automatically captured!
```

## Quick Start

### Basic Automagic Usage

```python
from tracelet import Experiment

# Enable automagic mode
experiment = Experiment(
    name="automagic_experiment",
    backend=["mlflow"],
    automagic=True  # Enable automagic instrumentation
)
experiment.start()

# Define hyperparameters normally - they're captured automatically
learning_rate = 3e-4
batch_size = 128
epochs = 50
dropout_rate = 0.1
num_layers = 6

# Your training code here...
# Automagic captures all relevant variables!

experiment.end()
```

### Framework Integration

Automagic automatically hooks into popular ML frameworks:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tracelet import Experiment

# Enable automagic
experiment = Experiment("pytorch_training", automagic=True)
experiment.start()

# Model hyperparameters (automatically captured)
learning_rate = 0.001
weight_decay = 1e-4
momentum = 0.9
batch_size = 32
epochs = 10
input_size = 784
hidden_size = 128
num_classes = 10

# Create a simple model (automatically instrumented)
model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(hidden_size, num_classes)
)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

# Create dummy training data for demonstration
train_data = torch.randn(1000, input_size)
train_targets = torch.randint(0, num_classes, (1000,))
train_dataset = TensorDataset(train_data, train_targets)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop - metrics captured via framework hooks
for epoch in range(epochs):
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()  # Learning rate automatically logged
        # Loss and gradient norms captured automatically (if track_model_gradients=True)

        # Optional: Break after a few batches for demonstration
        if batch_idx >= 2:
            break

experiment.end()
```

## How Automagic Works

### Intelligent Parameter Detection

Automagic uses sophisticated heuristics to identify ML-relevant parameters:

#### 1. Name Pattern Recognition

```python
# These are automatically detected by name patterns
learning_rate = 0.001      # Contains "rate"
batch_size = 64           # Contains "size"
num_layers = 5            # Starts with "num_"
hidden_dim = 256          # Contains "dim"
max_epochs = 100          # Contains "epoch"
```

#### 2. Value Range Analysis

```python
# Detected by typical ML value ranges
learning_rate = 3e-4      # 0.00001 - 0.1 range
dropout = 0.3             # 0 - 1 range for rates
batch_size = 128          # 1 - 1024 range (power of 2)
temperature = 2.0         # Scientific notation
```

#### 3. Data Type Classification

```python
# Boolean hyperparameters
use_layer_norm = True     # Boolean flags
enable_dropout = False    # use_*, enable_*, has_*

# String configurations
optimizer = "adamw"       # Optimizer names
activation = "gelu"       # Activation functions
lr_scheduler = "cosine"   # Scheduler types
```

#### 4. Keyword Detection

```python
# ML-specific keywords automatically recognized
alpha = 0.7              # Regularization parameter
beta1 = 0.9              # Optimizer beta
epsilon = 1e-8           # Numerical stability
patience = 10            # Early stopping
threshold = 1e-4         # Convergence threshold
```

### Framework Hooks

Automagic automatically instruments popular ML frameworks:

#### PyTorch Integration

- **Optimizer hooks**: Automatically log learning rates and gradient norms
- **Loss function hooks**: Capture loss values during forward passes _(Note: Only works with nn.Module-based losses like nn.CrossEntropyLoss(), not functional equivalents like torch.nn.functional.cross_entropy)_
- **Model hooks**: Track model architecture and parameter counts
- **Checkpoint hooks**: Monitor model saving and loading

#### Scikit-learn Integration

- **Estimator hooks**: Capture model hyperparameters during `.fit()` calls
- **Dataset hooks**: Log training set size and feature dimensions
- **Prediction hooks**: Track inference statistics

#### XGBoost Integration

- **Training hooks**: Capture boosting parameters and evaluation metrics
- **Parameter extraction**: Automatic detection of tree-specific settings

### Smart Filtering

Automagic intelligently excludes non-relevant variables:

```python
# Automatically excluded
i = 0                     # Loop variables
model = nn.Sequential()   # Complex objects
device = "cuda"           # System variables
tmp_value = 123          # Temporary variables
_private_var = "test"    # Private variables

# Automatically included
learning_rate = 0.001    # ML hyperparameter
batch_size = 64          # Training parameter
use_attention = True     # Boolean configuration
```

## Configuration

### Automagic Settings

Control automagic behavior through configuration:

```python
from tracelet.automagic import AutomagicConfig

config = AutomagicConfig(
    # Hyperparameter detection
    detect_function_args=True,      # Function argument scanning
    detect_class_attributes=True,   # Class attribute detection
    detect_argparse=True,          # Command-line argument parsing
    detect_config_files=True,      # Configuration file parsing

    # Model tracking
    track_model_architecture=True, # Model structure capture
    track_model_checkpoints=True,  # Checkpoint monitoring
    track_model_gradients=False,   # Gradient tracking (expensive)

    # Dataset tracking
    track_dataset_info=True,       # Dataset statistics
    track_data_samples=False,      # Data sample logging (privacy)

    # Training monitoring
    monitor_training_loop=True,    # Training progress detection
    monitor_loss_curves=True,      # Loss trend analysis
    monitor_learning_rate=True,    # LR schedule tracking

    # Resource monitoring
    monitor_gpu_memory=True,       # GPU usage tracking
    monitor_cpu_usage=True,        # CPU utilization

    # Framework selection
    frameworks={"pytorch", "sklearn", "xgboost"}
)

experiment = Experiment(
    "configured_experiment",
    automagic=True,
    automagic_config=config
)
```

### Environment Variables

Configure automagic via environment variables:

```bash
# Enable/disable automagic
export TRACELET_ENABLE_AUTOMAGIC=true

# Select frameworks to instrument
export TRACELET_AUTOMAGIC_FRAMEWORKS="pytorch,sklearn"

# Control detection scope
export TRACELET_DETECT_FUNCTION_ARGS=true
export TRACELET_DETECT_CLASS_ATTRIBUTES=true
export TRACELET_TRACK_MODEL_ARCHITECTURE=true
```

## Advanced Usage

### Manual Hyperparameter Capture

Force capture of specific variables:

```python
from tracelet.automagic import capture_hyperparams

# Capture current scope variables and log them automatically
# Note: This function automatically logs to the experiment, no need to call log_params
hyperparams = capture_hyperparams(experiment)

# Alternative: Use the experiment's built-in method
hyperparams = experiment.capture_hyperparams()
```

### Custom Detection Rules

Automagic uses built-in detection patterns that can be customized through configuration:

```python
from tracelet.automagic import AutomagicConfig

# Configure detection patterns through AutomagicConfig
config = AutomagicConfig(
    # Enable/disable detection methods
    detect_function_args=True,
    detect_argparse=True,
    detect_config_files=True,

    # Framework-specific detection
    frameworks={"pytorch", "sklearn", "xgboost"}
)

experiment = Experiment(
    "custom_detection",
    automagic=True,
    automagic_config=config
)
```

### Integration with Existing Code

Automagic works seamlessly with existing tracking:

```python
experiment = Experiment("mixed_tracking", automagic=True)
experiment.start()

# Automagic captures these automatically
learning_rate = 0.001
batch_size = 64

# Manual logging still works
experiment.log_params({
    "model_name": "custom_transformer",
    "dataset_version": "v2.1"
})

# Both automatic and manual tracking combined!
```

## Performance Considerations

### Overhead

Automagic is designed for minimal performance impact:

- **Frame inspection**: ~0.1ms per variable check
- **Hook installation**: One-time cost at experiment start
- **Metric capture**: Asynchronous, non-blocking
- **Memory usage**: <10MB for typical experiments

### Best Practices

1. **Scope management**: Define hyperparameters at function/class level
2. **Naming conventions**: Use descriptive, ML-specific variable names
3. **Framework integration**: Let automagic handle metric capture
4. **Selective enabling**: Disable expensive features if not needed

```python
# Good practice
def train_model():
    learning_rate = 0.001  # Clear scope
    batch_size = 64        # Descriptive name

    experiment = Experiment("training", automagic=True)
    # Automagic captures hyperparameters from this scope

# Avoid
learning_rate = 0.001      # Global scope (harder to detect)
lr = 0.001                 # Ambiguous name
```

## Troubleshooting

### Common Issues

**Hyperparameters not detected**:

```python
# Check variable scope and naming
def train():
    learning_rate = 0.001  # Function scope
    lr = 0.001            # Ambiguous name

# Ensure automagic is enabled
experiment = Experiment("test", automagic=True)  # Correct
```

**Framework hooks not working**:

```python
# Import frameworks after starting experiment
experiment.start()
import torch  # Hooks installed after this import

# Or restart Python session if hooks conflict
```

**Performance concerns**:

```python
# Disable expensive features
config = AutomagicConfig(
    track_model_gradients=False,  # Expensive
    track_data_samples=False,     # Privacy risk
    monitor_cpu_usage=False       # High frequency
)
```

### Debug Mode

Enable detailed logging to understand automagic behavior:

```python
import logging
logging.getLogger("tracelet.automagic").setLevel(logging.DEBUG)

experiment = Experiment("debug", automagic=True)
# Detailed logs show detection process
```

## Examples

See comprehensive examples in the [examples documentation](examples.md) and the [examples directory](https://github.com/prassanna-ravishankar/tracelet/tree/main/examples) for automagic usage patterns and best practices.

## API Reference

For detailed API documentation, refer to the automagic module source code and docstrings.
