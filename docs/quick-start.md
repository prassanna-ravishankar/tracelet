# Quick Start

Get up and running with Tracelet in under 5 minutes!

## 1. Installation

=== "pip"
`bash
    pip install tracelet mlflow
    `

=== "uv"
`bash
    uv add tracelet mlflow
    `

## 2. Basic Usage

Here's a complete example using PyTorch with TensorBoard:

```python
import tracelet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

# 1. Start experiment tracking
tracelet.start_logging(
    exp_name="my_first_experiment",
    project="tracelet_demo",
    backend="mlflow"  # or "clearml", "wandb", "aim"
)

# 2. Create a simple model and data
model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Synthetic data
X = torch.randn(100, 10)
y = torch.randn(100, 1)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16)

# 3. Use TensorBoard as normal - metrics are automatically captured!
writer = SummaryWriter()

for epoch in range(50):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Log batch metrics - automatically sent to MLflow!
        writer.add_scalar('Loss/batch', loss.item(), epoch * len(dataloader) + batch_idx)

    # Log epoch metrics
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('Loss/epoch', avg_loss, epoch)

    print(f"Epoch {epoch:2d}: Loss = {avg_loss:.4f}")

# 4. Log additional experiment info
exp = tracelet.get_active_experiment()
exp.log_params({
    "learning_rate": 0.01,
    "batch_size": 16,
    "epochs": 50,
    "model": "linear"
})

# 5. Clean up
writer.close()
tracelet.stop_logging()

print("✅ Experiment completed! Check your MLflow UI to see the results.")
```

## 3. View Results

### MLflow

```bash
mlflow ui
# Visit http://localhost:5000
```

### ClearML

Visit [app.clear.ml](https://app.clear.ml) or your ClearML server

### Weights & Biases

Visit [wandb.ai](https://wandb.ai/home)

### AIM

```bash
aim up
# Visit http://localhost:43800
```

## What Just Happened?

1. **Automatic Capture**: Your existing `SummaryWriter.add_scalar()` calls were automatically intercepted
2. **Zero Code Changes**: No modifications to your existing TensorBoard code
3. **Multi-Backend**: Same code works with MLflow, ClearML, W&B, or AIM
4. **Rich Logging**: Scalars, parameters, and system info automatically tracked

## Next Steps

- [Configuration Guide](configuration.md) - Customize your setup
- [Backend Guides](backends/index.md) - Deep dive into each backend
- [PyTorch Integration](integrations/pytorch.md) - Advanced PyTorch features
- [Examples](examples/basic.md) - More comprehensive examples

!!! tip "Pro Tip"
Try the [multi-backend example](examples/multi-backend.md) to compare different tracking platforms with the same experiment!
