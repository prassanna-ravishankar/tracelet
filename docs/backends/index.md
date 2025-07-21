# Backend Overview

Tracelet supports multiple experiment tracking backends, allowing you to choose the platform that best fits your needs.

## Supported Backends

| Backend                      | Type             | Hosting    | Best For                 |
| ---------------------------- | ---------------- | ---------- | ------------------------ |
| [MLflow](mlflow.md)          | Open Source      | Self/Cloud | Traditional ML workflows |
| [ClearML](clearml.md)        | Enterprise/SaaS  | SaaS/Self  | Enterprise MLOps         |
| [Weights & Biases](wandb.md) | SaaS/Open Source | SaaS/Self  | Deep learning research   |
| [AIM](aim.md)                | Open Source      | Self       | Lightweight tracking     |

## Choosing a Backend

### MLflow

- ✅ **Best for**: Traditional ML, production deployments
- ✅ **Strengths**: Model registry, serving, mature ecosystem
- ❌ **Limitations**: Basic visualization, manual setup

### ClearML

- ✅ **Best for**: Enterprise teams, automated pipelines
- ✅ **Strengths**: Rich UI, automatic logging, pipeline orchestration
- ❌ **Limitations**: Complex setup, resource intensive

### Weights & Biases

- ✅ **Best for**: Deep learning research, collaboration
- ✅ **Strengths**: Best-in-class visualization, sharing, reports
- ❌ **Limitations**: SaaS dependency, pricing for teams

### AIM

- ✅ **Best for**: Simple tracking, local development
- ✅ **Strengths**: Lightweight, fast queries, local-first
- ❌ **Limitations**: Fewer features, smaller ecosystem

## Backend Comparison

### Feature Matrix

| Feature                | MLflow | ClearML | W&B | AIM |
| ---------------------- | ------ | ------- | --- | --- |
| Metrics Logging        | ✅     | ✅      | ✅  | ✅  |
| Hyperparameters        | ✅     | ✅      | ✅  | ✅  |
| Artifacts              | ✅     | ✅      | ✅  | ⚠️  |
| Model Registry         | ✅     | ✅      | ✅  | ❌  |
| Visualizations         | ⚠️     | ✅      | ✅  | ✅  |
| Collaboration          | ⚠️     | ✅      | ✅  | ⚠️  |
| Auto-logging           | ⚠️     | ✅      | ✅  | ❌  |
| Pipeline Orchestration | ❌     | ✅      | ⚠️  | ❌  |
| Self-hosting           | ✅     | ✅      | ✅  | ✅  |
| Free Tier              | ✅     | ✅      | ✅  | ✅  |

Legend: ✅ Full support, ⚠️ Limited support, ❌ Not supported

## Multi-Backend Support

Use multiple backends simultaneously:

```python
import tracelet

# Log to both MLflow and W&B
tracelet.start_logging(
    backend=["mlflow", "wandb"],
    exp_name="multi_backend_experiment",
    project="comparison_study"
)

# All metrics go to both platforms
writer = SummaryWriter()
writer.add_scalar("loss", 0.5, 1)  # → MLflow + W&B
```

Benefits:

- **Backup**: Redundant logging prevents data loss
- **Comparison**: Evaluate different platform features
- **Migration**: Gradual transition between platforms
- **Team preferences**: Support different tool preferences

## Performance Comparison

Typical overhead per logged metric:

| Backend | Latency | Memory | Notes                  |
| ------- | ------- | ------ | ---------------------- |
| MLflow  | ~5ms    | Low    | Local file-based       |
| ClearML | ~15ms   | Medium | Rich automatic logging |
| W&B     | ~20ms   | Medium | Network-dependent      |
| AIM     | ~2ms    | Low    | Optimized for speed    |

## Getting Started

1. **Choose your backend** based on your needs
2. **Install dependencies** for your chosen backend
3. **Configure authentication** if using hosted services
4. **Start logging** with a simple example

Quick setup links:

- [MLflow Installation →](mlflow.md#installation)
- [ClearML Setup →](clearml.md#setup-and-authentication)
- [W&B Setup →](wandb.md#setup-and-authentication)
- [AIM Installation →](aim.md#installation)

## Migration Between Backends

Switching backends is easy with Tracelet:

```python
# Change from MLflow to W&B
# tracelet.start_logging(backend="mlflow")  # Old
tracelet.start_logging(backend="wandb")     # New

# Your existing TensorBoard code remains unchanged!
writer.add_scalar("loss", loss_value, step)
```

See our [Migration Guide](../guides/migration.md) for detailed instructions.

## Next Steps

- [Choose and configure your backend](mlflow.md)
- [Try the multi-backend example](../examples/multi-backend.md)
- [Learn about advanced features](../guides/best-practices.md)
