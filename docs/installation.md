# Installation

## Quick Installation

=== "pip"
`bash
    pip install tracelet
    `

=== "uv"
`bash
    uv add tracelet
    `

=== "conda"
`bash
    conda install -c conda-forge tracelet
    `

## Backend Dependencies

Tracelet supports multiple experiment tracking backends. Install the backend you want to use:

=== "MLflow"
`bash
    pip install tracelet[mlflow]
    # or
    pip install mlflow
    `

=== "ClearML"
`bash
    pip install tracelet[clearml]
    # or
    pip install clearml
    `

=== "Weights & Biases"
`bash
    pip install tracelet[wandb]
    # or
    pip install wandb
    `

=== "AIM"
`bash
    pip install tracelet[aim]
    # or
    pip install aim
    `

## System Requirements

- Python 3.8+
- PyTorch (optional, for enhanced integrations)
- PyTorch Lightning (optional, for Lightning integration)

## Development Installation

For development or to get the latest features:

```bash
git clone https://github.com/prassanna-ravishankar/tracelet.git
cd tracelet
uv pip install -e ".[dev,all]"
```

## Verification

Verify your installation:

```python
import tracelet
print(tracelet.__version__)
```

## Next Steps

Choose your backend and follow the configuration guide:

- [MLflow Setup](backends/mlflow.md)
- [ClearML Setup](backends/clearml.md)
- [W&B Setup](backends/wandb.md)
- [AIM Setup](backends/aim.md)
