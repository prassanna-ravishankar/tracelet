# Tracelet End-to-End Testing Framework

This directory contains a comprehensive end-to-end testing framework for Tracelet that validates the complete system works correctly across all supported backends and workflows.

## Overview

The E2E testing framework provides:

1. **Backend Environment Management**: Automated setup/teardown of MLflow, ClearML, and W&B backends
2. **Realistic Training Workflows**: Complete PyTorch and PyTorch Lightning training scenarios
3. **Cross-Backend Validation**: Ensures consistent behavior across all backends
4. **Performance Benchmarking**: Measures execution time and resource usage
5. **Advanced Scenarios**: Computer vision, NLP, time series, and multi-GPU workflows
6. **Comprehensive Reporting**: Detailed test reports with performance metrics

## Architecture

### Core Components

- **`framework.py`**: Core E2E testing framework with backend environments and workflow abstractions
- **`test_basic_workflows.py`**: Basic PyTorch and Lightning workflow tests across all backends
- **`test_advanced_workflows.py`**: Advanced scenarios including CV, NLP, and time series
- **`test_runner.py`**: Comprehensive test runner with detailed reporting
- **`conftest.py`**: Pytest configuration and fixtures

### Backend Environments

Each backend has a dedicated environment class that handles:

- **MLflowEnvironment**: Local file-based MLflow tracking
- **ClearMLEnvironment**: Offline mode ClearML setup
- **WandbEnvironment**: Offline mode W&B configuration

### Training Workflows

Pre-built realistic training scenarios:

- **SimplePyTorchWorkflow**: Basic regression with synthetic data
- **LightningWorkflow**: PyTorch Lightning training loop
- **ComputerVisionWorkflow**: CNN image classification
- **TimeSeriesWorkflow**: LSTM time series forecasting

## Quick Start

### Prerequisites

Install the required dependencies:

```bash
# Basic requirements (always needed)
pip install pytest

# Backend dependencies (install at least one)
pip install mlflow          # For MLflow backend tests
pip install wandb           # For W&B backend tests
pip install clearml         # For ClearML backend tests

# Framework dependencies (for workflows)
pip install torch           # For PyTorch workflows
pip install pytorch-lightning  # For Lightning workflows
pip install torchvision     # For computer vision workflows
pip install numpy           # For advanced workflows
pip install matplotlib     # For visualization tests
```

### Running Tests

#### Basic Test Suite

```bash
# Run all basic E2E tests
pytest tests/e2e/test_basic_workflows.py -v

# Run tests for specific backend
pytest tests/e2e/test_basic_workflows.py -v -k "mlflow"

# Run tests for specific workflow
pytest tests/e2e/test_basic_workflows.py -v -k "pytorch"
```

#### Advanced Test Suite

```bash
# Run advanced E2E tests (requires additional dependencies)
pytest tests/e2e/test_advanced_workflows.py -v

# Run computer vision tests
pytest tests/e2e/test_advanced_workflows.py -v -k "computer_vision"

# Run time series tests
pytest tests/e2e/test_advanced_workflows.py -v -k "time_series"
```

#### Comprehensive Testing

```bash
# Run comprehensive test suite with custom backends
pytest tests/e2e/ --e2e-backends=mlflow,wandb

# Run with custom workflows
pytest tests/e2e/ --e2e-workflows=simple_pytorch,lightning

# Skip slow tests
pytest tests/e2e/ --e2e-skip-slow
```

#### Using the Test Runner

```bash
# Run comprehensive test suite with detailed reporting
python tests/e2e/test_runner.py

# Test specific backends and workflows
python tests/e2e/test_runner.py --backends mlflow wandb --workflows simple_pytorch

# Custom output directory
python tests/e2e/test_runner.py --output-dir ./my_results

# Don't save result files
python tests/e2e/test_runner.py --no-save
```

### Running the Demo

```bash
# Run the interactive demo
python examples/e2e_demo.py
```

## Framework Usage

### Creating Custom Workflows

```python
from tests.e2e.framework import TrainingWorkflow
import tracelet

class MyCustomWorkflow(TrainingWorkflow):
    def __init__(self, config=None):
        super().__init__("my_workflow", config)

    def run(self, backend_config):
        # Start experiment tracking
        exp = tracelet.start_logging(
            exp_name="my_experiment",
            project=backend_config.get("project", "test"),
            backend=backend_config["backend"]
        )

        # Your training code here
        for epoch in range(10):
            # Simulate training
            loss = 1.0 / (epoch + 1)
            exp.log_metric("loss", loss, epoch)

        tracelet.stop_logging()

        return {
            "epochs_completed": 10,
            "final_loss": loss
        }

    def get_expected_metrics(self):
        return ["loss"]

    def validate_results(self, results):
        return results["epochs_completed"] == 10

# Register your workflow
from tests.e2e.framework import e2e_framework
e2e_framework.workflows["my_workflow"] = MyCustomWorkflow
```

### Running Custom Tests

```python
from tests.e2e.framework import e2e_framework

# Run your custom workflow with all backends
results = e2e_framework.run_comprehensive_test(
    workflows=["my_workflow"]
)

# Or test with specific backend
with e2e_framework.backend_environment("mlflow") as backend_config:
    result = e2e_framework.run_workflow("my_workflow", backend_config)
    print(f"Success: {result['success']}")
```

## Available Test Markers

Use pytest markers to run specific test categories:

```bash
# Basic integration tests
pytest tests/e2e/ -m e2e_basic

# Advanced workflow tests
pytest tests/e2e/ -m e2e_advanced

# Performance benchmarks
pytest tests/e2e/ -m e2e_performance

# Comprehensive cross-backend tests
pytest tests/e2e/ -m e2e_comprehensive
```

## Test Results and Reports

### Automated Reports

The test runner automatically generates:

- **JSON Report**: Complete test results with detailed metrics
- **Markdown Summary**: Human-readable test summary
- **Performance Analysis**: Execution time breakdowns
- **Error Analysis**: Detailed failure information

Example report structure:

```
e2e_test_results/
├── e2e_report_2024-01-15T10-30-00.json
└── e2e_summary_2024-01-15T10-30-00.md
```

### Understanding Results

Key metrics in test reports:

- **Pass Rate**: Percentage of successful tests
- **Execution Time**: Total and per-test timing
- **Backend Performance**: Comparison across backends
- **Workflow Success**: Per-workflow reliability
- **Error Patterns**: Common failure modes

## Troubleshooting

### Common Issues

**No backends available**:

- Install at least one backend: `pip install mlflow`
- Check import errors in the logs

**No workflows available**:

- Install PyTorch: `pip install torch`
- For advanced workflows: `pip install torchvision numpy`

**ClearML connection errors**:

- ClearML tests run in offline mode by default
- Server connection errors are expected in CI environments

**W&B authentication errors**:

- W&B tests run in offline mode (no login required)
- Authentication errors indicate configuration issues

### Debugging Tests

```bash
# Run with verbose output
pytest tests/e2e/ -v -s

# Run single test with full traceback
pytest tests/e2e/test_basic_workflows.py::TestBasicWorkflows::test_pytorch_with_mlflow -vvv

# Enable debug logging
TRACELET_LOG_LEVEL=DEBUG pytest tests/e2e/ -v

# Keep temporary files for inspection
pytest tests/e2e/ --basetemp=./debug_temp
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: E2E Tests

on: [push, pull_request]

jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        backend: [mlflow, wandb]

    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest torch ${{ matrix.backend }}

      - name: Run E2E tests
        run: |
          pytest tests/e2e/ --e2e-backends=${{ matrix.backend }} -v
```

## Performance Expectations

### Typical Execution Times

| Test Category   | Duration | Description                       |
| --------------- | -------- | --------------------------------- |
| Basic PyTorch   | 10-30s   | Simple regression with 10 epochs  |
| Lightning       | 15-40s   | Lightning trainer with validation |
| Computer Vision | 30-60s   | CNN with image data               |
| Time Series     | 20-45s   | LSTM forecasting                  |

### Resource Usage

- **Memory**: 1-2GB for basic tests, 2-4GB for advanced
- **Disk**: 100-500MB for temporary files and logs
- **CPU**: Single-threaded training (no GPU required)

## Contributing

### Adding New Backends

1. Create a new environment class in `framework.py`
2. Implement required abstract methods
3. Add backend to the environments dict
4. Write integration tests

### Adding New Workflows

1. Create workflow class inheriting from `TrainingWorkflow`
2. Implement training logic with proper Tracelet integration
3. Add validation and expected metrics
4. Register workflow in framework
5. Add comprehensive tests

### Best Practices

- Always test across multiple backends
- Include proper error handling
- Add performance benchmarks
- Document expected behavior
- Use realistic training scenarios
