"""
Basic example showing how to use Tracelet with different backends.

Before running this example, install the appropriate backend:
- pip install tracelet[mlflow]   # for MLflow
- pip install tracelet[clearml]  # for ClearML
- pip install tracelet[wandb]    # for Weights & Biases
"""

import tracelet


# Example 1: Basic usage with MLflow (local file store)
def example_mlflow():
    """Example using MLflow backend"""
    try:
        experiment = tracelet.start_logging(
            exp_name="basic_example_mlflow", project="tracelet_examples", backend="mlflow"
        )

        # Your ML code here - metrics will be automatically captured
        print("Started MLflow experiment")

        # Manual metric logging is also supported
        experiment.log_metric("example_metric", 0.95, iteration=1)

        tracelet.stop_logging()
        print("✅ MLflow example completed")

    except ImportError:
        print("❌ MLflow not installed. Run: pip install tracelet[mlflow]")


# Example 2: Basic usage with ClearML
def example_clearml():
    """Example using ClearML backend"""
    try:
        experiment = tracelet.start_logging(
            exp_name="basic_example_clearml", project="tracelet_examples", backend="clearml"
        )

        print("Started ClearML experiment")

        # Manual metric logging
        experiment.log_metric("example_metric", 0.88, iteration=1)

        tracelet.stop_logging()
        print("✅ ClearML example completed")

    except ImportError:
        print("❌ ClearML not installed. Run: pip install tracelet[clearml]")


# Example 3: Environment variable configuration
def example_env_config():
    """Example using environment variables for configuration"""
    import os

    # Set environment variables
    os.environ["TRACELET_PROJECT"] = "tracelet_examples"
    os.environ["TRACELET_BACKEND"] = "mlflow"

    try:
        # No need to pass parameters - they'll be read from env vars
        experiment = tracelet.start_logging(exp_name="env_config_example")

        print("Started experiment with env var config")
        experiment.log_metric("env_metric", 0.77, iteration=1)

        tracelet.stop_logging()
        print("✅ Environment config example completed")

    except ImportError as e:
        print(f"❌ Backend not installed: {e}")


if __name__ == "__main__":
    print("🚀 Tracelet Basic Usage Examples")
    print("=" * 40)

    print("\n1. MLflow Example:")
    example_mlflow()

    print("\n2. ClearML Example:")
    example_clearml()

    print("\n3. Environment Config Example:")
    example_env_config()

    print("\n✨ Examples complete!")
