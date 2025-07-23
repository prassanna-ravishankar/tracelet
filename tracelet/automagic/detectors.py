"""
Automatic detection of ML experiment components.
"""

import contextlib
import json
import warnings
from pathlib import Path
from typing import Any, Union

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class HyperparameterDetector:
    """Automatically detect hyperparameters from various sources."""

    def __init__(self, config):
        self.config = config
        self._common_hyperparam_names = {
            "lr",
            "learning_rate",
            "batch_size",
            "epochs",
            "num_epochs",
            "dropout",
            "weight_decay",
            "momentum",
            "alpha",
            "beta",
            "hidden_size",
            "hidden_dim",
            "num_layers",
            "num_heads",
            "temperature",
            "threshold",
            "gamma",
            "eps",
            "epsilon",
            "patience",
            "max_iter",
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "random_state",
            "seed",
        }

    def extract_from_frame(self, frame) -> dict[str, Any]:
        """Extract hyperparameters from a stack frame."""
        hyperparams = {}

        if not frame:
            return hyperparams

        # Get local variables from frame
        local_vars = frame.f_locals

        # Extract function arguments
        if self.config.detect_function_args:
            func_args = self._extract_function_args(frame)
            hyperparams.update(func_args)

        # Extract class attributes if we're in a method
        if self.config.detect_class_attributes:
            class_attrs = self._extract_class_attributes(local_vars)
            hyperparams.update(class_attrs)

        # Look for common hyperparameter patterns in locals
        for name, value in local_vars.items():
            if self._is_likely_hyperparameter(name, value):
                hyperparams[name] = self._serialize_value(value)

        return hyperparams

    def extract_from_argparse(self, args) -> dict[str, Any]:
        """Extract hyperparameters from argparse Namespace."""
        if not self.config.detect_argparse:
            return {}

        hyperparams = {}
        if hasattr(args, "__dict__"):
            for name, value in args.__dict__.items():
                if self._is_serializable(value):
                    hyperparams[f"args_{name}"] = value

        return hyperparams

    def extract_from_config_file(self, config_path: Union[str, Path]) -> dict[str, Any]:
        """Extract hyperparameters from configuration files."""
        if not self.config.detect_config_files:
            return {}

        config_path = Path(config_path)
        if not config_path.exists():
            return {}

        hyperparams = {}
        try:
            if config_path.suffix.lower() == ".json":
                with open(config_path) as f:
                    data = json.load(f)
                hyperparams = self._flatten_dict(data, prefix="config")

            elif config_path.suffix.lower() in [".yaml", ".yml"]:
                try:
                    import yaml

                    with open(config_path) as f:
                        data = yaml.safe_load(f)
                    hyperparams = self._flatten_dict(data, prefix="config")
                except ImportError:
                    warnings.warn("PyYAML not installed, cannot parse YAML config files", stacklevel=2)

        except Exception as e:
            warnings.warn(f"Failed to parse config file {config_path}: {e}", stacklevel=2)

        return hyperparams

    def _extract_function_args(self, frame) -> dict[str, Any]:
        """Extract function arguments from frame."""
        hyperparams = {}

        # Get function signature
        func = frame.f_code
        arg_names = func.co_varnames[: func.co_argcount]

        # Extract argument values from locals
        for name in arg_names:
            if name in frame.f_locals and name != "self":
                value = frame.f_locals[name]
                if self._is_likely_hyperparameter(name, value):
                    hyperparams[name] = self._serialize_value(value)

        return hyperparams

    def _extract_class_attributes(self, local_vars: dict[str, Any]) -> dict[str, Any]:
        """Extract hyperparameters from class attributes."""
        hyperparams = {}

        # Look for 'self' in locals (indicates we're in a method)
        if "self" in local_vars:
            obj = local_vars["self"]
            for attr_name in dir(obj):
                if not attr_name.startswith("_"):
                    with contextlib.suppress(Exception):
                        value = getattr(obj, attr_name)
                        if self._is_likely_hyperparameter(attr_name, value):
                            hyperparams[f"self.{attr_name}"] = self._serialize_value(value)

        return hyperparams

    def _is_likely_hyperparameter(self, name: str, value: Any) -> bool:
        """Heuristic to determine if a variable is likely a hyperparameter."""
        # Skip common non-hyperparameter names
        skip_names = {
            "self",
            "cls",
            "args",
            "kwargs",
            "i",
            "j",
            "k",
            "x",
            "y",
            "data",
            "target",
            "model",
            "optimizer",
            "loss",
            "device",
            "cuda",
            "cpu",
            "train",
            "test",
            "val",
            "dataset",
            "dataloader",
            "experiment",
            "config",
            "logger",
            "__",
            "_",
            "tmp",
            "temp",
            "debug",
            "verbose",
            "print",
            "len",
            "range",
            "true",
            "false",
            "none",
            "null",
            "tensor",
            "array",
            "df",
            "np",
            "torch",
            "pd",
        }

        if name.lower() in skip_names or name.startswith("_"):
            return False

        # Check if name matches common hyperparameter patterns
        if name.lower() in self._common_hyperparam_names:
            return True

        # Check if name contains hyperparameter-like terms
        hyperparam_terms = [
            "rate",
            "size",
            "dim",
            "layer",
            "epoch",
            "step",
            "decay",
            "momentum",
            "batch",
            "num",
            "max",
            "min",
            "alpha",
            "beta",
            "gamma",
            "threshold",
            "temperature",
            "patience",
            "interval",
            "factor",
            "scale",
            "weight",
            "depth",
            "width",
            "kernel",
            "stride",
            "padding",
            "dilation",
            "heads",
            "attention",
            "embed",
            "hidden",
            "feature",
            "channel",
        ]
        if any(term in name.lower() for term in hyperparam_terms):
            return True

        # Check if value type and range suggests hyperparameter
        if isinstance(value, (int, float)):
            # Reasonable hyperparameter ranges
            if isinstance(value, int) and 1 <= value <= 10000:
                return True
            if isinstance(value, float) and 0.0001 <= value <= 100:
                return True

        if isinstance(value, bool):
            return True

        return isinstance(value, str) and len(value) < 50  # Short string configs

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for logging."""
        if self._is_serializable(value):
            return value
        else:
            return str(value)

    def _is_serializable(self, value: Any) -> bool:
        """Check if a value is JSON serializable."""
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False

    def _flatten_dict(self, d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{prefix}_{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)


class ModelDetector:
    """Automatically detect and analyze ML models."""

    def __init__(self, config):
        self.config = config

    def analyze_model(self, model: Any) -> dict[str, Any]:
        """Analyze a model and extract metadata."""
        model_info = {
            "type": type(model).__name__,
            "module": type(model).__module__,
        }

        # PyTorch model analysis
        if HAS_TORCH and isinstance(model, nn.Module):
            model_info.update(self._analyze_torch_model(model))

        # Scikit-learn model analysis
        elif hasattr(model, "fit") and hasattr(model, "predict"):
            model_info.update(self._analyze_sklearn_model(model))

        # Generic analysis
        else:
            model_info.update(self._analyze_generic_model(model))

        return model_info

    def _analyze_torch_model(self, model: "nn.Module") -> dict[str, Any]:
        """Analyze PyTorch model."""
        info = {}

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        info.update({
            "parameter_count": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
        })

        # Model architecture
        if self.config.track_model_architecture:
            info["architecture"] = str(model)

        # Layer information
        layers = []
        for name, module in model.named_modules():
            if name:  # Skip the root module
                layers.append({
                    "name": name,
                    "type": type(module).__name__,
                    "parameters": sum(p.numel() for p in module.parameters()),
                })
        info["layers"] = layers

        return info

    def _analyze_sklearn_model(self, model: Any) -> dict[str, Any]:
        """Analyze scikit-learn model."""
        info = {}

        # Extract hyperparameters
        if hasattr(model, "get_params"):
            params = model.get_params()
            info["hyperparameters"] = {k: v for k, v in params.items() if self._is_serializable(v)}

        # Model-specific info
        if hasattr(model, "feature_importances_"):
            info["has_feature_importances"] = True

        if hasattr(model, "coef_"):
            info["has_coefficients"] = True

        return info

    def _analyze_generic_model(self, model: Any) -> dict[str, Any]:
        """Analyze generic model object."""
        info = {}

        # Basic reflection
        if hasattr(model, "__dict__"):
            attributes = {}
            for k, v in model.__dict__.items():
                if not k.startswith("_") and self._is_serializable(v):
                    attributes[k] = v
            if attributes:
                info["attributes"] = attributes

        return info

    def _is_serializable(self, value: Any) -> bool:
        """Check if a value is JSON serializable."""
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False


class DatasetDetector:
    """Automatically detect and analyze datasets."""

    def __init__(self, config):
        self.config = config

    def analyze_dataset(self, dataset: Any) -> dict[str, Any]:
        """Analyze a dataset and extract metadata."""
        dataset_info = {
            "type": type(dataset).__name__,
            "module": type(dataset).__module__,
        }

        # PyTorch dataset analysis
        if HAS_TORCH and hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
            dataset_info.update(self._analyze_torch_dataset(dataset))

        # NumPy array analysis
        elif HAS_NUMPY and isinstance(dataset, np.ndarray):
            dataset_info.update(self._analyze_numpy_array(dataset))

        # Generic iterable analysis
        elif hasattr(dataset, "__len__"):
            dataset_info.update(self._analyze_generic_dataset(dataset))

        return dataset_info

    def _analyze_torch_dataset(self, dataset) -> dict[str, Any]:
        """Analyze PyTorch dataset."""
        info = {"size": len(dataset)}

        # Analyze sample structure
        if len(dataset) > 0:
            try:
                sample = dataset[0]
                info["sample_structure"] = self._describe_sample_structure(sample)
            except Exception as e:
                info["sample_error"] = str(e)

        return info

    def _analyze_numpy_array(self, array: "np.ndarray") -> dict[str, Any]:
        """Analyze NumPy array."""
        return {
            "shape": array.shape,
            "dtype": str(array.dtype),
            "size": array.size,
            "ndim": array.ndim,
        }

    def _analyze_generic_dataset(self, dataset) -> dict[str, Any]:
        """Analyze generic dataset."""
        info = {}

        if hasattr(dataset, "__len__"):
            info["size"] = len(dataset)

        # Try to get sample if possible
        if hasattr(dataset, "__getitem__") and len(dataset) > 0:
            with contextlib.suppress(Exception):
                sample = dataset[0]
                info["sample_structure"] = self._describe_sample_structure(sample)

        return info

    def _describe_sample_structure(self, sample: Any) -> dict[str, Any]:
        """Describe the structure of a sample."""
        if HAS_TORCH and torch.is_tensor(sample):
            return {
                "type": "tensor",
                "shape": list(sample.shape),
                "dtype": str(sample.dtype),
            }
        elif HAS_NUMPY and isinstance(sample, np.ndarray):
            return {
                "type": "numpy",
                "shape": list(sample.shape),
                "dtype": str(sample.dtype),
            }
        elif isinstance(sample, (tuple, list)):
            return {
                "type": "sequence",
                "length": len(sample),
                "element_types": [type(x).__name__ for x in sample],
            }
        elif isinstance(sample, dict):
            return {
                "type": "dict",
                "keys": list(sample.keys()),
            }
        else:
            return {
                "type": type(sample).__name__,
            }
