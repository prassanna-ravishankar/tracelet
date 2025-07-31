"""
Core artifact management system for Tracelet.

This module provides a unified interface for handling artifacts (models, checkpoints,
images, audio, etc.) across different MLOps platforms with intelligent routing and
platform-specific optimizations.
"""

import json
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import yaml


class ArtifactType(Enum):
    """Semantic artifact types for intelligent routing."""

    # ML Assets
    MODEL = "model"  # Trained models (pytorch, sklearn, etc.)
    CHECKPOINT = "checkpoint"  # Training checkpoints (.ckpt, .pth)
    WEIGHTS = "weights"  # Model weights only

    # Data Assets
    DATASET = "dataset"  # Training/validation datasets
    SAMPLE = "sample"  # Evaluation samples/predictions

    # Media Assets
    IMAGE = "image"  # Single images or batches
    AUDIO = "audio"  # Audio files or arrays
    VIDEO = "video"  # Video files

    # Analysis Assets
    VISUALIZATION = "viz"  # Plots, charts, attention maps
    REPORT = "report"  # HTML reports, notebooks

    # Configuration Assets
    CONFIG = "config"  # Configuration files (.yaml, .json)
    CODE = "code"  # Source code snapshots

    # General Assets
    CUSTOM = "custom"  # User-defined artifacts


@dataclass
class ArtifactFile:
    """Represents a file within an artifact."""

    local_path: str
    artifact_path: str
    description: Optional[str] = None
    size_bytes: int = 0
    checksum: Optional[str] = None


@dataclass
class ArtifactReference:
    """Represents an external storage reference."""

    uri: str
    size_bytes: Optional[int] = None
    description: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactObject:
    """Represents a Python object to be serialized."""

    obj: Any
    name: str
    serializer: str = "pickle"
    description: Optional[str] = None


@dataclass
class ModelInfo:
    """Information about ML models."""

    model: Any
    framework: str
    input_example: Optional[Any] = None
    signature: Optional[Any] = None
    description: Optional[str] = None


@dataclass
class ArtifactResult:
    """Result of artifact logging operation."""

    version: str
    uri: str
    size_bytes: int
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)


class TraceletArtifact:
    """Unified artifact representation across all backends."""

    def __init__(
        self,
        name: str,
        artifact_type: ArtifactType,
        description: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.name = name
        self.type = artifact_type
        self.description = description
        self.metadata = metadata or {}

        # Tracking fields
        self.version: Optional[str] = None
        self.size_bytes = 0
        self.created_at = time.time()

        # Content containers
        self.files: list[ArtifactFile] = []
        self.references: list[ArtifactReference] = []
        self.objects: list[ArtifactObject] = []

    def add_file(
        self, local_path: Union[str, Path], artifact_path: Optional[str] = None, description: Optional[str] = None
    ) -> "TraceletArtifact":
        """Add local file to artifact."""
        file_path = Path(local_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")

        size = file_path.stat().st_size
        self.files.append(
            ArtifactFile(
                local_path=str(file_path),
                artifact_path=artifact_path or file_path.name,
                description=description,
                size_bytes=size,
            )
        )
        self.size_bytes += size
        return self

    def add_reference(
        self,
        uri: str,
        size_bytes: Optional[int] = None,
        description: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "TraceletArtifact":
        """Add external storage reference."""
        self.references.append(
            ArtifactReference(uri=uri, size_bytes=size_bytes, description=description, metadata=metadata or {})
        )
        if size_bytes:
            self.size_bytes += size_bytes
        return self

    def add_object(
        self, obj: Any, name: str, serializer: str = "pickle", description: Optional[str] = None
    ) -> "TraceletArtifact":
        """Add Python object with automatic serialization."""
        self.objects.append(ArtifactObject(obj=obj, name=name, serializer=serializer, description=description))
        return self

    def add_model(
        self,
        model: Any,
        framework: Optional[str] = None,
        input_example: Optional[Any] = None,
        signature: Optional[Any] = None,
        description: Optional[str] = None,
    ) -> "TraceletArtifact":
        """Add ML model with framework-specific handling."""
        if self.type not in [ArtifactType.MODEL, ArtifactType.CHECKPOINT, ArtifactType.WEIGHTS]:
            raise ValueError("add_model() only valid for MODEL, CHECKPOINT, or WEIGHTS artifacts")

        model_info = ModelInfo(
            model=model,
            framework=framework or self._detect_framework(model),
            input_example=input_example,
            signature=signature,
            description=description,
        )

        self.metadata["model_info"] = model_info
        return self

    def _detect_framework(self, model: Any) -> str:
        """Auto-detect ML framework."""
        if hasattr(model, "__module__"):
            module = model.__module__
            if "torch" in module:
                if "lightning" in module:
                    return "pytorch_lightning"
                return "pytorch"
            elif "sklearn" in module:
                return "sklearn"
            elif "tensorflow" in module or "keras" in module:
                return "tensorflow"
            elif "xgboost" in module:
                return "xgboost"
        return "unknown"

    def serialize_object(self, obj_info: ArtifactObject, temp_dir: str) -> str:
        """Serialize object to temporary file."""
        temp_path = Path(temp_dir) / f"{obj_info.name}.{obj_info.serializer}"
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        if obj_info.serializer == "pickle":
            with open(temp_path, "wb") as f:
                pickle.dump(obj_info.obj, f)
        elif obj_info.serializer == "json":
            with open(temp_path, "w") as f:
                json.dump(obj_info.obj, f, indent=2)
        elif obj_info.serializer == "yaml":
            with open(temp_path, "w") as f:
                yaml.dump(obj_info.obj, f)
        else:
            raise ValueError(f"Unsupported serializer: {obj_info.serializer}")

        return str(temp_path)

    def to_dict(self) -> dict[str, Any]:
        """Convert artifact to dictionary representation."""
        return {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "metadata": self.metadata,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "files": [
                {
                    "local_path": f.local_path,
                    "artifact_path": f.artifact_path,
                    "description": f.description,
                    "size_bytes": f.size_bytes,
                }
                for f in self.files
            ],
            "references": [
                {"uri": r.uri, "size_bytes": r.size_bytes, "description": r.description, "metadata": r.metadata}
                for r in self.references
            ],
            "objects": [
                {"name": o.name, "serializer": o.serializer, "description": o.description} for o in self.objects
            ],
        }


class ArtifactHandler(ABC):
    """Abstract base for platform-specific artifact handling."""

    def __init__(self, backend_instance: Any):
        self.backend = backend_instance

    @abstractmethod
    def log_artifact(self, artifact: TraceletArtifact) -> ArtifactResult:
        """Log artifact and return platform-specific result."""

    @abstractmethod
    def get_artifact(self, name: str, version: str = "latest") -> Optional[TraceletArtifact]:
        """Retrieve artifact by name/version."""

    @abstractmethod
    def list_artifacts(self, type_filter: Optional[ArtifactType] = None) -> list[TraceletArtifact]:
        """List available artifacts."""

    def should_handle_type(self, artifact_type: ArtifactType) -> bool:
        """Return True if this handler is optimal for the artifact type."""
        return True  # Default: handle all types

    def get_large_file_score(self) -> int:
        """Score for large file handling capability (higher = better)."""
        return 1  # Default score


# File size threshold for large file optimization (100MB)
LARGE_FILE_THRESHOLD = 100 * 1024 * 1024
