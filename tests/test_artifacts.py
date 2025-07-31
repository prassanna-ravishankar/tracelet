"""
Tests for the artifact system.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tracelet.core.artifact_manager import ArtifactManager
from tracelet.core.artifacts import ArtifactType, TraceletArtifact


class TestTraceletArtifact:
    """Test TraceletArtifact class."""

    def test_artifact_creation(self):
        """Test basic artifact creation."""
        artifact = TraceletArtifact(name="test_model", artifact_type=ArtifactType.MODEL, description="A test model")

        assert artifact.name == "test_model"
        assert artifact.type == ArtifactType.MODEL
        assert artifact.description == "A test model"
        assert artifact.size_bytes == 0
        assert len(artifact.files) == 0
        assert len(artifact.references) == 0
        assert len(artifact.objects) == 0

    def test_add_file(self):
        """Test adding files to artifact."""
        artifact = TraceletArtifact("test", ArtifactType.MODEL)

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            artifact.add_file(temp_path, "model.txt", "Test file")

            assert len(artifact.files) == 1
            assert artifact.files[0].local_path == temp_path
            assert artifact.files[0].artifact_path == "model.txt"
            assert artifact.files[0].description == "Test file"
            assert artifact.size_bytes > 0
        finally:
            Path(temp_path).unlink()

    def test_add_file_nonexistent(self):
        """Test adding non-existent file raises error."""
        artifact = TraceletArtifact("test", ArtifactType.MODEL)

        with pytest.raises(FileNotFoundError):
            artifact.add_file("/nonexistent/file.txt")

    def test_add_reference(self):
        """Test adding external references."""
        artifact = TraceletArtifact("test", ArtifactType.DATASET)

        artifact.add_reference("s3://bucket/dataset.tar.gz", size_bytes=1000000, description="Large dataset")

        assert len(artifact.references) == 1
        assert artifact.references[0].uri == "s3://bucket/dataset.tar.gz"
        assert artifact.references[0].size_bytes == 1000000
        assert artifact.size_bytes == 1000000

    def test_add_object(self):
        """Test adding Python objects."""
        artifact = TraceletArtifact("test", ArtifactType.CONFIG)

        config = {"learning_rate": 0.01, "batch_size": 32}
        artifact.add_object(config, "config", "json", "Model configuration")

        assert len(artifact.objects) == 1
        assert artifact.objects[0].obj == config
        assert artifact.objects[0].name == "config"
        assert artifact.objects[0].serializer == "json"

    def test_add_model(self):
        """Test adding ML models."""
        artifact = TraceletArtifact("test", ArtifactType.MODEL)

        # Mock model object
        mock_model = Mock()
        mock_model.__module__ = "torch.nn"

        artifact.add_model(mock_model, framework="pytorch", description="Test model")

        assert "model_info" in artifact.metadata
        model_info = artifact.metadata["model_info"]
        assert model_info.model == mock_model
        assert model_info.framework == "pytorch"
        assert model_info.description == "Test model"

    def test_add_model_invalid_type(self):
        """Test adding model to non-model artifact raises error."""
        artifact = TraceletArtifact("test", ArtifactType.IMAGE)

        with pytest.raises(ValueError, match="add_model\\(\\) only valid for MODEL"):
            artifact.add_model(Mock())

    def test_framework_detection(self):
        """Test automatic framework detection."""
        artifact = TraceletArtifact("test", ArtifactType.MODEL)

        # Test PyTorch detection
        mock_torch_model = Mock()
        mock_torch_model.__module__ = "torch.nn.modules.linear"

        artifact.add_model(mock_torch_model)
        assert artifact.metadata["model_info"].framework == "pytorch"

        # Test sklearn detection
        mock_sklearn_model = Mock()
        mock_sklearn_model.__module__ = "sklearn.ensemble.forest"

        artifact2 = TraceletArtifact("test2", ArtifactType.MODEL)
        artifact2.add_model(mock_sklearn_model)
        assert artifact2.metadata["model_info"].framework == "sklearn"

    def test_serialize_object_pickle(self):
        """Test object serialization with pickle."""
        artifact = TraceletArtifact("test", ArtifactType.CONFIG)

        data = {"key": "value", "number": 42}
        artifact.add_object(data, "test_data", "pickle")

        with tempfile.TemporaryDirectory() as temp_dir:
            serialized_path = artifact.serialize_object(artifact.objects[0], temp_dir)

            assert Path(serialized_path).exists()
            assert serialized_path.endswith(".pickle")

            # Verify we can load it back
            import pickle

            with open(serialized_path, "rb") as f:
                loaded_data = pickle.load(f)  # noqa: S301
            assert loaded_data == data

    def test_serialize_object_json(self):
        """Test object serialization with JSON."""
        artifact = TraceletArtifact("test", ArtifactType.CONFIG)

        data = {"key": "value", "number": 42}
        artifact.add_object(data, "test_data", "json")

        with tempfile.TemporaryDirectory() as temp_dir:
            serialized_path = artifact.serialize_object(artifact.objects[0], temp_dir)

            assert Path(serialized_path).exists()
            assert serialized_path.endswith(".json")

            # Verify we can load it back
            import json

            with open(serialized_path) as f:
                loaded_data = json.load(f)
            assert loaded_data == data

    def test_to_dict(self):
        """Test artifact serialization to dictionary."""
        artifact = TraceletArtifact("test_artifact", ArtifactType.MODEL, "Test artifact for serialization")

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test")
            temp_path = f.name

        try:
            artifact.add_file(temp_path, "test.txt")
            artifact.add_reference("s3://bucket/file.txt", 1000)
            artifact.add_object({"key": "value"}, "config")

            artifact_dict = artifact.to_dict()

            assert artifact_dict["name"] == "test_artifact"
            assert artifact_dict["type"] == "model"
            assert artifact_dict["description"] == "Test artifact for serialization"
            assert len(artifact_dict["files"]) == 1
            assert len(artifact_dict["references"]) == 1
            assert len(artifact_dict["objects"]) == 1

        finally:
            Path(temp_path).unlink()


class TestArtifactManager:
    """Test ArtifactManager class."""

    def test_manager_creation(self):
        """Test artifact manager creation."""
        backend_instances = {"mlflow": Mock(), "wandb": Mock()}

        manager = ArtifactManager(backend_instances)

        assert len(manager.handlers) == 2
        assert manager.backend_instances == backend_instances

    def test_log_artifact(self):
        """Test artifact logging."""
        # Create mock backend
        mock_backend = Mock()
        backend_instances = {"test_backend": mock_backend}

        # Create mock handler
        mock_handler = Mock()
        mock_result = Mock()
        mock_handler.log_artifact.return_value = mock_result

        manager = ArtifactManager(backend_instances)
        manager.handlers = [mock_handler]  # Replace with mock handler

        # Create test artifact
        artifact = TraceletArtifact("test", ArtifactType.MODEL)

        # Log artifact
        results = manager.log_artifact(artifact)

        # Verify handler was called
        mock_handler.log_artifact.assert_called_once_with(artifact)
        assert len(results) == 1

    def test_get_stats(self):
        """Test manager statistics."""
        backend_instances = {"mlflow": Mock(), "wandb": Mock()}
        manager = ArtifactManager(backend_instances)

        stats = manager.get_stats()

        assert stats["num_handlers"] == 2
        assert "handler_types" in stats
        assert stats["backend_names"] == ["mlflow", "wandb"]
        assert "cache_size" in stats
        assert "large_file_threshold_mb" in stats


class TestArtifactIntegration:
    """Test artifact system integration."""

    def test_mlflow_handler_integration(self):
        """Test MLflow handler integration."""
        with patch.dict("sys.modules", {"mlflow": Mock()}):
            import sys

            mock_mlflow = sys.modules["mlflow"]

            from tracelet.core.artifact_handlers import MLflowArtifactHandler

            # Setup mock
            mock_backend = Mock()
            mock_run = Mock()
            mock_run.info.run_id = "test_run_123"
            mock_mlflow.active_run.return_value = mock_run

            handler = MLflowArtifactHandler(mock_backend)

            # Create test artifact
            artifact = TraceletArtifact("test_model", ArtifactType.MODEL)

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                f.write("test model data")
                temp_path = f.name

            try:
                artifact.add_file(temp_path)

                # Log artifact
                result = handler.log_artifact(artifact)

                # Verify result
                assert result.backend == "mlflow"
                assert result.version == "test_run_123"
                assert "runs:/test_run_123" in result.uri

            finally:
                Path(temp_path).unlink()

    def test_wandb_handler_integration(self):
        """Test W&B handler integration."""
        with patch.dict("sys.modules", {"wandb": Mock()}):
            import sys

            mock_wandb = sys.modules["wandb"]

            from tracelet.core.artifact_handlers import WANDBArtifactHandler

            # Setup mock
            mock_backend = Mock()
            mock_run = Mock()
            mock_run.id = "test_run_456"
            mock_wandb.run = mock_run

            handler = WANDBArtifactHandler(mock_backend)

            # Create test artifact with image
            artifact = TraceletArtifact("test_image", ArtifactType.IMAGE)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(b"fake image data")
                temp_path = f.name

            try:
                artifact.add_file(temp_path)

                # Mock W&B Image creation
                mock_image = Mock()
                mock_wandb.Image.return_value = mock_image

                # Log artifact
                result = handler.log_artifact(artifact)

                # Verify W&B Image was created
                mock_wandb.Image.assert_called_once()

                # Verify result
                assert result.backend == "wandb"
                assert "wandb://" in result.uri

            finally:
                Path(temp_path).unlink()


class TestArtifactTypes:
    """Test artifact type detection and handling."""

    def test_all_artifact_types(self):
        """Test all artifact types can be created."""
        for artifact_type in ArtifactType:
            artifact = TraceletArtifact(
                f"test_{artifact_type.value}", artifact_type, f"Test {artifact_type.value} artifact"
            )

            assert artifact.type == artifact_type
            assert artifact.name == f"test_{artifact_type.value}"

    def test_artifact_type_values(self):
        """Test artifact type string values."""
        expected_types = {
            ArtifactType.MODEL: "model",
            ArtifactType.CHECKPOINT: "checkpoint",
            ArtifactType.WEIGHTS: "weights",
            ArtifactType.DATASET: "dataset",
            ArtifactType.SAMPLE: "sample",
            ArtifactType.IMAGE: "image",
            ArtifactType.AUDIO: "audio",
            ArtifactType.VIDEO: "video",
            ArtifactType.VISUALIZATION: "viz",
            ArtifactType.REPORT: "report",
            ArtifactType.CONFIG: "config",
            ArtifactType.CODE: "code",
            ArtifactType.CUSTOM: "custom",
        }

        for artifact_type, expected_value in expected_types.items():
            assert artifact_type.value == expected_value


if __name__ == "__main__":
    pytest.main([__file__])
