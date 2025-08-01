"""
Platform-specific artifact handlers for different MLOps backends.

Each handler optimizes artifact logging for their respective platform's strengths:
- MLflow: Models, checkpoints, general files
- W&B: Rich media, datasets, visualizations
- ClearML: Auto-detection, dynamic updates
- AIM: Lightweight tracking with file references
"""

import contextlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import ClassVar, Optional

from .artifacts import (
    LARGE_FILE_THRESHOLD,
    ArtifactHandler,
    ArtifactResult,
    ArtifactType,
    TraceletArtifact,
)

logger = logging.getLogger(__name__)


class MLflowArtifactHandler(ArtifactHandler):
    """MLflow-optimized artifact handling."""

    # Optimal artifact types for MLflow
    OPTIMAL_TYPES: ClassVar = {
        ArtifactType.MODEL,
        ArtifactType.CHECKPOINT,
        ArtifactType.WEIGHTS,
        ArtifactType.CONFIG,
        ArtifactType.CODE,
    }

    def should_handle_type(self, artifact_type: ArtifactType) -> bool:
        return artifact_type in self.OPTIMAL_TYPES

    def get_large_file_score(self) -> int:
        return 3  # Excellent large file support with multipart upload

    def log_artifact(self, artifact: TraceletArtifact) -> ArtifactResult:
        """Log artifact using MLflow's APIs."""
        try:
            # Handle model artifacts with MLflow's model APIs
            if artifact.type == ArtifactType.MODEL and "model_info" in artifact.metadata:
                return self._log_model_artifact(artifact)
            else:
                return self._log_standard_artifact(artifact)

        except Exception as e:
            logger.exception(f"MLflow artifact logging failed: {e}")
            raise

    def _log_model_artifact(self, artifact: TraceletArtifact) -> ArtifactResult:
        """Use MLflow's model logging with flavor detection."""
        import mlflow

        model_info = artifact.metadata["model_info"]
        framework = model_info.framework

        # Use framework-specific model logging
        if framework == "pytorch":
            try:
                import mlflow.pytorch

                mlflow.pytorch.log_model(
                    pytorch_model=model_info.model,
                    artifact_path=artifact.name,
                    input_example=model_info.input_example,
                    signature=model_info.signature,
                )
            except ImportError:
                logger.warning("PyTorch not available, falling back to standard artifact logging")
                return self._log_standard_artifact(artifact)

        elif framework == "sklearn":
            try:
                import mlflow.sklearn

                mlflow.sklearn.log_model(
                    sk_model=model_info.model,
                    artifact_path=artifact.name,
                    input_example=model_info.input_example,
                    signature=model_info.signature,
                )
            except ImportError:
                logger.warning("Scikit-learn not available, falling back to standard artifact logging")
                return self._log_standard_artifact(artifact)

        elif framework == "pytorch_lightning":
            # Log as PyTorch model
            try:
                import mlflow.pytorch

                mlflow.pytorch.log_model(
                    pytorch_model=model_info.model,
                    artifact_path=artifact.name,
                    input_example=model_info.input_example,
                    signature=model_info.signature,
                )
            except ImportError:
                return self._log_standard_artifact(artifact)
        else:
            # Fallback to standard artifact logging
            return self._log_standard_artifact(artifact)

        # Log additional files if any
        for file_info in artifact.files:
            mlflow.log_artifact(file_info.local_path, artifact.name)

        run = mlflow.active_run()
        return ArtifactResult(
            version=run.info.run_id if run else "unknown",
            uri=f"runs:/{run.info.run_id}/{artifact.name}" if run else f"artifact:{artifact.name}",
            size_bytes=artifact.size_bytes,
            backend="mlflow",
            metadata={"framework": framework},
        )

    def _log_standard_artifact(self, artifact: TraceletArtifact) -> ArtifactResult:
        """Log artifact using standard MLflow artifact APIs."""
        import mlflow

        # Serialize objects to temporary files
        temp_files = []
        if artifact.objects:
            temp_dir = tempfile.mkdtemp(prefix="tracelet_artifacts_")
            for obj_info in artifact.objects:
                temp_path = artifact.serialize_object(obj_info, temp_dir)
                temp_files.append(temp_path)
                mlflow.log_artifact(temp_path, artifact.name)

        # Log files
        for file_info in artifact.files:
            if artifact.size_bytes > LARGE_FILE_THRESHOLD:
                # For large files, log directly (MLflow handles multipart internally)
                mlflow.log_artifact(file_info.local_path, artifact.name)
            else:
                mlflow.log_artifact(file_info.local_path, artifact.name)

        # Log references as text file with URIs
        if artifact.references:
            refs_content = "\n".join([f"{ref.uri}" for ref in artifact.references])
            refs_path = os.path.join(tempfile.gettempdir(), f"{artifact.name}_references.txt")
            with open(refs_path, "w") as f:
                f.write(refs_content)
            mlflow.log_artifact(refs_path, artifact.name)
            temp_files.append(refs_path)

        # Cleanup temp files
        for temp_file in temp_files:
            with contextlib.suppress(OSError):
                os.remove(temp_file)

        run = mlflow.active_run()
        return ArtifactResult(
            version=run.info.run_id if run else "unknown",
            uri=f"runs:/{run.info.run_id}/{artifact.name}" if run else f"artifact:{artifact.name}",
            size_bytes=artifact.size_bytes,
            backend="mlflow",
        )

    def get_artifact(self, name: str, version: str = "latest") -> Optional[TraceletArtifact]:
        """Retrieve artifact from MLflow."""
        # Implementation would retrieve from MLflow artifact store
        logger.warning("MLflow artifact retrieval not yet implemented")
        return None

    def list_artifacts(self, type_filter: Optional[ArtifactType] = None) -> list[TraceletArtifact]:
        """List MLflow artifacts."""
        logger.warning("MLflow artifact listing not yet implemented")
        return []


class WANDBArtifactHandler(ArtifactHandler):
    """W&B-optimized artifact handling."""

    # Optimal artifact types for W&B
    OPTIMAL_TYPES: ClassVar = {
        ArtifactType.IMAGE,
        ArtifactType.AUDIO,
        ArtifactType.VIDEO,
        ArtifactType.DATASET,
        ArtifactType.VISUALIZATION,
        ArtifactType.SAMPLE,
    }

    def should_handle_type(self, artifact_type: ArtifactType) -> bool:
        return artifact_type in self.OPTIMAL_TYPES

    def get_large_file_score(self) -> int:
        return 2  # Good large file support with reference artifacts

    def log_artifact(self, artifact: TraceletArtifact) -> ArtifactResult:
        """Log artifact using W&B's APIs."""
        try:
            # Handle rich media types with W&B's media APIs
            if artifact.type in {ArtifactType.IMAGE, ArtifactType.AUDIO, ArtifactType.VIDEO}:
                return self._log_media_artifact(artifact)
            elif artifact.size_bytes > LARGE_FILE_THRESHOLD:
                return self._log_reference_artifact(artifact)
            else:
                return self._log_wandb_artifact(artifact)

        except Exception as e:
            logger.exception(f"W&B artifact logging failed: {e}")
            raise

    def _log_media_artifact(self, artifact: TraceletArtifact) -> ArtifactResult:
        """Use W&B's rich media APIs."""
        import wandb

        media_objects = {}

        for file_info in artifact.files:
            if artifact.type == ArtifactType.IMAGE:
                try:
                    media_objects[file_info.artifact_path] = wandb.Image(
                        file_info.local_path, caption=file_info.description
                    )
                except Exception as e:
                    logger.warning(f"Failed to create W&B Image: {e}")

            elif artifact.type == ArtifactType.AUDIO:
                try:
                    media_objects[file_info.artifact_path] = wandb.Audio(
                        file_info.local_path, caption=file_info.description
                    )
                except Exception as e:
                    logger.warning(f"Failed to create W&B Audio: {e}")

            elif artifact.type == ArtifactType.VIDEO:
                try:
                    media_objects[file_info.artifact_path] = wandb.Video(
                        file_info.local_path, caption=file_info.description
                    )
                except Exception as e:
                    logger.warning(f"Failed to create W&B Video: {e}")

        # Log as rich media
        if media_objects:
            wandb.log({artifact.name: media_objects})

        run = wandb.run
        return ArtifactResult(
            version="logged_as_media",
            uri=f"wandb://{run.id}/{artifact.name}" if run else f"wandb://unknown/{artifact.name}",
            size_bytes=artifact.size_bytes,
            backend="wandb",
            metadata={"logged_as": "media"},
        )

    def _log_reference_artifact(self, artifact: TraceletArtifact) -> ArtifactResult:
        """Log large files as reference artifacts."""
        import wandb

        wandb_artifact = wandb.Artifact(name=artifact.name, type=artifact.type.value, description=artifact.description)

        # Add references
        for ref in artifact.references:
            wandb_artifact.add_reference(ref.uri, name=f"ref_{len(wandb_artifact._manifest.entries)}")

        # Add files as references if large
        for file_info in artifact.files:
            if file_info.size_bytes > LARGE_FILE_THRESHOLD:
                # For very large files, could add as reference if already in cloud storage
                wandb_artifact.add_file(file_info.local_path, name=file_info.artifact_path)
            else:
                wandb_artifact.add_file(file_info.local_path, name=file_info.artifact_path)

        # Log artifact
        if wandb.run:
            wandb.run.log_artifact(wandb_artifact)

        return ArtifactResult(
            version=wandb_artifact.version if hasattr(wandb_artifact, "version") else "v0",
            uri=f"wandb://artifact/{artifact.name}",
            size_bytes=artifact.size_bytes,
            backend="wandb",
            metadata={"artifact_type": artifact.type.value},
        )

    def _log_wandb_artifact(self, artifact: TraceletArtifact) -> ArtifactResult:
        """Log standard W&B artifact."""
        import wandb

        wandb_artifact = wandb.Artifact(
            name=artifact.name, type=artifact.type.value, description=artifact.description, metadata=artifact.metadata
        )

        # Add files
        for file_info in artifact.files:
            wandb_artifact.add_file(file_info.local_path, name=file_info.artifact_path)

        # Serialize and add objects
        if artifact.objects:
            temp_dir = tempfile.mkdtemp(prefix="tracelet_wandb_")
            for obj_info in artifact.objects:
                temp_path = artifact.serialize_object(obj_info, temp_dir)
                wandb_artifact.add_file(temp_path, name=f"objects/{obj_info.name}")

        # Log artifact
        if wandb.run:
            wandb.run.log_artifact(wandb_artifact)

        return ArtifactResult(
            version=wandb_artifact.version if hasattr(wandb_artifact, "version") else "v0",
            uri=f"wandb://artifact/{artifact.name}",
            size_bytes=artifact.size_bytes,
            backend="wandb",
        )

    def get_artifact(self, name: str, version: str = "latest") -> Optional[TraceletArtifact]:
        """Retrieve artifact from W&B."""
        logger.warning("W&B artifact retrieval not yet implemented")
        return None

    def list_artifacts(self, type_filter: Optional[ArtifactType] = None) -> list[TraceletArtifact]:
        """List W&B artifacts."""
        logger.warning("W&B artifact listing not yet implemented")
        return []


class ClearMLArtifactHandler(ArtifactHandler):
    """ClearML-optimized artifact handling."""

    # Optimal artifact types for ClearML
    OPTIMAL_TYPES: ClassVar = {ArtifactType.MODEL, ArtifactType.CHECKPOINT, ArtifactType.REPORT}

    def should_handle_type(self, artifact_type: ArtifactType) -> bool:
        return artifact_type in self.OPTIMAL_TYPES

    def get_large_file_score(self) -> int:
        return 2  # Good large file support with direct access

    def log_artifact(self, artifact: TraceletArtifact) -> ArtifactResult:
        """Log artifact using ClearML's APIs."""
        try:
            # Access ClearML backend instance
            if not hasattr(self.backend, "_task") or not self.backend._task:
                raise RuntimeError("ClearML task not initialized")

            task = self.backend._task

            if artifact.type == ArtifactType.MODEL and "model_info" in artifact.metadata:
                return self._log_model_with_auto_detection(artifact, task)
            elif artifact.type == ArtifactType.REPORT:
                return self._log_report_artifact(artifact, task)
            else:
                return self._log_clearml_artifact(artifact, task)

        except Exception as e:
            logger.exception(f"ClearML artifact logging failed: {e}")
            raise

    def _log_model_with_auto_detection(self, artifact: TraceletArtifact, task) -> ArtifactResult:
        """Leverage ClearML's automatic model detection."""
        model_info = artifact.metadata["model_info"]

        # For PyTorch models, save and upload
        if model_info.framework in ["pytorch", "pytorch_lightning"]:
            try:
                import torch

                temp_dir = tempfile.mkdtemp(prefix="tracelet_clearml_")
                model_path = os.path.join(temp_dir, f"{artifact.name}.pth")

                # Save model state
                if hasattr(model_info.model, "state_dict"):
                    torch.save(model_info.model.state_dict(), model_path)
                else:
                    torch.save(model_info.model, model_path)

                # Upload as artifact
                task.upload_artifact(name=artifact.name, artifact_object=model_path)

                # Cleanup
                os.remove(model_path)
                os.rmdir(temp_dir)

            except Exception as e:
                logger.warning(f"PyTorch model save failed: {e}")
                return self._log_clearml_artifact(artifact, task)
        else:
            # Fallback to standard artifact logging
            return self._log_clearml_artifact(artifact, task)

        return ArtifactResult(
            version=str(int(time.time())),
            uri=f"clearml://{task.id}/{artifact.name}",
            size_bytes=artifact.size_bytes,
            backend="clearml",
            metadata={"framework": model_info.framework},
        )

    def _log_report_artifact(self, artifact: TraceletArtifact, task) -> ArtifactResult:
        """Log report artifacts with ClearML's reporting."""
        # Upload files as artifacts
        for file_info in artifact.files:
            task.upload_artifact(
                name=f"{artifact.name}_{file_info.artifact_path}", artifact_object=file_info.local_path
            )

        return ArtifactResult(
            version=str(int(time.time())),
            uri=f"clearml://{task.id}/{artifact.name}",
            size_bytes=artifact.size_bytes,
            backend="clearml",
            metadata={"type": "report"},
        )

    def _log_clearml_artifact(self, artifact: TraceletArtifact, task) -> ArtifactResult:
        """Log standard ClearML artifact."""
        # Upload files
        for file_info in artifact.files:
            task.upload_artifact(name=file_info.artifact_path, artifact_object=file_info.local_path)

        # Serialize and upload objects
        if artifact.objects:
            temp_dir = tempfile.mkdtemp(prefix="tracelet_clearml_objects_")
            for obj_info in artifact.objects:
                temp_path = artifact.serialize_object(obj_info, temp_dir)
                task.upload_artifact(name=f"{artifact.name}_{obj_info.name}", artifact_object=temp_path)

        return ArtifactResult(
            version=str(int(time.time())),
            uri=f"clearml://{task.id}/{artifact.name}",
            size_bytes=artifact.size_bytes,
            backend="clearml",
        )

    def get_artifact(self, name: str, version: str = "latest") -> Optional[TraceletArtifact]:
        """Retrieve artifact from ClearML."""
        logger.warning("ClearML artifact retrieval not yet implemented")
        return None

    def list_artifacts(self, type_filter: Optional[ArtifactType] = None) -> list[TraceletArtifact]:
        """List ClearML artifacts."""
        logger.warning("ClearML artifact listing not yet implemented")
        return []


class AiMArtifactHandler(ArtifactHandler):
    """AIM-optimized artifact handling."""

    def should_handle_type(self, artifact_type: ArtifactType) -> bool:
        # AIM handles all types but isn't optimal for any specific type
        return True

    def get_large_file_score(self) -> int:
        return 1  # Basic large file support with file references

    def log_artifact(self, artifact: TraceletArtifact) -> ArtifactResult:
        """Log artifact using AIM's APIs."""
        try:
            if not hasattr(self.backend, "_run") or not self.backend._run:
                raise RuntimeError("AIM run not initialized")

            run = self.backend._run

            # Store file references as parameters
            for file_info in artifact.files:
                param_name = f"artifacts/{artifact.name}/{file_info.artifact_path}"
                run.set(param_name, file_info.local_path, strict=False)

                # For small text files, embed content
                if self._is_small_text_file(file_info.local_path):
                    try:
                        import aim

                        content = Path(file_info.local_path).read_text()
                        run.track(aim.Text(content), name=f"artifact_content/{artifact.name}/{file_info.artifact_path}")
                    except Exception as e:
                        logger.warning(f"Failed to embed text content: {e}")

            # Store references
            for ref in artifact.references:
                param_name = f"artifact_refs/{artifact.name}/{ref.uri}"
                run.set(param_name, ref.uri, strict=False)

            # Store metadata
            for key, value in artifact.metadata.items():
                run.set(f"artifact_metadata/{artifact.name}/{key}", str(value), strict=False)

            return ArtifactResult(
                version="stored_as_reference",
                uri=f"aim://{artifact.name}",
                size_bytes=artifact.size_bytes,
                backend="aim",
                metadata={"stored_as": "reference"},
            )

        except Exception as e:
            logger.exception(f"AIM artifact logging failed: {e}")
            raise

    def _is_small_text_file(self, file_path: str) -> bool:
        """Check if file is small text file suitable for embedding."""
        try:
            path = Path(file_path)
            if path.suffix.lower() in [".txt", ".json", ".yaml", ".yml", ".md"]:
                return path.stat().st_size < 1024 * 1024  # 1MB limit
        except Exception as e:
            logger.debug(f"Failed to check small text file: {e}")
        return False

    def get_artifact(self, name: str, version: str = "latest") -> Optional[TraceletArtifact]:
        """Retrieve artifact from AIM."""
        logger.warning("AIM artifact retrieval not yet implemented")
        return None

    def list_artifacts(self, type_filter: Optional[ArtifactType] = None) -> list[TraceletArtifact]:
        """List AIM artifacts."""
        logger.warning("AIM artifact listing not yet implemented")
        return []
