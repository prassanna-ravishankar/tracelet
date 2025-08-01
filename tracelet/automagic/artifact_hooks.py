"""
Framework-specific artifact hooks for automatic artifact detection.

These hooks integrate with ML frameworks to automatically detect and log
artifacts like models, checkpoints, and evaluation samples without requiring
explicit user code changes.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml

from ..core.artifact_manager import ArtifactManager
from ..core.artifacts import ArtifactType, TraceletArtifact
from .hooks import FrameworkHook

logger = logging.getLogger(__name__)


class LightningArtifactHook(FrameworkHook):
    """Automatic artifact detection for PyTorch Lightning."""

    def __init__(self, artifact_manager: ArtifactManager):
        super().__init__()
        self.artifact_manager = artifact_manager
        self.logged_checkpoints: set[str] = set()
        self.sample_frequency = 100  # Log samples every N batches

        # Ensure matplotlib is available for visualizations
        try:
            import matplotlib

            matplotlib.use("Agg")  # Use non-interactive backend
        except ImportError:
            logger.warning("Matplotlib not available for artifact visualization")

    def apply_hook(self):
        """Apply all Lightning artifact hooks."""
        try:
            import pytorch_lightning as pl

            self._hook_checkpoint_saving(pl)
            self._hook_model_export(pl)
            self._hook_validation_samples(pl)

            logger.info("Lightning artifact hooks applied successfully")

        except ImportError:
            logger.warning("PyTorch Lightning not available, skipping artifact hooks")
        except Exception as e:
            logger.exception(f"Failed to apply Lightning artifact hooks: {e}")

    def _hook_checkpoint_saving(self, pl):
        """Auto-log Lightning checkpoints."""
        if not hasattr(pl.callbacks, "ModelCheckpoint"):
            logger.warning("ModelCheckpoint callback not found")
            return

        original_save = pl.callbacks.ModelCheckpoint._save_checkpoint

        def patched_save(callback_self, trainer, pl_module, filepath):
            result = original_save(callback_self, trainer, pl_module, filepath)

            try:
                # Determine checkpoint type
                is_best = getattr(callback_self, "best_model_path", None) == filepath
                is_last = getattr(callback_self, "last_model_path", None) == filepath

                checkpoint_type = "best" if is_best else "last" if is_last else "periodic"

                if filepath not in self.logged_checkpoints:
                    self._log_checkpoint_artifact(filepath, checkpoint_type, trainer, pl_module)
                    self.logged_checkpoints.add(filepath)

            except Exception as e:
                logger.exception(f"Failed to log checkpoint artifact: {e}")

            return result

        pl.callbacks.ModelCheckpoint._save_checkpoint = patched_save

    def _log_checkpoint_artifact(self, filepath: str, checkpoint_type: str, trainer, pl_module):
        """Log checkpoint as artifact."""
        artifact = TraceletArtifact(
            name=f"checkpoint_{checkpoint_type}_epoch_{trainer.current_epoch}",
            type=ArtifactType.CHECKPOINT,
            description=f"{checkpoint_type.title()} checkpoint at epoch {trainer.current_epoch}, step {trainer.global_step}",
        )

        # Add checkpoint file
        artifact.add_file(filepath, f"checkpoints/{Path(filepath).name}")

        # Add model metadata
        artifact.add_model(
            model=pl_module,
            framework="pytorch_lightning",
            description=f"Lightning module at epoch {trainer.current_epoch}",
        )

        # Add training metadata
        artifact.metadata.update({
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "checkpoint_type": checkpoint_type,
            "val_loss": trainer.callback_metrics.get("val_loss"),
            "train_loss": trainer.callback_metrics.get("train_loss"),
            "learning_rate": self._get_learning_rate(trainer),
        })

        self.artifact_manager.log_artifact(artifact)
        logger.info(f"Logged {checkpoint_type} checkpoint artifact: {artifact.name}")

    def _get_learning_rate(self, trainer) -> Optional[float]:
        """Extract current learning rate from trainer."""
        try:
            if trainer.optimizers and len(trainer.optimizers) > 0:
                optimizer = trainer.optimizers[0]
                if hasattr(optimizer, "param_groups") and len(optimizer.param_groups) > 0:
                    return optimizer.param_groups[0].get("lr")
        except Exception as e:
            logger.debug(f"Failed to get learning rate: {e}")
        return None

    def _hook_model_export(self, pl):
        """Auto-export final model at training end."""
        original_teardown = pl.Trainer.teardown

        def patched_teardown(trainer_self, *args, **kwargs):
            result = original_teardown(trainer_self, *args, **kwargs)

            try:
                # Export final model if training completed successfully
                if (
                    hasattr(trainer_self, "lightning_module")
                    and hasattr(trainer_self, "state")
                    and trainer_self.state.finished
                ):
                    self._export_final_model(trainer_self, trainer_self.lightning_module)
            except Exception as e:
                logger.exception(f"Failed to export final model: {e}")

            return result

        pl.Trainer.teardown = patched_teardown

    def _export_final_model(self, trainer, pl_module):
        """Export final trained model as artifact."""
        try:
            artifact = TraceletArtifact(
                name=f"final_model_epoch_{trainer.current_epoch}",
                type=ArtifactType.MODEL,
                description=f"Final trained model after {trainer.current_epoch} epochs",
            )

            # Create temporary directory for model artifacts
            temp_dir = tempfile.mkdtemp(prefix="tracelet_lightning_final_")

            # Save model state
            final_model_path = os.path.join(temp_dir, "final_model.pth")
            if hasattr(pl_module, "state_dict"):
                import torch

                torch.save(pl_module.state_dict(), final_model_path)
                artifact.add_file(final_model_path, "model/final_model.pth")

            # Add complete model
            artifact.add_model(
                model=pl_module,
                framework="pytorch_lightning",
                description="Complete Lightning module with architecture",
            )

            # Add hyperparameters as config
            if hasattr(pl_module, "hparams") and pl_module.hparams:
                hparams_path = os.path.join(temp_dir, "hyperparameters.yaml")
                with open(hparams_path, "w") as f:
                    yaml.dump(dict(pl_module.hparams), f)
                artifact.add_file(hparams_path, "config/hyperparameters.yaml")

            # Add training summary
            self._add_training_summary(artifact, trainer, temp_dir)

            self.artifact_manager.log_artifact(artifact)
            logger.info(f"Logged final model artifact: {artifact.name}")

        except Exception as e:
            logger.exception(f"Failed to export final model: {e}")

    def _add_training_summary(self, artifact: TraceletArtifact, trainer, temp_dir: str):
        """Add training summary to artifact."""
        try:
            summary = {
                "total_epochs": trainer.current_epoch,
                "total_steps": trainer.global_step,
                "final_train_loss": trainer.callback_metrics.get("train_loss"),
                "final_val_loss": trainer.callback_metrics.get("val_loss"),
                "final_val_acc": trainer.callback_metrics.get("val_acc"),
                "device": str(trainer.root_device) if hasattr(trainer, "root_device") else None,
                "precision": getattr(trainer, "precision", None),
            }

            # Add timing information if available
            if hasattr(trainer, "fit_time"):
                summary["training_time_seconds"] = trainer.fit_time

            summary_path = os.path.join(temp_dir, "training_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            artifact.add_file(summary_path, "metadata/training_summary.json")

        except Exception as e:
            logger.warning(f"Failed to add training summary: {e}")

    def _hook_validation_samples(self, pl):
        """Auto-log validation samples."""
        original_validation_step = pl.LightningModule.validation_step

        def patched_validation_step(pl_module_self, batch, batch_idx, *args, **kwargs):
            result = original_validation_step(pl_module_self, batch, batch_idx, *args, **kwargs)

            try:
                # Log samples periodically
                if batch_idx % self.sample_frequency == 0:
                    self._log_validation_samples(pl_module_self, batch, batch_idx)
            except Exception as e:
                logger.debug(f"Failed to log validation samples: {e}")

            return result

        pl.LightningModule.validation_step = patched_validation_step

    def _log_validation_samples(self, model, batch, batch_idx):
        """Log validation samples with predictions."""
        try:
            x, y = self._unpack_batch(batch)
            if x is None:
                return

            # Generate predictions
            model.eval()
            import torch

            with torch.no_grad():
                preds = model(x)

            # Detect data modality and log appropriately
            if self._is_image_data(x):
                self._log_image_validation_samples(x, y, preds, batch_idx, model.current_epoch)
            elif self._is_audio_data(x):
                self._log_audio_validation_samples(x, y, preds, batch_idx, model.current_epoch)

        except Exception as e:
            logger.debug(f"Failed to log validation samples: {e}")

    def _log_image_validation_samples(self, images, targets, predictions, batch_idx, epoch):
        """Log image samples with predictions and targets."""
        try:
            artifact = TraceletArtifact(
                name=f"val_samples_epoch_{epoch}_batch_{batch_idx}",
                type=ArtifactType.SAMPLE,
                description=f"Validation samples from epoch {epoch}, batch {batch_idx}",
            )

            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="tracelet_val_samples_")

            # Take first few samples
            sample_size = min(8, images.size(0))

            for i in range(sample_size):
                img_tensor = images[i].cpu()

                # Create visualization
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))

                # Convert tensor to displayable format
                img_np = self._tensor_to_image_array(img_tensor)

                if img_np.ndim == 2:
                    ax.imshow(img_np, cmap="gray")
                elif img_np.ndim == 3:
                    ax.imshow(img_np)
                else:
                    # Fallback for other formats
                    ax.imshow(img_np.flatten().reshape(28, -1), cmap="gray")

                # Add prediction vs target info
                if targets is not None and predictions is not None:
                    pred_class = self._get_prediction_class(predictions[i])
                    true_class = self._get_target_class(targets[i])
                    ax.set_title(f"Pred: {pred_class}, True: {true_class}")

                ax.axis("off")

                # Save visualization
                img_path = os.path.join(temp_dir, f"sample_{i}.png")
                plt.savefig(img_path, bbox_inches="tight", dpi=150)
                plt.close()

                artifact.add_file(img_path, f"samples/sample_{i}.png")

            # Add batch metadata
            artifact.metadata.update({
                "epoch": epoch,
                "batch_idx": batch_idx,
                "batch_size": images.size(0),
                "sample_count": sample_size,
                "image_shape": list(images.shape[1:]),
                "data_type": "image",
            })

            self.artifact_manager.log_artifact(artifact)

        except Exception as e:
            logger.warning(f"Failed to log image validation samples: {e}")

    def _log_audio_validation_samples(self, audio, targets, predictions, batch_idx, epoch):
        """Log audio samples with predictions."""
        try:
            artifact = TraceletArtifact(
                name=f"val_audio_epoch_{epoch}_batch_{batch_idx}",
                type=ArtifactType.SAMPLE,
                description=f"Audio validation samples from epoch {epoch}, batch {batch_idx}",
            )

            temp_dir = tempfile.mkdtemp(prefix="tracelet_audio_samples_")
            sample_size = min(4, audio.size(0))

            for i in range(sample_size):
                audio_tensor = audio[i].cpu()

                # Save audio file
                audio_path = os.path.join(temp_dir, f"sample_{i}.wav")

                # Convert to numpy and save
                audio_np = audio_tensor.numpy()
                if audio_np.ndim > 1:
                    audio_np = audio_np.mean(axis=0)  # Convert to mono

                # Try to save as WAV file
                try:
                    import soundfile as sf

                    sf.write(audio_path, audio_np, 22050)
                    artifact.add_file(audio_path, f"audio/sample_{i}.wav")
                except ImportError:
                    logger.warning("soundfile not available, saving audio as numpy array")
                    np.save(audio_path.replace(".wav", ".npy"), audio_np)
                    artifact.add_file(audio_path.replace(".wav", ".npy"), f"audio/sample_{i}.npy")

            # Add metadata
            artifact.metadata.update({
                "epoch": epoch,
                "batch_idx": batch_idx,
                "sample_rate": 22050,
                "sample_count": sample_size,
                "audio_length": audio.shape[-1],
                "data_type": "audio",
            })

            self.artifact_manager.log_artifact(artifact)

        except Exception as e:
            logger.warning(f"Failed to log audio validation samples: {e}")

    def _tensor_to_image_array(self, tensor):
        """Convert tensor to displayable image array."""
        if tensor.dim() == 3 and tensor.size(0) in [1, 3]:
            # CHW format
            img_np = tensor.permute(1, 2, 0).numpy()
            img_np = img_np.squeeze(-1) if tensor.size(0) == 1 else np.clip(img_np, 0, 1)
            return img_np
        elif tensor.dim() == 2:
            # HW format
            return tensor.numpy()
        else:
            # Flatten for other formats
            size = int(np.sqrt(tensor.numel()))
            return tensor.numpy().flatten()[: size * size].reshape(size, size)

    def _get_prediction_class(self, prediction):
        """Extract class prediction from model output."""
        try:
            if hasattr(prediction, "argmax"):
                return prediction.argmax().item()
            elif hasattr(prediction, "item"):
                return prediction.item()
            else:
                return str(prediction)
        except Exception:
            return "unknown"

    def _get_target_class(self, target):
        """Extract target class."""
        try:
            if hasattr(target, "item"):
                return target.item()
            else:
                return str(target)
        except Exception:
            return "unknown"

    def _is_image_data(self, x) -> bool:
        """Detect if tensor represents image data."""
        try:
            import torch

            if not isinstance(x, torch.Tensor):
                return False

            # Common image tensor shapes: (B, C, H, W) or (B, H, W, C)
            if (
                x.dim() == 4
                or (x.dim() == 3 and x.size(0) in [1, 3])
                or (x.dim() == 2 and x.numel() in [28 * 28, 32 * 32, 64 * 64, 224 * 224])
            ):
                return True

        except Exception as e:
            logger.debug(f"Failed to detect image data: {e}")
        return False

    def _is_audio_data(self, x) -> bool:
        """Detect if tensor represents audio data."""
        try:
            import torch

            if not isinstance(x, torch.Tensor):
                return False

            # Audio often has large sequence length dimension
            if (x.dim() == 2 and x.size(-1) > 1000) or (
                x.dim() == 3 and x.size(-1) > 1000
            ):  # (B, T) with long time dimension
                return True

        except Exception as e:
            logger.debug(f"Failed to detect audio data: {e}")
        return False

    def _unpack_batch(self, batch):
        """Safely unpack batch data."""
        try:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                return batch[0], batch[1]
            elif isinstance(batch, dict):
                # Common keys for inputs and targets
                x = batch.get("input", batch.get("x", batch.get("data")))
                y = batch.get("target", batch.get("y", batch.get("label")))
                return x, y
            else:
                return batch, None
        except Exception as e:
            logger.debug(f"Failed to unpack batch data for artifact logging: {e}")
            logger.debug(
                f"Batch type: {type(batch)}, batch structure: {batch if hasattr(batch, '__dict__') else 'complex structure'}"
            )
            return None, None


class PyTorchArtifactHook(FrameworkHook):
    """Automatic artifact detection for vanilla PyTorch."""

    def __init__(self, artifact_manager: ArtifactManager):
        super().__init__()
        self.artifact_manager = artifact_manager
        self.monitored_saves: set[str] = set()

    def apply_hook(self):
        """Apply PyTorch artifact hooks."""
        try:
            import torch

            self._hook_torch_save(torch)
            logger.info("PyTorch artifact hooks applied successfully")
        except ImportError:
            logger.warning("PyTorch not available, skipping artifact hooks")
        except Exception as e:
            logger.exception(f"Failed to apply PyTorch artifact hooks: {e}")

    def _hook_torch_save(self, torch):
        """Hook torch.save() calls."""
        original_save = torch.save

        def patched_save(obj, f, *args, **kwargs):
            result = original_save(obj, f, *args, **kwargs)

            try:
                # Determine if this is a model save
                file_path = str(f) if isinstance(f, (str, Path)) else None
                if file_path and file_path not in self.monitored_saves:
                    self._handle_torch_save(obj, file_path)
                    self.monitored_saves.add(file_path)
            except Exception as e:
                logger.debug(f"Failed to handle torch.save: {e}")

            return result

        torch.save = patched_save

    def _handle_torch_save(self, obj, file_path: str):
        """Handle torch.save() artifact logging."""
        try:
            path = Path(file_path)

            # Determine artifact type
            artifact_type = self._determine_artifact_type(obj, path)
            if not artifact_type:
                return

            artifact = TraceletArtifact(
                name=path.stem, type=artifact_type, description=f"PyTorch {artifact_type.value} saved to {path.name}"
            )

            artifact.add_file(file_path)

            # Add object metadata
            if isinstance(obj, dict):
                self._add_dict_metadata(artifact, obj)
            elif hasattr(obj, "state_dict"):
                artifact.add_model(obj, framework="pytorch")

            self.artifact_manager.log_artifact(artifact)
            logger.info(f"Logged PyTorch artifact: {artifact.name}")

        except Exception as e:
            logger.warning(f"Failed to handle torch.save artifact: {e}")

    def _add_dict_metadata(self, artifact: TraceletArtifact, obj: dict):
        """Add metadata from dictionary object."""
        if "state_dict" in obj or "model_state_dict" in obj:
            artifact.metadata["contains_state_dict"] = True
        if "optimizer_state_dict" in obj:
            artifact.metadata["contains_optimizer"] = True
        if "epoch" in obj:
            artifact.metadata["epoch"] = obj["epoch"]
        if "loss" in obj:
            artifact.metadata["loss"] = obj["loss"]

    def _determine_artifact_type(self, obj, path: Path) -> Optional[ArtifactType]:
        """Determine artifact type from object and filename."""
        filename = path.name.lower()

        # Check filename patterns
        if "checkpoint" in filename or "ckpt" in filename:
            return ArtifactType.CHECKPOINT
        elif "model" in filename:
            return ArtifactType.MODEL
        elif "weights" in filename:
            return ArtifactType.WEIGHTS

        # Check object type
        if hasattr(obj, "state_dict"):
            # Model object
            return ArtifactType.MODEL
        elif isinstance(obj, dict) and ("state_dict" in obj or "model_state_dict" in obj):
            return ArtifactType.CHECKPOINT if "epoch" in obj else ArtifactType.MODEL

        return None
