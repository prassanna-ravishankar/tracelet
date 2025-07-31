# Framework-Specific Artifact Integration

## Overview

This document details how Tracelet's artifact system integrates with specific ML frameworks to provide seamless, automatic artifact logging. Each framework has unique patterns for saving models, checkpoints, and generating evaluation outputs.

## 1. PyTorch Lightning Integration

### Hook Points

```python
class LightningArtifactHook(FrameworkHook):
    """Comprehensive Lightning artifact integration"""

    def __init__(self, artifact_manager: ArtifactManager):
        super().__init__()
        self.artifact_manager = artifact_manager
        self.logged_checkpoints = set()
        self.sample_frequency = 100  # Log samples every N batches

    def apply_hook(self):
        """Apply all Lightning hooks"""
        self._hook_checkpoint_saving()
        self._hook_model_export()
        self._hook_validation_samples()
        self._hook_test_samples()
        self._hook_training_artifacts()

    def _hook_checkpoint_saving(self):
        """Auto-log Lightning checkpoints"""
        # Hook ModelCheckpoint callback
        original_save = pl.callbacks.ModelCheckpoint._save_checkpoint

        def patched_save(callback_self, trainer, pl_module, filepath):
            result = original_save(callback_self, trainer, pl_module, filepath)

            # Determine checkpoint type
            is_best = callback_self.best_model_path == filepath
            is_last = callback_self.last_model_path == filepath

            checkpoint_type = "best" if is_best else "last" if is_last else "periodic"

            if filepath not in self.logged_checkpoints:
                artifact = TraceletArtifact(
                    name=f"checkpoint_{checkpoint_type}_epoch_{trainer.current_epoch}",
                    type=ArtifactType.CHECKPOINT,
                    description=f"{checkpoint_type.title()} checkpoint at epoch {trainer.current_epoch}, step {trainer.global_step}"
                )

                # Add checkpoint file
                artifact.add_file(filepath, f"checkpoints/{Path(filepath).name}")

                # Add model metadata
                artifact.add_model(
                    model=pl_module,
                    framework="pytorch_lightning",
                    description=f"Lightning module at epoch {trainer.current_epoch}"
                )

                # Add training metadata
                artifact.metadata.update({
                    "epoch": trainer.current_epoch,
                    "global_step": trainer.global_step,
                    "checkpoint_type": checkpoint_type,
                    "val_loss": trainer.callback_metrics.get("val_loss"),
                    "train_loss": trainer.callback_metrics.get("train_loss"),
                    "learning_rate": trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else None
                })

                self.artifact_manager.log_artifact(artifact)
                self.logged_checkpoints.add(filepath)

            return result

        pl.callbacks.ModelCheckpoint._save_checkpoint = patched_save

    def _hook_model_export(self):
        """Auto-export final model at training end"""
        original_teardown = pl.Trainer.teardown

        def patched_teardown(trainer_self, *args, **kwargs):
            result = original_teardown(trainer_self, *args, **kwargs)

            # Export final model if training completed successfully
            if hasattr(trainer_self, 'lightning_module') and trainer_self.state.finished:
                self._export_final_model(trainer_self, trainer_self.lightning_module)

            return result

        pl.Trainer.teardown = patched_teardown

    def _export_final_model(self, trainer, pl_module):
        """Export final trained model"""
        # Create final model artifact
        artifact = TraceletArtifact(
            name=f"final_model_epoch_{trainer.current_epoch}",
            type=ArtifactType.MODEL,
            description=f"Final trained model after {trainer.current_epoch} epochs"
        )

        # Save model state
        final_model_path = f"./artifacts/final_model_epoch_{trainer.current_epoch}.pth"
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        torch.save(pl_module.state_dict(), final_model_path)

        artifact.add_file(final_model_path, "model/final_model.pth")

        # Add complete model with structure
        artifact.add_model(
            model=pl_module,
            framework="pytorch_lightning",
            description="Complete Lightning module with architecture"
        )

        # Add hyperparameters as config
        if hasattr(pl_module, 'hparams'):
            hparams_path = f"./artifacts/hparams_epoch_{trainer.current_epoch}.yaml"
            with open(hparams_path, 'w') as f:
                yaml.dump(dict(pl_module.hparams), f)
            artifact.add_file(hparams_path, "config/hyperparameters.yaml")

        # Add training summary
        summary = {
            "total_epochs": trainer.current_epoch,
            "total_steps": trainer.global_step,
            "final_train_loss": trainer.callback_metrics.get("train_loss"),
            "final_val_loss": trainer.callback_metrics.get("val_loss"),
            "final_val_acc": trainer.callback_metrics.get("val_acc"),
            "training_time": getattr(trainer, 'training_time', None),
            "device": str(trainer.root_device)
        }

        summary_path = f"./artifacts/training_summary_epoch_{trainer.current_epoch}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        artifact.add_file(summary_path, "metadata/training_summary.json")

        self.artifact_manager.log_artifact(artifact)

    def _hook_validation_samples(self):
        """Auto-log validation samples"""
        original_validation_step = pl.LightningModule.validation_step

        def patched_validation_step(pl_module_self, batch, batch_idx, *args, **kwargs):
            result = original_validation_step(pl_module_self, batch, batch_idx, *args, **kwargs)

            # Log samples periodically
            if batch_idx % self.sample_frequency == 0:
                self._log_validation_samples(pl_module_self, batch, batch_idx)

            return result

        pl.LightningModule.validation_step = patched_validation_step

    def _log_validation_samples(self, model, batch, batch_idx):
        """Log validation samples with predictions"""
        try:
            x, y = self._unpack_batch(batch)

            # Generate predictions
            model.eval()
            with torch.no_grad():
                preds = model(x)

            # Detect data modality and log appropriately
            if self._is_image_data(x):
                self._log_image_validation_samples(x, y, preds, batch_idx, model.current_epoch)
            elif self._is_audio_data(x):
                self._log_audio_validation_samples(x, y, preds, batch_idx, model.current_epoch)
            elif self._is_text_data(x):
                self._log_text_validation_samples(x, y, preds, batch_idx, model.current_epoch)

        except Exception as e:
            logger.warning(f"Failed to log validation samples: {e}")

    def _log_image_validation_samples(self, images, targets, predictions, batch_idx, epoch):
        """Log image samples with predictions and targets"""
        artifact = TraceletArtifact(
            name=f"val_samples_epoch_{epoch}_batch_{batch_idx}",
            type=ArtifactType.SAMPLE,
            description=f"Validation samples from epoch {epoch}, batch {batch_idx}"
        )

        # Take first few samples
        sample_size = min(8, images.size(0))

        for i in range(sample_size):
            img_tensor = images[i].cpu()

            # Create visualization with prediction vs target
            fig, axes = plt.subplots(1, 1, figsize=(8, 6))

            # Convert tensor to displayable format
            if img_tensor.dim() == 3 and img_tensor.size(0) in [1, 3]:
                # CHW format
                img_np = img_tensor.permute(1, 2, 0).numpy()
                if img_tensor.size(0) == 1:
                    img_np = img_np.squeeze(-1)
                    axes.imshow(img_np, cmap='gray')
                else:
                    img_np = np.clip(img_np, 0, 1)
                    axes.imshow(img_np)
            else:
                # Flatten for other formats
                img_np = img_tensor.numpy().flatten()
                axes.imshow(img_np.reshape(int(np.sqrt(len(img_np))), -1), cmap='gray')

            # Add prediction vs target info
            if targets is not None and predictions is not None:
                pred_class = predictions[i].argmax().item() if predictions[i].dim() > 0 else predictions[i].item()
                true_class = targets[i].item() if hasattr(targets[i], 'item') else targets[i]
                axes.set_title(f"Pred: {pred_class}, True: {true_class}")

            axes.axis('off')

            # Save visualization
            img_path = f"./artifacts/val_sample_e{epoch}_b{batch_idx}_i{i}.png"
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            plt.savefig(img_path, bbox_inches='tight', dpi=150)
            plt.close()

            artifact.add_file(img_path, f"samples/sample_{i}.png")

        # Add batch metadata
        artifact.metadata.update({
            "epoch": epoch,
            "batch_idx": batch_idx,
            "batch_size": images.size(0),
            "sample_count": sample_size,
            "image_shape": list(images.shape[1:]),
            "data_type": "image"
        })

        self.artifact_manager.log_artifact(artifact)

    def _log_audio_validation_samples(self, audio, targets, predictions, batch_idx, epoch):
        """Log audio samples with predictions"""
        artifact = TraceletArtifact(
            name=f"val_audio_epoch_{epoch}_batch_{batch_idx}",
            type=ArtifactType.SAMPLE,
            description=f"Audio validation samples from epoch {epoch}, batch {batch_idx}"
        )

        sample_size = min(4, audio.size(0))

        for i in range(sample_size):
            audio_tensor = audio[i].cpu()

            # Save audio file
            audio_path = f"./artifacts/val_audio_e{epoch}_b{batch_idx}_i{i}.wav"
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)

            # Convert to numpy and save (assuming sample rate of 22050)
            audio_np = audio_tensor.numpy()
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=0)  # Convert to mono

            import soundfile as sf
            sf.write(audio_path, audio_np, 22050)

            artifact.add_file(audio_path, f"audio/sample_{i}.wav")

        # Add metadata
        artifact.metadata.update({
            "epoch": epoch,
            "batch_idx": batch_idx,
            "sample_rate": 22050,
            "sample_count": sample_size,
            "audio_length": audio.shape[-1],
            "data_type": "audio"
        })

        self.artifact_manager.log_artifact(artifact)

    def _is_image_data(self, x: torch.Tensor) -> bool:
        """Detect if tensor represents image data"""
        if not isinstance(x, torch.Tensor):
            return False

        # Common image tensor shapes: (B, C, H, W) or (B, H, W, C)
        if x.dim() == 4:
            return True
        elif x.dim() == 3 and x.size(0) in [1, 3]:  # Single image (C, H, W)
            return True
        elif x.dim() == 2 and x.numel() in [28*28, 32*32, 64*64, 224*224]:  # Flattened images
            return True

        return False

    def _is_audio_data(self, x: torch.Tensor) -> bool:
        """Detect if tensor represents audio data"""
        if not isinstance(x, torch.Tensor):
            return False

        # Audio often has large sequence length dimension
        if x.dim() == 2 and x.size(-1) > 1000:  # (B, T) with long time dimension
            return True
        elif x.dim() == 3 and x.size(-1) > 1000:  # (B, C, T) multi-channel audio
            return True

        return False

    def _unpack_batch(self, batch):
        """Safely unpack batch data"""
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]
        elif isinstance(batch, dict):
            # Common keys for inputs and targets
            x = batch.get('input', batch.get('x', batch.get('data')))
            y = batch.get('target', batch.get('y', batch.get('label')))
            return x, y
        else:
            return batch, None
```

## 2. PyTorch (Vanilla) Integration

```python
class PyTorchArtifactHook(FrameworkHook):
    """Vanilla PyTorch artifact integration"""

    def __init__(self, artifact_manager: ArtifactManager):
        super().__init__()
        self.artifact_manager = artifact_manager
        self.monitored_saves = set()

    def apply_hook(self):
        """Apply PyTorch hooks"""
        self._hook_torch_save()
        self._hook_model_loading()

    def _hook_torch_save(self):
        """Hook torch.save() calls"""
        original_save = torch.save

        def patched_save(obj, f, *args, **kwargs):
            result = original_save(obj, f, *args, **kwargs)

            # Determine if this is a model save
            file_path = str(f) if isinstance(f, (str, Path)) else None
            if file_path and file_path not in self.monitored_saves:
                self._handle_torch_save(obj, file_path)
                self.monitored_saves.add(file_path)

            return result

        torch.save = patched_save

    def _handle_torch_save(self, obj, file_path: str):
        """Handle torch.save() artifact logging"""
        path = Path(file_path)

        # Determine artifact type based on object and filename
        artifact_type = self._determine_artifact_type(obj, path)
        if not artifact_type:
            return

        artifact = TraceletArtifact(
            name=path.stem,
            type=artifact_type,
            description=f"PyTorch {artifact_type.value} saved to {path.name}"
        )

        artifact.add_file(file_path)

        # Add object metadata
        if isinstance(obj, dict):
            # Common patterns: state_dict, checkpoint dict
            if 'state_dict' in obj or 'model_state_dict' in obj:
                artifact.metadata['contains_state_dict'] = True
            if 'optimizer_state_dict' in obj:
                artifact.metadata['contains_optimizer'] = True
            if 'epoch' in obj:
                artifact.metadata['epoch'] = obj['epoch']
            if 'loss' in obj:
                artifact.metadata['loss'] = obj['loss']

        elif hasattr(obj, 'state_dict'):
            # Direct model object
            artifact.add_model(obj, framework="pytorch")

        self.artifact_manager.log_artifact(artifact)

    def _determine_artifact_type(self, obj, path: Path) -> Optional[ArtifactType]:
        """Determine artifact type from object and filename"""
        filename = path.name.lower()

        # Check filename patterns
        if 'checkpoint' in filename or 'ckpt' in filename:
            return ArtifactType.CHECKPOINT
        elif 'model' in filename:
            return ArtifactType.MODEL
        elif 'weights' in filename:
            return ArtifactType.WEIGHTS

        # Check object type
        if hasattr(obj, 'state_dict'):
            # Model object
            return ArtifactType.MODEL
        elif isinstance(obj, dict):
            if 'state_dict' in obj or 'model_state_dict' in obj:
                return ArtifactType.CHECKPOINT if 'epoch' in obj else ArtifactType.MODEL

        return None
```

## 3. Hugging Face Transformers Integration

```python
class TransformersArtifactHook(FrameworkHook):
    """Hugging Face Transformers artifact integration"""

    def __init__(self, artifact_manager: ArtifactManager):
        super().__init__()
        self.artifact_manager = artifact_manager

    def apply_hook(self):
        """Apply Transformers hooks"""
        self._hook_model_saving()
        self._hook_tokenizer_saving()
        self._hook_trainer_callbacks()

    def _hook_model_saving(self):
        """Hook model.save_pretrained() calls"""
        try:
            from transformers import PreTrainedModel

            original_save = PreTrainedModel.save_pretrained

            def patched_save(model_self, save_directory, *args, **kwargs):
                result = original_save(model_self, save_directory, *args, **kwargs)

                # Log saved model as artifact
                save_path = Path(save_directory)

                artifact = TraceletArtifact(
                    name=f"hf_model_{model_self.config.model_type}",
                    type=ArtifactType.MODEL,
                    description=f"Hugging Face {model_self.config.model_type} model"
                )

                # Add all saved files
                for file_path in save_path.glob("*"):
                    if file_path.is_file():
                        artifact.add_file(str(file_path), f"model/{file_path.name}")

                # Add model metadata
                artifact.metadata.update({
                    "model_type": model_self.config.model_type,
                    "num_parameters": model_self.num_parameters(),
                    "config": model_self.config.to_dict(),
                    "framework": "transformers"
                })

                self.artifact_manager.log_artifact(artifact)
                return result

            PreTrainedModel.save_pretrained = patched_save

        except ImportError:
            logger.warning("Transformers not available, skipping model hook")

    def _hook_trainer_callbacks(self):
        """Hook Transformers Trainer for checkpoint logging"""
        try:
            from transformers import Trainer

            original_save_checkpoint = Trainer._save_checkpoint

            def patched_save_checkpoint(trainer_self, model, trial, metrics=None):
                result = original_save_checkpoint(trainer_self, model, trial, metrics)

                # Log checkpoint artifact
                if hasattr(trainer_self.state, 'best_model_checkpoint') and trainer_self.state.best_model_checkpoint:
                    checkpoint_path = Path(trainer_self.state.best_model_checkpoint)

                    artifact = TraceletArtifact(
                        name=f"hf_checkpoint_{trainer_self.state.global_step}",
                        type=ArtifactType.CHECKPOINT,
                        description=f"Transformers checkpoint at step {trainer_self.state.global_step}"
                    )

                    # Add checkpoint directory contents
                    for file_path in checkpoint_path.glob("*"):
                        if file_path.is_file():
                            artifact.add_file(str(file_path), f"checkpoint/{file_path.name}")

                    # Add training metadata
                    artifact.metadata.update({
                        "global_step": trainer_self.state.global_step,
                        "epoch": trainer_self.state.epoch,
                        "learning_rate": trainer_self.get_lr()[-1] if trainer_self.get_lr() else None,
                        "train_loss": trainer_self.state.log_history[-1].get("train_loss") if trainer_self.state.log_history else None,
                        "eval_loss": metrics.get("eval_loss") if metrics else None
                    })

                    self.artifact_manager.log_artifact(artifact)

                return result

            Trainer._save_checkpoint = patched_save_checkpoint

        except ImportError:
            logger.warning("Transformers Trainer not available, skipping trainer hook")
```

## 4. Scikit-learn Integration

```python
class SklearnArtifactHook(FrameworkHook):
    """Scikit-learn artifact integration"""

    def __init__(self, artifact_manager: ArtifactManager):
        super().__init__()
        self.artifact_manager = artifact_manager

    def apply_hook(self):
        """Apply sklearn hooks"""
        self._hook_joblib_dump()
        self._hook_pickle_dump()

    def _hook_joblib_dump(self):
        """Hook joblib.dump() for sklearn models"""
        try:
            import joblib
            original_dump = joblib.dump

            def patched_dump(value, filename, *args, **kwargs):
                result = original_dump(value, filename, *args, **kwargs)

                # Check if dumped object is sklearn model
                if self._is_sklearn_model(value):
                    self._log_sklearn_model(value, filename)

                return result

            joblib.dump = patched_dump

        except ImportError:
            logger.warning("joblib not available, skipping sklearn hook")

    def _is_sklearn_model(self, obj) -> bool:
        """Check if object is sklearn model"""
        try:
            from sklearn.base import BaseEstimator
            return isinstance(obj, BaseEstimator)
        except ImportError:
            return False

    def _log_sklearn_model(self, model, filename):
        """Log sklearn model artifact"""
        artifact = TraceletArtifact(
            name=f"sklearn_{model.__class__.__name__}",
            type=ArtifactType.MODEL,
            description=f"Scikit-learn {model.__class__.__name__} model"
        )

        artifact.add_file(str(filename))
        artifact.add_model(model, framework="sklearn")

        # Add model metadata
        artifact.metadata.update({
            "model_class": model.__class__.__name__,
            "parameters": model.get_params(),
            "framework": "sklearn"
        })

        self.artifact_manager.log_artifact(artifact)
```

## Integration Priority & Rollout Plan

### Phase 1: Core Integration (Immediate)

1. **PyTorch Lightning** - Most requested, clear hook points
2. **PyTorch vanilla** - torch.save() monitoring
3. **Basic file watching** - Generic checkpoint detection

### Phase 2: Extended Integration (Next)

1. **Hugging Face Transformers** - Large user base
2. **Scikit-learn** - Widespread adoption
3. **TensorFlow/Keras** - Major framework

### Phase 3: Advanced Features (Later)

1. **Custom framework adapters** - User-defined integrations
2. **Cross-framework lineage** - Track model conversions
3. **Distributed training** - Multi-node artifact coordination

This integration plan provides comprehensive artifact tracking across the ML ecosystem while maintaining the seamless "automagic" experience that makes Tracelet unique.
