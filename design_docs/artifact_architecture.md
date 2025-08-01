# Tracelet Unified Artifact Architecture

## Executive Summary

This document outlines the design for a unified artifact management system that leverages each MLOps platform's strengths while providing a seamless developer experience. The system uses **type-based routing**, **size-based optimization**, and **automagic detection** to handle models, checkpoints, images, audio, and other artifacts intelligently.

## Core Architecture

### 1. Artifact Type System

```python
from enum import Enum
from typing import Union, Optional, Any, List, Dict
from pathlib import Path
from abc import ABC, abstractmethod

class ArtifactType(Enum):
    """Semantic artifact types for intelligent routing"""
    # ML Assets
    MODEL = "model"              # Trained models (pytorch, sklearn, etc.)
    CHECKPOINT = "checkpoint"    # Training checkpoints (.ckpt, .pth)
    WEIGHTS = "weights"          # Model weights only

    # Data Assets
    DATASET = "dataset"          # Training/validation datasets
    SAMPLE = "sample"            # Evaluation samples/predictions

    # Media Assets
    IMAGE = "image"              # Single images or batches
    AUDIO = "audio"              # Audio files or arrays
    VIDEO = "video"              # Video files

    # Analysis Assets
    VISUALIZATION = "viz"        # Plots, charts, attention maps
    REPORT = "report"            # HTML reports, notebooks

    # Configuration Assets
    CONFIG = "config"            # Configuration files (.yaml, .json)
    CODE = "code"                # Source code snapshots

    # General Assets
    CUSTOM = "custom"            # User-defined artifacts
```

### 2. Unified Artifact Class

```python
class TraceletArtifact:
    """Unified artifact representation across all backends"""

    def __init__(
        self,
        name: str,
        type: ArtifactType,
        description: str = None,
        metadata: Dict[str, Any] = None
    ):
        self.name = name
        self.type = type
        self.description = description
        self.metadata = metadata or {}

        # Internal tracking
        self.version = None
        self.size_bytes = 0
        self.created_at = None

        # Content containers
        self.files: List[ArtifactFile] = []
        self.references: List[ArtifactReference] = []
        self.objects: List[ArtifactObject] = []

    def add_file(
        self,
        local_path: Union[str, Path],
        artifact_path: str = None,
        description: str = None
    ) -> 'TraceletArtifact':
        """Add local file to artifact"""
        file_path = Path(local_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")

        self.files.append(ArtifactFile(
            local_path=str(file_path),
            artifact_path=artifact_path or file_path.name,
            description=description,
            size_bytes=file_path.stat().st_size
        ))
        self.size_bytes += file_path.stat().st_size
        return self

    def add_reference(
        self,
        uri: str,
        size_bytes: int = None,
        description: str = None
    ) -> 'TraceletArtifact':
        """Add external storage reference (S3, GCS, etc.)"""
        self.references.append(ArtifactReference(
            uri=uri,
            size_bytes=size_bytes,
            description=description
        ))
        if size_bytes:
            self.size_bytes += size_bytes
        return self

    def add_object(
        self,
        obj: Any,
        name: str,
        serializer: str = "pickle",
        description: str = None
    ) -> 'TraceletArtifact':
        """Add Python object with automatic serialization"""
        self.objects.append(ArtifactObject(
            obj=obj,
            name=name,
            serializer=serializer,
            description=description
        ))
        return self

    def add_model(
        self,
        model: Any,
        framework: str = None,
        input_example: Any = None,
        signature: Any = None,
        description: str = None
    ) -> 'TraceletArtifact':
        """Add ML model with framework-specific handling"""
        if self.type not in [ArtifactType.MODEL, ArtifactType.CHECKPOINT]:
            raise ValueError("add_model() only valid for MODEL or CHECKPOINT artifacts")

        model_info = ModelInfo(
            model=model,
            framework=framework or self._detect_framework(model),
            input_example=input_example,
            signature=signature,
            description=description
        )

        self.metadata["model_info"] = model_info
        return self

    def _detect_framework(self, model: Any) -> str:
        """Auto-detect ML framework"""
        if hasattr(model, '__module__'):
            module = model.__module__
            if 'torch' in module:
                return 'pytorch'
            elif 'sklearn' in module:
                return 'sklearn'
            elif 'tensorflow' in module or 'keras' in module:
                return 'tensorflow'
        return 'unknown'
```

### 3. Platform-Specific Handlers

```python
class ArtifactHandler(ABC):
    """Abstract base for platform-specific artifact handling"""

    @abstractmethod
    def log_artifact(self, artifact: TraceletArtifact) -> ArtifactResult:
        """Log artifact and return platform-specific result"""

    @abstractmethod
    def get_artifact(self, name: str, version: str = "latest") -> TraceletArtifact:
        """Retrieve artifact by name/version"""

    @abstractmethod
    def list_artifacts(self, type_filter: ArtifactType = None) -> List[TraceletArtifact]:
        """List available artifacts"""

    def should_handle_type(self, artifact_type: ArtifactType) -> bool:
        """Return True if this handler is optimal for the artifact type"""
        return True  # Default: handle all types

class MLflowArtifactHandler(ArtifactHandler):
    """MLflow-optimized artifact handling"""

    # Optimal for: Models, checkpoints, general files
    OPTIMAL_TYPES = {
        ArtifactType.MODEL,
        ArtifactType.CHECKPOINT,
        ArtifactType.WEIGHTS,
        ArtifactType.CONFIG,
        ArtifactType.CODE
    }

    def should_handle_type(self, artifact_type: ArtifactType) -> bool:
        return artifact_type in self.OPTIMAL_TYPES

    def log_artifact(self, artifact: TraceletArtifact) -> ArtifactResult:
        if artifact.type == ArtifactType.MODEL and "model_info" in artifact.metadata:
            return self._log_model_artifact(artifact)
        elif artifact.size_bytes > LARGE_FILE_THRESHOLD:
            return self._log_large_artifact(artifact)
        else:
            return self._log_standard_artifact(artifact)

    def _log_model_artifact(self, artifact: TraceletArtifact) -> ArtifactResult:
        """Use MLflow's model logging with flavor detection"""
        model_info = artifact.metadata["model_info"]

        if model_info.framework == "pytorch":
            mlflow.pytorch.log_model(
                pytorch_model=model_info.model,
                artifact_path=artifact.name,
                input_example=model_info.input_example,
                signature=model_info.signature
            )
        elif model_info.framework == "sklearn":
            mlflow.sklearn.log_model(
                sk_model=model_info.model,
                artifact_path=artifact.name,
                input_example=model_info.input_example,
                signature=model_info.signature
            )
        # ... other frameworks

        return ArtifactResult(
            version=self._get_run_id(),
            uri=f"runs:/{self._run_id}/{artifact.name}",
            size_bytes=artifact.size_bytes
        )

class WANDBArtifactHandler(ArtifactHandler):
    """W&B-optimized artifact handling"""

    # Optimal for: Rich media, datasets, visualizations
    OPTIMAL_TYPES = {
        ArtifactType.IMAGE,
        ArtifactType.AUDIO,
        ArtifactType.VIDEO,
        ArtifactType.DATASET,
        ArtifactType.VISUALIZATION,
        ArtifactType.SAMPLE
    }

    def log_artifact(self, artifact: TraceletArtifact) -> ArtifactResult:
        if artifact.type in {ArtifactType.IMAGE, ArtifactType.AUDIO, ArtifactType.VIDEO}:
            return self._log_media_artifact(artifact)
        elif artifact.size_bytes > LARGE_FILE_THRESHOLD:
            return self._log_reference_artifact(artifact)
        else:
            return self._log_wandb_artifact(artifact)

    def _log_media_artifact(self, artifact: TraceletArtifact) -> ArtifactResult:
        """Use W&B's rich media APIs"""
        media_objects = {}

        for file in artifact.files:
            if artifact.type == ArtifactType.IMAGE:
                media_objects[file.artifact_path] = wandb.Image(
                    file.local_path,
                    caption=file.description
                )
            elif artifact.type == ArtifactType.AUDIO:
                media_objects[file.artifact_path] = wandb.Audio(
                    file.local_path,
                    caption=file.description
                )
            elif artifact.type == ArtifactType.VIDEO:
                media_objects[file.artifact_path] = wandb.Video(
                    file.local_path,
                    caption=file.description
                )

        # Log as rich media, not artifact
        wandb.log({artifact.name: media_objects})

        return ArtifactResult(
            version="logged_as_media",
            uri=f"wandb://{self._run.id}/{artifact.name}",
            size_bytes=artifact.size_bytes
        )

class ClearMLArtifactHandler(ArtifactHandler):
    """ClearML-optimized artifact handling"""

    # Optimal for: Auto-detection, dynamic updates
    OPTIMAL_TYPES = {
        ArtifactType.MODEL,
        ArtifactType.CHECKPOINT,
        ArtifactType.REPORT
    }

    def log_artifact(self, artifact: TraceletArtifact) -> ArtifactResult:
        if artifact.type == ArtifactType.MODEL and "model_info" in artifact.metadata:
            return self._log_model_with_auto_detection(artifact)
        elif artifact.type == ArtifactType.REPORT:
            return self._log_report_artifact(artifact)
        else:
            return self._log_clearml_artifact(artifact)

    def _log_model_with_auto_detection(self, artifact: TraceletArtifact) -> ArtifactResult:
        """Leverage ClearML's automatic model detection"""
        model_info = artifact.metadata["model_info"]

        # ClearML will auto-detect and log the model
        # We just need to trigger a save operation
        if model_info.framework == "pytorch":
            temp_path = f"/tmp/{artifact.name}.pth"
            torch.save(model_info.model.state_dict(), temp_path)

            # Upload as artifact
            self._task.upload_artifact(
                name=artifact.name,
                artifact_object=temp_path
            )

        return ArtifactResult(
            version=str(int(time.time())),
            uri=f"clearml://{self._task.id}/{artifact.name}",
            size_bytes=artifact.size_bytes
        )
```

### 4. Intelligent Artifact Manager

```python
class ArtifactManager:
    """Central artifact management with intelligent routing"""

    def __init__(self, handlers: List[ArtifactHandler]):
        self.handlers = handlers
        self.routing_cache = {}

    def log_artifact(self, artifact: TraceletArtifact) -> Dict[str, ArtifactResult]:
        """Log artifact using optimal handler(s)"""
        results = {}

        # Find best handler for this artifact type
        optimal_handlers = self._get_optimal_handlers(artifact)

        for handler in optimal_handlers:
            try:
                result = handler.log_artifact(artifact)
                results[handler.__class__.__name__] = result
            except Exception as e:
                logger.error(f"Failed to log artifact with {handler.__class__.__name__}: {e}")

        return results

    def _get_optimal_handlers(self, artifact: TraceletArtifact) -> List[ArtifactHandler]:
        """Select optimal handlers based on artifact type and size"""

        # Cache key for routing decisions
        cache_key = (artifact.type, artifact.size_bytes > LARGE_FILE_THRESHOLD)

        if cache_key in self.routing_cache:
            return self.routing_cache[cache_key]

        # Find handlers that claim to be optimal for this type
        optimal = [h for h in self.handlers if h.should_handle_type(artifact.type)]

        # If no optimal handlers, use all handlers
        if not optimal:
            optimal = self.handlers

        # For large files, prefer handlers with good large file support
        if artifact.size_bytes > LARGE_FILE_THRESHOLD:
            # Prefer MLflow (multipart), W&B (references), ClearML (direct access)
            optimal = sorted(optimal, key=lambda h: self._get_large_file_score(h))

        self.routing_cache[cache_key] = optimal
        return optimal

    def _get_large_file_score(self, handler: ArtifactHandler) -> int:
        """Score handlers for large file capability (higher = better)"""
        scores = {
            'MLflowArtifactHandler': 3,  # Multipart upload
            'WANDBArtifactHandler': 2,   # Reference artifacts
            'ClearMLArtifactHandler': 2, # Direct access
            'AiMArtifactHandler': 1      # File references only
        }
        return scores.get(handler.__class__.__name__, 1)
```

## 5. Automagic Integration

### Framework Hooks

```python
class LightningArtifactHook(FrameworkHook):
    """Automatic artifact detection for PyTorch Lightning"""

    def __init__(self, artifact_manager: ArtifactManager):
        super().__init__()
        self.artifact_manager = artifact_manager
        self.logged_checkpoints = set()

    def apply_hook(self):
        """Hook into Lightning callbacks"""
        self._patch_checkpoint_callback()
        self._patch_validation_step()

    def _patch_checkpoint_callback(self):
        """Auto-log checkpoints as they're saved"""
        original_save = pl.callbacks.ModelCheckpoint._save_checkpoint

        def patched_save(callback_self, trainer, pl_module, filepath):
            result = original_save(callback_self, trainer, pl_module, filepath)

            # Auto-log checkpoint as artifact
            if filepath not in self.logged_checkpoints:
                artifact = TraceletArtifact(
                    name=f"checkpoint_epoch_{trainer.current_epoch}",
                    type=ArtifactType.CHECKPOINT,
                    description=f"Checkpoint at epoch {trainer.current_epoch}"
                ).add_file(filepath).add_model(
                    model=pl_module,
                    framework="pytorch_lightning",
                    description=f"Model state at epoch {trainer.current_epoch}"
                )

                self.artifact_manager.log_artifact(artifact)
                self.logged_checkpoints.add(filepath)

            return result

        pl.callbacks.ModelCheckpoint._save_checkpoint = patched_save

    def _patch_validation_step(self):
        """Auto-log evaluation samples during validation"""
        original_validation_step = pl.LightningModule.validation_step

        def patched_validation_step(pl_module_self, batch, batch_idx, *args, **kwargs):
            result = original_validation_step(pl_module_self, batch, batch_idx, *args, **kwargs)

            # Sample logging every N batches
            if batch_idx % 100 == 0:  # Log samples every 100 batches
                self._log_evaluation_samples(pl_module_self, batch, batch_idx)

            return result

        pl.LightningModule.validation_step = patched_validation_step

    def _log_evaluation_samples(self, model, batch, batch_idx):
        """Log evaluation samples based on data type"""
        x, y = batch if isinstance(batch, (list, tuple)) else (batch, None)

        # Generate predictions
        with torch.no_grad():
            preds = model(x)

        # Auto-detect data type and log appropriately
        if self._is_image_data(x):
            self._log_image_samples(x, y, preds, batch_idx)
        elif self._is_audio_data(x):
            self._log_audio_samples(x, y, preds, batch_idx)

    def _log_image_samples(self, images, targets, predictions, batch_idx):
        """Log image samples with predictions"""
        # Take first few samples from batch
        sample_size = min(4, images.size(0))

        artifact = TraceletArtifact(
            name=f"eval_samples_batch_{batch_idx}",
            type=ArtifactType.SAMPLE,
            description=f"Evaluation samples from batch {batch_idx}"
        )

        for i in range(sample_size):
            img_tensor = images[i]
            # Convert to PIL/numpy for logging
            img_path = self._save_tensor_as_image(img_tensor, f"sample_{batch_idx}_{i}.png")
            artifact.add_file(img_path, f"images/sample_{i}.png")

        self.artifact_manager.log_artifact(artifact)
```

### File System Watchers

```python
class FileSystemArtifactDetector:
    """Automatically detect artifacts created in watched directories"""

    def __init__(self, artifact_manager: ArtifactManager, watch_dirs: List[str]):
        self.artifact_manager = artifact_manager
        self.watch_dirs = watch_dirs
        self.observer = Observer()

    def start_watching(self):
        """Start file system monitoring"""
        for watch_dir in self.watch_dirs:
            event_handler = ArtifactFileHandler(self.artifact_manager)
            self.observer.schedule(event_handler, watch_dir, recursive=True)

        self.observer.start()

    def stop_watching(self):
        """Stop file system monitoring"""
        self.observer.stop()
        self.observer.join()

class ArtifactFileHandler(FileSystemEventHandler):
    """Handle file system events for artifact detection"""

    def __init__(self, artifact_manager: ArtifactManager):
        self.artifact_manager = artifact_manager

    def on_created(self, event):
        """Handle file creation events"""
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        artifact_type = self._detect_artifact_type(file_path)

        if artifact_type:
            # Auto-log detected artifact
            artifact = TraceletArtifact(
                name=file_path.stem,
                type=artifact_type,
                description=f"Auto-detected {artifact_type.value}"
            ).add_file(file_path)

            self.artifact_manager.log_artifact(artifact)

    def _detect_artifact_type(self, file_path: Path) -> Optional[ArtifactType]:
        """Detect artifact type from file extension/name"""
        suffix = file_path.suffix.lower()
        name = file_path.name.lower()

        if suffix in ['.pth', '.pt', '.ckpt']:
            return ArtifactType.CHECKPOINT if 'checkpoint' in name else ArtifactType.MODEL
        elif suffix in ['.pkl', '.pickle']:
            return ArtifactType.MODEL
        elif suffix in ['.png', '.jpg', '.jpeg']:
            return ArtifactType.IMAGE
        elif suffix in ['.wav', '.mp3', '.flac']:
            return ArtifactType.AUDIO
        elif suffix in ['.mp4', '.avi', '.mov']:
            return ArtifactType.VIDEO
        elif suffix in ['.yaml', '.yml', '.json']:
            return ArtifactType.CONFIG
        elif suffix in ['.html', '.pdf']:
            return ArtifactType.REPORT

        return None
```

## 6. Integration with Existing Experiment System

```python
# Extend the existing Experiment class
class Experiment:
    def __init__(self, ...):
        # Existing initialization...

        # Add artifact management
        self.artifact_manager = ArtifactManager(self._create_artifact_handlers())
        self.filesystem_detector = None

        if self.config.automagic:
            # Enable automagic artifact detection
            self._enable_automagic_artifacts()

    def _create_artifact_handlers(self) -> List[ArtifactHandler]:
        """Create platform-specific artifact handlers"""
        handlers = []

        for backend_name in self.backends:
            if backend_name == "mlflow":
                handlers.append(MLflowArtifactHandler(self.backends[backend_name]))
            elif backend_name == "wandb":
                handlers.append(WANDBArtifactHandler(self.backends[backend_name]))
            elif backend_name == "clearml":
                handlers.append(ClearMLArtifactHandler(self.backends[backend_name]))
            elif backend_name == "aim":
                handlers.append(AiMArtifactHandler(self.backends[backend_name]))

        return handlers

    def _enable_automagic_artifacts(self):
        """Enable automatic artifact detection"""
        # Framework hooks
        if self._is_framework_available("pytorch_lightning"):
            hook = LightningArtifactHook(self.artifact_manager)
            hook.apply_hook()

        # File system watching  (optional, can be resource intensive)
        if self.config.watch_filesystem:
            watch_dirs = self.config.artifact_watch_dirs or ["./checkpoints", "./outputs", "./artifacts"]
            self.filesystem_detector = FileSystemArtifactDetector(
                self.artifact_manager,
                watch_dirs
            )
            self.filesystem_detector.start_watching()

    def log_artifact(
        self,
        name: str,
        type: ArtifactType,
        description: str = None
    ) -> TraceletArtifact:
        """Create artifact builder for manual logging"""
        return TraceletArtifact(name, type, description)

    def save_artifact(self, artifact: TraceletArtifact) -> Dict[str, ArtifactResult]:
        """Save artifact to all backends"""
        return self.artifact_manager.log_artifact(artifact)

    def stop(self):
        """Stop experiment and cleanup"""
        # Existing stop logic...

        # Stop file system watching if enabled
        if self.filesystem_detector:
            self.filesystem_detector.stop_watching()
```

## Usage Examples

### Manual Artifact Logging

```python
# Initialize experiment with artifact support
exp = Experiment(
    name="artifact_demo",
    backend=["wandb", "mlflow", "clearml"],
    automagic=True
)
exp.start()

# Log model artifact
model_artifact = exp.log_artifact("my_model", ArtifactType.MODEL)\
    .add_model(model, framework="pytorch", input_example=sample_input)\
    .add_file("model_config.yaml", "config/model.yaml")

exp.save_artifact(model_artifact)

# Log evaluation samples
eval_artifact = exp.log_artifact("eval_samples", ArtifactType.SAMPLE)\
    .add_file("predictions.png", "images/predictions.png")\
    .add_file("attention_map.png", "visualizations/attention.png")

exp.save_artifact(eval_artifact)

# Log dataset with external reference
dataset_artifact = exp.log_artifact("training_data", ArtifactType.DATASET)\
    .add_reference("s3://my-bucket/datasets/train.tar.gz", size_bytes=1024*1024*100)\
    .add_file("dataset_stats.json", "metadata/stats.json")

exp.save_artifact(dataset_artifact)
```

### Automagic Lightning Integration

```python
# Just add automagic=True - artifacts logged automatically!
exp = Experiment(
    name="lightning_automagic",
    backend=["wandb", "clearml"],
    automagic=True  # Enables automatic artifact detection
)
exp.start()

# Your normal Lightning code - checkpoints and samples auto-logged
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train_loader, val_loader)

exp.stop()  # Artifacts automatically captured throughout training
```

This architecture provides:

1. **Intelligent Routing**: Artifacts routed to optimal backends based on type
2. **Size Optimization**: Large files handled efficiently per platform
3. **Framework Integration**: Seamless Lightning/PyTorch checkpoint logging
4. **Automagic Detection**: Automatic artifact discovery and logging
5. **Unified Interface**: Simple API hiding platform complexity
6. **Performance**: Leverages each platform's strengths

The system is designed to be extensible, allowing new artifact types and platform handlers to be added easily.
