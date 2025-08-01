"""
File system artifact detector for automatic artifact detection.

This module watches specified directories for new files and automatically
logs them as artifacts based on file type and naming patterns.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from ..core.artifact_manager import ArtifactManager
from ..core.artifacts import ArtifactType, TraceletArtifact

logger = logging.getLogger(__name__)

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    _HAS_WATCHDOG = True
except ImportError:
    _HAS_WATCHDOG = False
    FileSystemEventHandler = object
    Observer = None


class FileSystemArtifactDetector:
    """Automatically detect artifacts created in watched directories."""

    def __init__(self, artifact_manager: ArtifactManager, watch_dirs: list[str]):
        if not _HAS_WATCHDOG:
            raise ImportError("watchdog package required for filesystem artifact detection")

        self.artifact_manager = artifact_manager
        self.watch_dirs = watch_dirs
        self.observer = Observer()
        self._is_watching = False

    def start_watching(self):
        """Start file system monitoring."""
        if self._is_watching:
            logger.warning("File system detector already watching")
            return

        try:
            for watch_dir in self.watch_dirs:
                # Create directory if it doesn't exist
                Path(watch_dir).mkdir(parents=True, exist_ok=True)

                event_handler = ArtifactFileHandler(self.artifact_manager)
                self.observer.schedule(event_handler, watch_dir, recursive=True)
                logger.info(f"Watching directory for artifacts: {watch_dir}")

            self.observer.start()
            self._is_watching = True

        except Exception as e:
            logger.exception(f"Failed to start file system watching: {e}")
            raise

    def stop_watching(self):
        """Stop file system monitoring."""
        if not self._is_watching:
            return

        try:
            self.observer.stop()
            self.observer.join(timeout=5.0)
            self._is_watching = False
            logger.info("File system artifact detection stopped")
        except Exception as e:
            logger.exception(f"Error stopping file system detector: {e}")


class ArtifactFileHandler(FileSystemEventHandler):
    """Handle file system events for artifact detection."""

    def __init__(self, artifact_manager: ArtifactManager):
        self.artifact_manager = artifact_manager
        self.processed_files = set()

    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return

        try:
            file_path = Path(event.src_path)

            # Avoid processing the same file multiple times
            if str(file_path) in self.processed_files:
                return

            # Wait for file to be fully written with stability check
            if not self._wait_for_file_stability(file_path):
                logger.debug(f"File not stable or accessible, skipping: {file_path}")
                return

            artifact_type = self._detect_artifact_type(file_path)

            if artifact_type:
                # Auto-log detected artifact
                artifact = TraceletArtifact(
                    name=file_path.stem,
                    type=artifact_type,
                    description=f"Auto-detected {artifact_type.value} from filesystem",
                ).add_file(str(file_path))

                # Add filesystem metadata
                stat = file_path.stat()
                artifact.metadata.update({
                    "auto_detected": True,
                    "detection_method": "filesystem_watcher",
                    "file_size": stat.st_size,
                    "created_time": stat.st_ctime,
                    "modified_time": stat.st_mtime,
                })

                self.artifact_manager.log_artifact(artifact)
                self.processed_files.add(str(file_path))

                logger.info(f"Auto-detected and logged artifact: {artifact.name} ({artifact_type.value})")

        except Exception as e:
            logger.debug(f"Error processing file creation event: {e}")

    def _wait_for_file_stability(
        self, file_path: Path, max_wait_time: float = 5.0, check_interval: float = 0.1
    ) -> bool:
        """Wait for file to be fully written by checking size stability."""
        if not file_path.exists() or not file_path.is_file():
            return False

        try:
            start_time = time.time()
            last_size = -1
            stable_count = 0
            required_stable_checks = 3  # File size must be stable for 3 consecutive checks

            while time.time() - start_time < max_wait_time:
                try:
                    current_size = file_path.stat().st_size

                    if current_size == last_size:
                        stable_count += 1
                        if stable_count >= required_stable_checks:
                            # Additional check: try to open file to ensure it's not locked
                            try:
                                with open(file_path, "rb") as f:
                                    f.read(1)  # Try to read one byte
                                return True
                            except OSError:
                                # File might still be locked, continue waiting
                                pass
                    else:
                        stable_count = 0
                        last_size = current_size

                    time.sleep(check_interval)

                except (FileNotFoundError, OSError):
                    # File might have been deleted or moved
                    return False

            logger.debug(f"File stability timeout reached for {file_path}")
            return False

        except Exception as e:
            logger.debug(f"Error checking file stability for {file_path}: {e}")
            return False

    def on_modified(self, event):
        """Handle file modification events."""
        # For now, we only handle creation events to avoid duplicate logging
        # Could be extended to handle model updates, etc.
        pass

    def _detect_artifact_type(self, file_path: Path) -> Optional[ArtifactType]:  # noqa: C901
        """Detect artifact type from file extension/name."""
        suffix = file_path.suffix.lower()
        name = file_path.name.lower()

        # Model and checkpoint files
        if suffix in [".pth", ".pt", ".ckpt"]:
            return ArtifactType.CHECKPOINT if "checkpoint" in name else ArtifactType.MODEL
        elif suffix in [".pkl", ".pickle"] or suffix in [".h5", ".hdf5"] or suffix in [".joblib"]:
            return ArtifactType.MODEL

        # Media files
        elif suffix in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"]:
            return ArtifactType.IMAGE
        elif suffix in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
            return ArtifactType.AUDIO
        elif suffix in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            return ArtifactType.VIDEO

        # Configuration files
        elif suffix in [".yaml", ".yml", ".json", ".toml", ".ini"] or suffix in [".cfg", ".conf"]:
            return ArtifactType.CONFIG

        # Reports and documentation
        elif suffix in [".html", ".pdf"] or (
            suffix in [".md", ".rst", ".txt"] and any(keyword in name for keyword in ["report", "summary", "results"])
        ):
            return ArtifactType.REPORT

        # Code files
        elif suffix in [".py", ".js", ".cpp", ".java", ".c", ".h"]:
            return ArtifactType.CODE

        # Dataset files (common formats)
        elif suffix in [".csv", ".parquet", ".arrow", ".feather"] or (
            suffix in [".npz", ".npy"]
            and any(keyword in name for keyword in ["data", "dataset", "train", "test", "val"])
        ):
            return ArtifactType.DATASET

        # Visualization files
        elif suffix in [".svg", ".eps"] or "plot" in name or "chart" in name:
            return ArtifactType.VISUALIZATION

        # Check for specific naming patterns
        elif any(keyword in name for keyword in ["checkpoint", "ckpt"]):
            return ArtifactType.CHECKPOINT
        elif any(keyword in name for keyword in ["model", "weights"]):
            return ArtifactType.MODEL
        elif any(keyword in name for keyword in ["config", "params", "hyperparams"]):
            return ArtifactType.CONFIG
        elif any(keyword in name for keyword in ["sample", "prediction", "output"]):
            return ArtifactType.SAMPLE

        return None
