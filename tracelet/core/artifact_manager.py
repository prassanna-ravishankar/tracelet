"""
Artifact manager with intelligent routing and optimization.

The ArtifactManager coordinates multiple platform-specific handlers to provide
optimal artifact logging based on artifact type, size, and platform capabilities.
"""

import logging
from typing import Optional

from .artifact_handlers import (
    AiMArtifactHandler,
    ArtifactHandler,
    ClearMLArtifactHandler,
    MLflowArtifactHandler,
    WANDBArtifactHandler,
)
from .artifacts import LARGE_FILE_THRESHOLD, ArtifactResult, ArtifactType, TraceletArtifact

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Central artifact management with intelligent routing."""

    def __init__(self, backend_instances: dict[str, any]):
        """Initialize with backend instances."""
        self.backend_instances = backend_instances
        self.handlers: list[ArtifactHandler] = []
        self.routing_cache: dict[tuple, list[ArtifactHandler]] = {}

        # Create handlers for available backends
        self._create_handlers()

    def _create_handlers(self):
        """Create platform-specific handlers for available backends."""
        for backend_name, backend_instance in self.backend_instances.items():
            try:
                if backend_name == "mlflow":
                    self.handlers.append(MLflowArtifactHandler(backend_instance))
                elif backend_name == "wandb":
                    self.handlers.append(WANDBArtifactHandler(backend_instance))
                elif backend_name == "clearml":
                    self.handlers.append(ClearMLArtifactHandler(backend_instance))
                elif backend_name == "aim":
                    self.handlers.append(AiMArtifactHandler(backend_instance))
                else:
                    logger.warning(f"No artifact handler available for backend: {backend_name}")
            except Exception as e:
                logger.exception(f"Failed to create artifact handler for {backend_name}: {e}")

        logger.info(f"Created {len(self.handlers)} artifact handlers")

    def log_artifact(self, artifact: TraceletArtifact) -> dict[str, ArtifactResult]:
        """Log artifact using optimal handler(s)."""
        if not self.handlers:
            logger.warning("No artifact handlers available")
            return {}

        results = {}
        optimal_handlers = self._get_optimal_handlers(artifact)

        for handler in optimal_handlers:
            try:
                result = handler.log_artifact(artifact)
                handler_name = handler.__class__.__name__.replace("ArtifactHandler", "").lower()
                results[handler_name] = result
                logger.info(f"Successfully logged artifact '{artifact.name}' to {handler_name}")
            except Exception as e:
                handler_name = handler.__class__.__name__
                logger.exception(f"Failed to log artifact '{artifact.name}' with {handler_name}: {e}")

        if not results:
            logger.error(f"Failed to log artifact '{artifact.name}' to any backend")

        return results

    def get_artifact(
        self, name: str, version: str = "latest", backend: Optional[str] = None
    ) -> Optional[TraceletArtifact]:
        """Retrieve artifact from specified backend or search all."""
        if backend:
            # Try specific backend
            handler = self._get_handler_by_name(backend)
            if handler:
                return handler.get_artifact(name, version)
        else:
            # Search all handlers
            for handler in self.handlers:
                try:
                    artifact = handler.get_artifact(name, version)
                    if artifact:
                        return artifact
                except Exception as e:
                    logger.debug(f"Failed to retrieve from {handler.__class__.__name__}: {e}")

        return None

    def list_artifacts(
        self, type_filter: Optional[ArtifactType] = None, backend: Optional[str] = None
    ) -> list[TraceletArtifact]:
        """List artifacts from specified backend or all backends."""
        artifacts = []

        handlers = [self._get_handler_by_name(backend)] if backend else self.handlers
        handlers = [h for h in handlers if h is not None]

        for handler in handlers:
            try:
                handler_artifacts = handler.list_artifacts(type_filter)
                artifacts.extend(handler_artifacts)
            except Exception as e:
                logger.debug(f"Failed to list from {handler.__class__.__name__}: {e}")

        return artifacts

    def _get_optimal_handlers(self, artifact: TraceletArtifact) -> list[ArtifactHandler]:
        """Select optimal handlers based on artifact type and size."""
        # Create cache key
        cache_key = (artifact.type, artifact.size_bytes > LARGE_FILE_THRESHOLD)

        if cache_key in self.routing_cache:
            return self.routing_cache[cache_key]

        # Find handlers that claim to be optimal for this type
        optimal_handlers = []
        general_handlers = []

        for handler in self.handlers:
            if handler.should_handle_type(artifact.type):
                optimal_handlers.append(handler)
            else:
                general_handlers.append(handler)

        # If no optimal handlers found, use all handlers
        if not optimal_handlers:
            optimal_handlers = self.handlers

        # For large files, sort by large file capability
        if artifact.size_bytes > LARGE_FILE_THRESHOLD:
            optimal_handlers = sorted(optimal_handlers, key=lambda h: h.get_large_file_score(), reverse=True)

        # Cache the result
        self.routing_cache[cache_key] = optimal_handlers

        return optimal_handlers

    def _get_handler_by_name(self, backend_name: str) -> Optional[ArtifactHandler]:
        """Get handler by backend name."""
        name_mapping = {
            "mlflow": MLflowArtifactHandler,
            "wandb": WANDBArtifactHandler,
            "clearml": ClearMLArtifactHandler,
            "aim": AiMArtifactHandler,
        }

        target_class = name_mapping.get(backend_name.lower())
        if not target_class:
            return None

        for handler in self.handlers:
            if isinstance(handler, target_class):
                return handler

        return None

    def get_stats(self) -> dict[str, any]:
        """Get artifact manager statistics."""
        handler_types = [h.__class__.__name__ for h in self.handlers]
        backend_names = list(self.backend_instances.keys())

        return {
            "num_handlers": len(self.handlers),
            "handler_types": handler_types,
            "backend_names": backend_names,
            "cache_size": len(self.routing_cache),
            "large_file_threshold_mb": LARGE_FILE_THRESHOLD / (1024 * 1024),
        }

    def clear_cache(self):
        """Clear routing cache."""
        self.routing_cache.clear()
        logger.info("Artifact routing cache cleared")
