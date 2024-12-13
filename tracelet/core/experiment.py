from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
import uuid

@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking"""
    track_metrics: bool = True
    track_environment: bool = True
    track_args: bool = True
    track_stdout: bool = True
    track_checkpoints: bool = True
    track_system_metrics: bool = True
    track_git: bool = True
    
class Experiment:
    """Main experiment tracking class that orchestrates all tracking functionality"""
    
    def __init__(
        self,
        name: str,
        config: Optional[ExperimentConfig] = None,
        backend: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        self.name = name
        self.id = str(uuid.uuid4())
        self.config = config or ExperimentConfig()
        self.created_at = datetime.utcnow()
        self.tags = tags or []
        self._current_iteration = 0
        self._active_collectors = []
        self._backend = None
        self._framework = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize all enabled collectors and backend"""
        # This will be implemented to set up all the tracking components
        pass
        
    def start(self):
        """Start the experiment tracking"""
        # Initialize all collectors
        pass
        
    def stop(self):
        """Stop the experiment tracking"""
        # Cleanup and flush all data
        pass
        
    def log_metric(self, name: str, value: Any, iteration: Optional[int] = None):
        """Log a metric value"""
        iteration = iteration or self._current_iteration
        # Implementation will delegate to active framework and backend
        pass
        
    def log_params(self, params: Dict[str, Any]):
        """Log experiment parameters"""
        pass
        
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a local file as an artifact"""
        pass
        
    def set_iteration(self, iteration: int):
        """Set the current iteration"""
        self._current_iteration = iteration
        
    @property
    def iteration(self) -> int:
        """Get current iteration"""
        return self._current_iteration 