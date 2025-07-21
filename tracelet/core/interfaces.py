from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .experiment import Experiment


class FrameworkInterface(ABC):
    """Base interface for ML framework integrations"""

    @abstractmethod
    def initialize(self, experiment: "Experiment"):
        """Initialize the framework binding"""
        pass

    @abstractmethod
    def start_tracking(self):
        """Start tracking metrics from the framework"""
        pass

    @abstractmethod
    def stop_tracking(self):
        """Stop tracking metrics"""
        pass

    @abstractmethod
    def log_metric(self, name: str, value: Any, iteration: int):
        """Log a metric from the framework"""
        pass


class BackendInterface(ABC):
    """Base interface for storage backend integrations"""

    @abstractmethod
    def initialize(self):
        """Initialize connection to the backend"""
        pass

    @abstractmethod
    def log_metric(self, name: str, value: Any, iteration: int):
        """Log a metric to the backend"""
        pass

    @abstractmethod
    def log_params(self, params: dict[str, Any]):
        """Log parameters to the backend"""
        pass

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Upload an artifact to the backend"""
        pass

    @abstractmethod
    def save_experiment(self, experiment_data: dict[str, Any]):
        """Save experiment metadata"""
        pass


class CollectorInterface(ABC):
    """Base interface for various collectors (git, env, system metrics, etc.)"""

    @abstractmethod
    def initialize(self):
        """Initialize the collector"""
        pass

    @abstractmethod
    def collect(self) -> dict[str, Any]:
        """Collect data"""
        pass

    @abstractmethod
    def start(self):
        """Start collecting (for continuous collectors)"""
        pass

    @abstractmethod
    def stop(self):
        """Stop collecting"""
        pass
