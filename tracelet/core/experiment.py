import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..automagic.core import AutomagicConfig

from .orchestrator import DataFlowOrchestrator, MetricData, MetricSource, MetricType, RoutingRule
from .plugins import PluginManager, PluginState, PluginType


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

    # Automagic instrumentation settings
    enable_automagic: bool = False
    automagic_frameworks: Optional[set[str]] = None


class Experiment(MetricSource):
    """Main experiment tracking class that orchestrates all tracking functionality"""

    def __init__(
        self,
        name: str,
        config: Optional[ExperimentConfig] = None,
        backend: Optional[list[str]] = None,  # Changed to list[str]
        tags: Optional[list[str]] = None,
        automagic: bool = False,  # Enable automagic instrumentation
        automagic_config: Optional["AutomagicConfig"] = None,  # Custom automagic configuration
    ):
        self.name = name
        self.id = str(uuid.uuid4())
        self.config = config or ExperimentConfig()
        self.created_at = datetime.now(timezone.utc)
        self.tags = tags or []
        self._current_iteration = 0
        self._active_collectors = []
        self._backends = backend if backend is not None else []  # Changed to _backends
        self._framework = None

        # Automagic instrumentation
        self._automagic_enabled = automagic or self.config.enable_automagic
        self._automagic_config = automagic_config
        self._automagic_instrumentor = None

        # Initialize data flow orchestrator
        self._orchestrator = DataFlowOrchestrator(max_queue_size=10000, num_workers=4)

        # Initialize plugin manager
        self._plugin_manager = PluginManager()

        self._initialize()

    def get_source_id(self) -> str:
        """Return unique identifier for this experiment source"""
        return f"experiment_{self.id}"

    def emit_metric(self, metric: MetricData):
        """Emit a metric to the orchestrator"""
        self._orchestrator.emit_metric(metric)

    def _initialize(self):
        """Initialize all enabled collectors and backends"""
        # Register this experiment as a source
        self._orchestrator.register_source(self)

        # Discover available plugins
        self._plugin_manager.discover_plugins()

        # Setup backend plugins if specified
        if self._backends:
            backend_plugins = self._plugin_manager.get_plugins_by_type(PluginType.BACKEND)
            available_backend_names = {p.metadata.name for p in backend_plugins}

            for backend_name in self._backends:
                if backend_name in available_backend_names:
                    if self._plugin_manager.initialize_plugin(backend_name):
                        backend_instance = self._plugin_manager.get_plugin_instance(backend_name)
                        self._orchestrator.register_sink(backend_instance)
                        self._orchestrator.add_routing_rule(
                            RoutingRule(source_pattern="*", sink_id=backend_instance.get_sink_id())
                        )
                else:
                    print(f"Warning: Backend '{backend_name}' not found or could not be initialized.")

        # Initialize automagic instrumentation if enabled
        if self._automagic_enabled:
            self._initialize_automagic()

    def start(self):
        """Start the experiment tracking"""
        # Start the orchestrator
        self._orchestrator.start()

        # Start backend plugins
        for backend_name in self._backends:
            self._plugin_manager.start_plugin(backend_name)

        # Start collector plugins
        for collector_info in self._plugin_manager.get_plugins_by_type(PluginType.COLLECTOR):
            if self._plugin_manager.initialize_plugin(collector_info.metadata.name):
                self._plugin_manager.start_plugin(collector_info.metadata.name)

    def stop(self):
        """Stop the experiment tracking and clean up resources"""
        # Clean up automagic instrumentation first
        if self._automagic_enabled and self._automagic_instrumentor:
            self._automagic_instrumentor.detach_experiment(self.id)

        # Stop all active plugins
        for plugin_name, plugin_info in self._plugin_manager.plugins.items():
            if plugin_info.state == PluginState.ACTIVE:
                self._plugin_manager.stop_plugin(plugin_name)

        # Stop the orchestrator
        self._orchestrator.stop()

    def log_metric(self, name: str, value: Any, iteration: Optional[int] = None):
        """Log a metric value"""
        iteration = iteration or self._current_iteration

        # Create metric data
        metric = MetricData(
            name=name,
            value=value,
            type=MetricType.SCALAR if isinstance(value, (int, float)) else MetricType.CUSTOM,
            iteration=iteration,
            source=self.get_source_id(),
        )

        # Emit to orchestrator
        self.emit_metric(metric)

    def log_params(self, params: dict[str, Any]):
        """Log experiment parameters"""
        for name, value in params.items():
            metric = MetricData(
                name=name,
                value=value,
                type=MetricType.PARAMETER,
                iteration=None,  # Parameters don't have iterations
                source=self.get_source_id(),
            )
            self.emit_metric(metric)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a local file as an artifact"""
        metric = MetricData(
            name=artifact_path or local_path,
            value=local_path,
            type=MetricType.ARTIFACT,
            iteration=None,  # Artifacts don't have iterations
            source=self.get_source_id(),
            metadata={"artifact_path": artifact_path},
        )
        self.emit_metric(metric)

    def set_iteration(self, iteration: int):
        """Set the current iteration"""
        self._current_iteration = iteration

    @property
    def iteration(self) -> int:
        """Get current iteration"""
        return self._current_iteration

    def _initialize_automagic(self):
        """Initialize automagic instrumentation."""
        try:
            from ..automagic import AutomagicConfig, AutomagicInstrumentor

            # Use provided automagic_config if available, otherwise create from ExperimentConfig
            if self._automagic_config is not None:
                automagic_config = self._automagic_config
            else:
                # Create automagic configuration from ExperimentConfig
                # Use explicit None check to allow intentional empty set
                if self.config.automagic_frameworks is not None:
                    automagic_config = AutomagicConfig(frameworks=self.config.automagic_frameworks)
                else:
                    # Use AutomagicConfig defaults when no frameworks specified
                    automagic_config = AutomagicConfig()

            # Initialize instrumentor and attach to this experiment
            self._automagic_instrumentor = AutomagicInstrumentor.get_instance(automagic_config)
            self._automagic_instrumentor.attach_experiment(self)

        except ImportError:
            print("Warning: Automagic instrumentation not available. Install optional dependencies.")
            self._automagic_enabled = False

    def capture_hyperparams(self) -> dict[str, Any]:
        """Capture hyperparameters from calling context using automagic instrumentation."""
        if not self._automagic_enabled or not self._automagic_instrumentor:
            return {}

        return self._automagic_instrumentor.capture_hyperparameters(self, frame_depth=2)

    def capture_model(self, model: Any) -> dict[str, Any]:
        """Capture model information using automagic instrumentation."""
        if not self._automagic_enabled or not self._automagic_instrumentor:
            return {}

        return self._automagic_instrumentor.capture_model_info(model, self)

    def capture_dataset(self, dataset: Any) -> dict[str, Any]:
        """Capture dataset information using automagic instrumentation."""
        if not self._automagic_enabled or not self._automagic_instrumentor:
            return {}

        return self._automagic_instrumentor.capture_dataset_info(dataset, self)

    def log_hyperparameter(self, name: str, value: Any):
        """Log a hyperparameter (alias for compatibility)."""
        # Create metric data directly to avoid intermediate dictionary creation
        metric = MetricData(
            name=name,
            value=value,
            type=MetricType.PARAMETER,
            iteration=None,  # Parameters don't have iterations
            source=self.get_source_id(),
        )
        self.emit_metric(metric)

    def end(self):
        """End the experiment and clean up resources (alias for stop)."""
        self.stop()
