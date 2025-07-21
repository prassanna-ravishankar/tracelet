import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can flow through the system"""

    SCALAR = "scalar"
    TENSOR = "tensor"
    ARTIFACT = "artifact"
    PARAMETER = "parameter"
    SYSTEM = "system"
    CUSTOM = "custom"
    # Enhanced TensorBoard types
    HISTOGRAM = "histogram"
    IMAGE = "image"
    TEXT = "text"
    FIGURE = "figure"
    EMBEDDING = "embedding"
    VIDEO = "video"
    AUDIO = "audio"
    MESH = "mesh"
    HPARAMS = "hparams"


@dataclass
class MetricData:
    """Container for metric data flowing through the system"""

    name: str
    value: Any
    type: MetricType
    iteration: Optional[int] = None
    timestamp: Optional[float] = None
    source: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class MetricSource(ABC):
    """Base class for metric sources that produce data"""

    @abstractmethod
    def get_source_id(self) -> str:
        """Return unique identifier for this source"""
        pass

    @abstractmethod
    def emit_metric(self, metric: MetricData):
        """Emit a metric to the orchestrator"""
        pass


class MetricSink(ABC):
    """Base class for metric sinks that consume data"""

    @abstractmethod
    def get_sink_id(self) -> str:
        """Return unique identifier for this sink"""
        pass

    @abstractmethod
    def receive_metric(self, metric: MetricData):
        """Receive and process a metric"""
        pass

    @abstractmethod
    def can_handle_type(self, metric_type: MetricType) -> bool:
        """Check if this sink can handle the given metric type"""
        pass


@dataclass
class RoutingRule:
    """Rule for routing metrics from sources to sinks"""

    source_pattern: str  # Can use wildcards
    sink_id: str
    metric_types: Optional[set[MetricType]] = None
    filter_func: Optional[Callable[[MetricData], bool]] = None

    def matches(self, metric: MetricData) -> bool:
        """Check if this rule matches the given metric"""
        # Check source pattern
        if (
            self.source_pattern != "*"
            and metric.source
            and not self._matches_pattern(metric.source, self.source_pattern)
        ):
            return False

        # Check metric type
        if self.metric_types and metric.type not in self.metric_types:
            return False

        # Apply custom filter
        return not (self.filter_func and not self.filter_func(metric))

    def _matches_pattern(self, source: str, pattern: str) -> bool:
        """Simple wildcard pattern matching"""
        if pattern == "*":
            return True
        if "*" in pattern:
            prefix = pattern.split("*")[0]
            return source.startswith(prefix)
        return source == pattern


class DataFlowOrchestrator:
    """Orchestrates data flow between metric sources and sinks"""

    def __init__(self, max_queue_size: int = 10000, num_workers: int = 4):
        self.sources: dict[str, MetricSource] = {}
        self.sinks: dict[str, MetricSink] = {}
        self.routing_rules: list[RoutingRule] = []
        self.max_queue_size = max_queue_size

        # Thread-safe queue for metrics
        self.metric_queue = queue.Queue(maxsize=max_queue_size)
        self.num_workers = num_workers
        self.workers: list[threading.Thread] = []
        self.running = False
        self._lock = threading.RLock()

        # Metrics for monitoring
        self.metrics_processed = 0
        self.metrics_dropped = 0
        self.processing_errors = 0

    def register_source(self, source: MetricSource):
        """Register a metric source"""
        with self._lock:
            source_id = source.get_source_id()
            if source_id in self.sources:
                msg = f"Source '{source_id}' already registered"
                raise ValueError(msg)
            self.sources[source_id] = source
            logger.info(f"Registered metric source: {source_id}")

    def register_sink(self, sink: MetricSink):
        """Register a metric sink"""
        with self._lock:
            sink_id = sink.get_sink_id()
            if sink_id in self.sinks:
                msg = f"Sink '{sink_id}' already registered"
                raise ValueError(msg)
            self.sinks[sink_id] = sink
            logger.info(f"Registered metric sink: {sink_id}")

    def add_routing_rule(self, rule: RoutingRule):
        """Add a routing rule"""
        with self._lock:
            self.routing_rules.append(rule)
            logger.info(f"Added routing rule: {rule.source_pattern} -> {rule.sink_id}")

    def emit_metric(self, metric: MetricData):
        """Emit a metric to the orchestrator"""
        try:
            self.metric_queue.put_nowait(metric)
        except queue.Full:
            self.metrics_dropped += 1
            logger.warning(f"Queue full, dropping metric: {metric.name}")
            if self.metrics_dropped % 100 == 0:
                logger.exception(f"Dropped {self.metrics_dropped} metrics due to backpressure")

    def start(self):
        """Start the orchestrator workers"""
        with self._lock:
            if self.running:
                return

            self.running = True

            # Start worker threads
            for i in range(self.num_workers):
                worker = threading.Thread(target=self._worker_loop, name=f"DataFlowWorker-{i}", daemon=True)
                worker.start()
                self.workers.append(worker)

            logger.info(f"Started {self.num_workers} worker threads")

    def stop(self, timeout: float = 5.0):
        """Stop the orchestrator workers"""
        with self._lock:
            if not self.running:
                return

            self.running = False

            # Add sentinel values to wake up workers
            for _ in range(self.num_workers):
                with suppress(queue.Full):
                    self.metric_queue.put(None, timeout=timeout / 2)

            # Wait for workers to finish
            for worker in self.workers:
                worker.join(timeout=timeout)
                if worker.is_alive():
                    logger.warning(f"Worker {worker.name} did not stop gracefully")

            self.workers.clear()
            logger.info("Stopped all worker threads")

    def _worker_loop(self):
        """Worker thread loop for processing metrics"""
        while self.running:
            try:
                # Get metric with timeout to allow checking running flag
                metric = self.metric_queue.get(timeout=1.0)

                if metric is None:  # Sentinel value
                    break

                self._process_metric(metric)
                self.metrics_processed += 1

            except queue.Empty:
                continue
            except Exception as e:
                self.processing_errors += 1
                logger.error(f"Error processing metric: {e}", exc_info=True)

    def _process_metric(self, metric: MetricData):
        """Process a single metric through routing rules"""
        matched_sinks = set()

        with self._lock:
            # Find matching routing rules
            for rule in self.routing_rules:
                if rule.matches(metric) and rule.sink_id in self.sinks:
                    matched_sinks.add(rule.sink_id)

            # Send to matched sinks
            for sink_id in matched_sinks:
                sink = self.sinks[sink_id]
                if sink.can_handle_type(metric.type):
                    try:
                        sink.receive_metric(metric)
                    except Exception:
                        self.processing_errors += 1
                        logger.exception(f"Sink '{sink_id}' failed to process metric")

        if not matched_sinks:
            logger.debug(f"No sinks matched for metric: {metric.name}")

    @contextmanager
    def batch_emit(self, source_id: str):
        """Context manager for batch emitting metrics"""
        batch = []

        def batch_emit_func(metric: MetricData):
            metric.source = source_id
            batch.append(metric)

        yield batch_emit_func

        # Emit all metrics in batch
        for metric in batch:
            self.emit_metric(metric)

    def get_stats(self) -> dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            "metrics_processed": self.metrics_processed,
            "metrics_dropped": self.metrics_dropped,
            "processing_errors": self.processing_errors,
            "queue_size": self.metric_queue.qsize(),
            "num_sources": len(self.sources),
            "num_sinks": len(self.sinks),
            "num_routing_rules": len(self.routing_rules),
            "running": self.running,
        }
