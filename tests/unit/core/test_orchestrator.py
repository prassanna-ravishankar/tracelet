import threading
import time

import pytest

from tracelet.core.orchestrator import (
    DataFlowOrchestrator,
    MetricData,
    MetricSink,
    MetricSource,
    MetricType,
    RoutingRule,
)


class TestMetricSource(MetricSource):
    """Test implementation of MetricSource"""

    def __init__(self, source_id: str):
        self.source_id = source_id
        self.orchestrator = None

    def get_source_id(self) -> str:
        return self.source_id

    def emit_metric(self, metric: MetricData):
        if self.orchestrator:
            metric.source = self.source_id
            self.orchestrator.emit_metric(metric)


class TestMetricSink(MetricSink):
    """Test implementation of MetricSink"""

    def __init__(self, sink_id: str, accepted_types=None):
        self.sink_id = sink_id
        self.accepted_types = accepted_types or set(MetricType)
        self.received_metrics = []
        self.lock = threading.Lock()

    def get_sink_id(self) -> str:
        return self.sink_id

    def receive_metric(self, metric: MetricData):
        with self.lock:
            self.received_metrics.append(metric)

    def can_handle_type(self, metric_type: MetricType) -> bool:
        return metric_type in self.accepted_types


class TestDataFlowOrchestrator:
    """Test suite for DataFlowOrchestrator"""

    def test_initialization(self):
        """Test orchestrator initialization"""
        orchestrator = DataFlowOrchestrator(max_queue_size=100, num_workers=2)

        assert orchestrator.max_queue_size == 100
        assert orchestrator.num_workers == 2
        assert not orchestrator.running
        assert len(orchestrator.sources) == 0
        assert len(orchestrator.sinks) == 0
        assert len(orchestrator.routing_rules) == 0

    def test_register_source(self):
        """Test source registration"""
        orchestrator = DataFlowOrchestrator()
        source = TestMetricSource("test_source")

        orchestrator.register_source(source)
        assert "test_source" in orchestrator.sources
        assert orchestrator.sources["test_source"] == source

        # Test duplicate registration
        with pytest.raises(ValueError, match="already registered"):
            orchestrator.register_source(source)

    def test_register_sink(self):
        """Test sink registration"""
        orchestrator = DataFlowOrchestrator()
        sink = TestMetricSink("test_sink")

        orchestrator.register_sink(sink)
        assert "test_sink" in orchestrator.sinks
        assert orchestrator.sinks["test_sink"] == sink

        # Test duplicate registration
        with pytest.raises(ValueError, match="already registered"):
            orchestrator.register_sink(sink)

    def test_routing_rules(self):
        """Test routing rule functionality"""
        rule = RoutingRule(
            source_pattern="test_*",
            sink_id="sink1",
            metric_types={MetricType.SCALAR}
        )

        # Test matching
        metric1 = MetricData(
            name="metric1",
            value=1.0,
            type=MetricType.SCALAR,
            source="test_source"
        )
        assert rule.matches(metric1)

        # Test non-matching source
        metric2 = MetricData(
            name="metric2",
            value=1.0,
            type=MetricType.SCALAR,
            source="other_source"
        )
        assert not rule.matches(metric2)

        # Test non-matching type
        metric3 = MetricData(
            name="metric3",
            value="artifact",
            type=MetricType.ARTIFACT,
            source="test_source"
        )
        assert not rule.matches(metric3)

    def test_basic_flow(self):
        """Test basic metric flow from source to sink"""
        orchestrator = DataFlowOrchestrator(num_workers=1)

        # Setup
        source = TestMetricSource("source1")
        source.orchestrator = orchestrator
        sink = TestMetricSink("sink1")

        orchestrator.register_source(source)
        orchestrator.register_sink(sink)
        orchestrator.add_routing_rule(
            RoutingRule(source_pattern="*", sink_id="sink1")
        )

        # Start orchestrator
        orchestrator.start()

        try:
            # Emit metric
            metric = MetricData(
                name="test_metric",
                value=42.0,
                type=MetricType.SCALAR
            )
            source.emit_metric(metric)

            # Wait for processing
            time.sleep(0.1)

            # Verify
            assert len(sink.received_metrics) == 1
            received = sink.received_metrics[0]
            assert received.name == "test_metric"
            assert received.value == 42.0
            assert received.source == "source1"
        finally:
            orchestrator.stop()

    def test_multiple_sinks(self):
        """Test routing to multiple sinks"""
        orchestrator = DataFlowOrchestrator(num_workers=2)

        # Setup multiple sinks
        sink1 = TestMetricSink("sink1", {MetricType.SCALAR})
        sink2 = TestMetricSink("sink2", {MetricType.SCALAR, MetricType.PARAMETER})

        orchestrator.register_sink(sink1)
        orchestrator.register_sink(sink2)

        # Add routing rules
        orchestrator.add_routing_rule(
            RoutingRule(source_pattern="*", sink_id="sink1")
        )
        orchestrator.add_routing_rule(
            RoutingRule(source_pattern="*", sink_id="sink2")
        )

        orchestrator.start()

        try:
            # Emit scalar metric
            metric = MetricData(
                name="scalar_metric",
                value=1.0,
                type=MetricType.SCALAR,
                source="test"
            )
            orchestrator.emit_metric(metric)

            time.sleep(0.1)

            # Both sinks should receive it
            assert len(sink1.received_metrics) == 1
            assert len(sink2.received_metrics) == 1
        finally:
            orchestrator.stop()

    def test_type_filtering(self):
        """Test metric type filtering in sinks"""
        orchestrator = DataFlowOrchestrator(num_workers=1)

        # Sink that only accepts scalars
        scalar_sink = TestMetricSink("scalar_sink", {MetricType.SCALAR})
        # Sink that only accepts artifacts
        artifact_sink = TestMetricSink("artifact_sink", {MetricType.ARTIFACT})

        orchestrator.register_sink(scalar_sink)
        orchestrator.register_sink(artifact_sink)

        # Route all metrics to both sinks
        orchestrator.add_routing_rule(
            RoutingRule(source_pattern="*", sink_id="scalar_sink")
        )
        orchestrator.add_routing_rule(
            RoutingRule(source_pattern="*", sink_id="artifact_sink")
        )

        orchestrator.start()

        try:
            # Emit different metric types
            scalar_metric = MetricData(
                name="scalar", value=1.0, type=MetricType.SCALAR
            )
            artifact_metric = MetricData(
                name="artifact", value="path/to/file", type=MetricType.ARTIFACT
            )

            orchestrator.emit_metric(scalar_metric)
            orchestrator.emit_metric(artifact_metric)

            time.sleep(0.1)

            # Verify filtering
            assert len(scalar_sink.received_metrics) == 1
            assert scalar_sink.received_metrics[0].type == MetricType.SCALAR

            assert len(artifact_sink.received_metrics) == 1
            assert artifact_sink.received_metrics[0].type == MetricType.ARTIFACT
        finally:
            orchestrator.stop()

    def test_backpressure_handling(self):
        """Test backpressure when queue is full"""
        # Small queue to test backpressure
        orchestrator = DataFlowOrchestrator(max_queue_size=5, num_workers=0)

        # Fill the queue
        for i in range(5):
            metric = MetricData(name=f"metric_{i}", value=i, type=MetricType.SCALAR)
            orchestrator.emit_metric(metric)

        # Queue should be full
        assert orchestrator.metric_queue.full()

        # Next metric should be dropped
        metric = MetricData(name="dropped", value=999, type=MetricType.SCALAR)
        orchestrator.emit_metric(metric)

        assert orchestrator.metrics_dropped == 1

    def test_worker_error_handling(self):
        """Test error handling in worker threads"""
        orchestrator = DataFlowOrchestrator(num_workers=1)

        # Sink that raises exception
        class FailingSink(MetricSink):
            def get_sink_id(self):
                return "failing_sink"

            def receive_metric(self, metric):
                raise ValueError("Intentional error")

            def can_handle_type(self, metric_type):
                return True

        sink = FailingSink()
        orchestrator.register_sink(sink)
        orchestrator.add_routing_rule(
            RoutingRule(source_pattern="*", sink_id="failing_sink")
        )

        orchestrator.start()

        try:
            # Emit metric that will cause error
            metric = MetricData(name="test", value=1, type=MetricType.SCALAR)
            orchestrator.emit_metric(metric)

            time.sleep(0.1)

            # Worker should handle error and continue
            assert orchestrator.metrics_processed == 1
            assert orchestrator.running
        finally:
            orchestrator.stop()

    def test_batch_emit(self):
        """Test batch emit context manager"""
        orchestrator = DataFlowOrchestrator(num_workers=1)
        sink = TestMetricSink("sink1")

        orchestrator.register_sink(sink)
        orchestrator.add_routing_rule(
            RoutingRule(source_pattern="batch_*", sink_id="sink1")
        )

        orchestrator.start()

        try:
            # Use batch emit
            with orchestrator.batch_emit("batch_source") as emit:
                for i in range(10):
                    emit(MetricData(
                        name=f"metric_{i}",
                        value=i,
                        type=MetricType.SCALAR
                    ))

            time.sleep(0.2)

            # All metrics should be received
            assert len(sink.received_metrics) == 10
            # All should have the batch source
            for metric in sink.received_metrics:
                assert metric.source == "batch_source"
        finally:
            orchestrator.stop()

    def test_custom_filter_function(self):
        """Test custom filter function in routing rules"""
        orchestrator = DataFlowOrchestrator(num_workers=1)
        sink = TestMetricSink("filtered_sink")

        orchestrator.register_sink(sink)

        # Only route metrics with value > 5
        def value_filter(metric: MetricData) -> bool:
            return metric.value > 5

        orchestrator.add_routing_rule(
            RoutingRule(
                source_pattern="*",
                sink_id="filtered_sink",
                filter_func=value_filter
            )
        )

        orchestrator.start()

        try:
            # Emit metrics with different values
            for i in range(10):
                metric = MetricData(
                    name=f"metric_{i}",
                    value=i,
                    type=MetricType.SCALAR
                )
                orchestrator.emit_metric(metric)

            time.sleep(0.1)

            # Only metrics with value > 5 should be received
            assert len(sink.received_metrics) == 4  # 6, 7, 8, 9
            for metric in sink.received_metrics:
                assert metric.value > 5
        finally:
            orchestrator.stop()

    def test_concurrent_access(self):
        """Test thread safety with concurrent access"""
        orchestrator = DataFlowOrchestrator(num_workers=4)
        sink = TestMetricSink("concurrent_sink")

        orchestrator.register_sink(sink)
        orchestrator.add_routing_rule(
            RoutingRule(source_pattern="*", sink_id="concurrent_sink")
        )

        orchestrator.start()

        try:
            # Multiple threads emitting metrics
            def emit_metrics(thread_id):
                for i in range(100):
                    metric = MetricData(
                        name=f"metric_t{thread_id}_i{i}",
                        value=i,
                        type=MetricType.SCALAR,
                        source=f"thread_{thread_id}"
                    )
                    orchestrator.emit_metric(metric)

            threads = []
            for i in range(5):
                t = threading.Thread(target=emit_metrics, args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # Wait for processing
            time.sleep(0.5)

            # All metrics should be processed
            assert len(sink.received_metrics) == 500
        finally:
            orchestrator.stop()

    def test_stats(self):
        """Test statistics collection"""
        orchestrator = DataFlowOrchestrator(num_workers=1)

        stats = orchestrator.get_stats()
        assert stats["metrics_processed"] == 0
        assert stats["metrics_dropped"] == 0
        assert stats["processing_errors"] == 0
        assert stats["running"] is False

        orchestrator.start()
        stats = orchestrator.get_stats()
        assert stats["running"] is True

        orchestrator.stop()
