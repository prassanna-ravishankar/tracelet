"""Tests for enhanced TensorBoard integration with histogram, image, and other media support."""

import tempfile

import numpy as np
import pytest

from tracelet.core.experiment import Experiment, ExperimentConfig
from tracelet.core.orchestrator import MetricType
from tracelet.frameworks.pytorch import PyTorchFramework


class TestEnhancedTensorBoardIntegration:
    """Test enhanced TensorBoard features like histograms, images, etc."""

    def setup_method(self):
        """Setup test environment"""
        self.experiment = Experiment(name="Enhanced TensorBoard Test", config=ExperimentConfig(track_metrics=True))

    def test_enhanced_metric_types_available(self):
        """Test that all enhanced metric types are available"""
        expected_types = [
            MetricType.HISTOGRAM,
            MetricType.IMAGE,
            MetricType.TEXT,
            MetricType.FIGURE,
            MetricType.EMBEDDING,
            MetricType.VIDEO,
            MetricType.AUDIO,
            MetricType.MESH,
            MetricType.HPARAMS,
        ]

        for metric_type in expected_types:
            assert hasattr(MetricType, metric_type.name)
            assert isinstance(metric_type.value, str)

    def test_enhanced_metric_logging(self):
        """Test that enhanced metrics can be logged directly"""
        framework = PyTorchFramework()
        framework.initialize(self.experiment)

        # Mock the experiment's emit_metric method to capture calls
        captured_metrics = []
        original_emit = self.experiment.emit_metric

        def capture_emit(metric):
            captured_metrics.append(metric)
            return original_emit(metric)

        self.experiment.emit_metric = capture_emit

        # Test various enhanced metric types
        test_cases = [
            ("histogram", np.random.normal(0, 1, 100), MetricType.HISTOGRAM, {"bins": 50}),
            ("sample_image", np.random.rand(3, 64, 64), MetricType.IMAGE, {"format": "CHW"}),
            ("log_message", "Test log with **markdown**", MetricType.TEXT, {}),
            ("hyperparams", {"lr": 0.001, "batch": 32}, MetricType.HPARAMS, {"run": "test"}),
        ]

        for name, value, metric_type, metadata in test_cases:
            framework.log_enhanced_metric(name, value, metric_type, iteration=1, metadata=metadata)

        # Verify all metrics were captured
        assert len(captured_metrics) == len(test_cases)

        # Check specific metrics
        histogram_metrics = [m for m in captured_metrics if m.type == MetricType.HISTOGRAM]
        assert len(histogram_metrics) == 1
        assert histogram_metrics[0].name == "histogram"

        image_metrics = [m for m in captured_metrics if m.type == MetricType.IMAGE]
        assert len(image_metrics) == 1
        assert image_metrics[0].name == "sample_image"

        text_metrics = [m for m in captured_metrics if m.type == MetricType.TEXT]
        assert len(text_metrics) == 1
        assert text_metrics[0].value == "Test log with **markdown**"

    @pytest.mark.skipif(
        not hasattr(PyTorchFramework, "_check_tensorboard") or not PyTorchFramework._check_tensorboard(),
        reason="TensorBoard not available",
    )
    def test_tensorboard_integration_with_real_tensorboard(self):
        """Test integration with actual TensorBoard SummaryWriter"""
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            pytest.skip("TensorBoard not installed")

        framework = PyTorchFramework()
        framework.initialize(self.experiment)

        # Capture metrics
        captured_metrics = []
        original_emit = self.experiment.emit_metric

        def capture_emit(metric):
            captured_metrics.append(metric)
            return original_emit(metric)

        self.experiment.emit_metric = capture_emit

        # Start framework to enable patching
        framework.start_tracking()

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create a real TensorBoard writer
                writer = SummaryWriter(log_dir=temp_dir)

                # Test basic scalar (should be captured automatically)
                writer.add_scalar("test/accuracy", 0.95, 1)
                writer.add_scalars("test/metrics", {"loss": 0.1, "acc": 0.9}, 2)

                # Close writer
                writer.close()

                # Check that metrics were captured
                scalar_metrics = [m for m in captured_metrics if "accuracy" in m.name]
                scalars_metrics = [m for m in captured_metrics if "metrics/" in m.name]

                assert len(scalar_metrics) >= 1, f"Expected scalar metrics, got {[m.name for m in captured_metrics]}"
                assert len(scalars_metrics) >= 1, f"Expected scalars metrics, got {[m.name for m in captured_metrics]}"

        finally:
            framework.stop_tracking()

    def test_backward_compatibility(self):
        """Test that existing scalar functionality still works with enhanced framework"""
        framework = PyTorchFramework()
        framework.initialize(self.experiment)

        captured_metrics = []
        original_emit = self.experiment.emit_metric

        def capture_emit(metric):
            captured_metrics.append(metric)
            return original_emit(metric)

        self.experiment.emit_metric = capture_emit

        # Test the existing log_metric method (should still work)
        framework.log_metric("accuracy", 0.95, 10)

        # Check that scalar metric was captured
        scalar_metrics = [m for m in captured_metrics if m.name == "accuracy"]
        assert len(scalar_metrics) > 0

        metric = scalar_metrics[0]
        assert metric.name == "accuracy"
        assert metric.value == 0.95
        assert metric.iteration == 10

    def test_framework_initialization_with_tensorboard_detection(self):
        """Test framework properly detects TensorBoard availability"""
        framework = PyTorchFramework()

        # The framework should detect TensorBoard availability
        # (True if installed, False if not)
        assert isinstance(framework._tensorboard_available, bool)

        # All enhanced method placeholders should be initialized
        enhanced_methods = [
            "_original_add_histogram",
            "_original_add_image",
            "_original_add_text",
            "_original_add_figure",
            "_original_add_embedding",
            "_original_add_video",
            "_original_add_audio",
            "_original_add_mesh",
            "_original_add_hparams",
        ]

        for method in enhanced_methods:
            assert hasattr(framework, method)
            assert getattr(framework, method) is None  # Should be None initially

    def test_custom_scalar_plots_via_figure_and_hparams(self):
        """Test custom scalar plot functionality through figure and hparams"""
        framework = PyTorchFramework()
        framework.initialize(self.experiment)

        captured_metrics = []
        original_emit = self.experiment.emit_metric

        def capture_emit(metric):
            captured_metrics.append(metric)
            return original_emit(metric)

        self.experiment.emit_metric = capture_emit

        # Test custom dashboard via hyperparameters (custom scalar plots)
        hparam_dict = {"learning_rate": 0.001, "batch_size": 32, "optimizer": "adam", "weight_decay": 0.0001}
        metric_dict = {"train_accuracy": 0.95, "val_accuracy": 0.92, "train_loss": 0.05, "val_loss": 0.08}

        framework.log_enhanced_metric(
            "custom_dashboard",
            {"hparams": hparam_dict, "metrics": metric_dict},
            MetricType.HPARAMS,
            iteration=None,  # Dashboard configs don't need iterations
            metadata={"dashboard_type": "custom_scalar_plot", "run_name": "custom_dashboard"},
        )

        # Test custom figure (matplotlib-based custom plots)
        custom_plot_data = {
            "x_values": [1, 2, 3, 4, 5],
            "y_values": [0.1, 0.3, 0.7, 0.9, 0.95],
            "plot_type": "custom_accuracy_curve",
        }

        framework.log_enhanced_metric(
            "custom_accuracy_plot",
            custom_plot_data,
            MetricType.FIGURE,
            iteration=5,
            metadata={"plot_config": "accuracy_over_epochs", "axes_config": "logarithmic"},
        )

        # Verify custom scalar plot metrics were captured
        hparam_metrics = [m for m in captured_metrics if m.type == MetricType.HPARAMS]
        figure_metrics = [m for m in captured_metrics if m.type == MetricType.FIGURE]

        assert len(hparam_metrics) == 1
        assert len(figure_metrics) == 1

        # Check hyperparameter dashboard details
        hparam_metric = hparam_metrics[0]
        assert "hparams" in hparam_metric.value
        assert "metrics" in hparam_metric.value
        assert hparam_metric.metadata["dashboard_type"] == "custom_scalar_plot"

        # Check custom figure details
        figure_metric = figure_metrics[0]
        assert figure_metric.name == "custom_accuracy_plot"
        assert "plot_type" in figure_metric.value
        assert figure_metric.metadata["plot_config"] == "accuracy_over_epochs"

    def test_multi_line_scalar_plots_via_add_scalars(self):
        """Test multi-line scalar plots functionality"""
        framework = PyTorchFramework()
        framework.initialize(self.experiment)

        captured_metrics = []
        original_emit = self.experiment.emit_metric

        def capture_emit(metric):
            captured_metrics.append(metric)
            return original_emit(metric)

        self.experiment.emit_metric = capture_emit

        # Simulate multi-line scalar plots (multiple related metrics)
        framework.log_metric("loss/train", 0.1, 1)
        framework.log_metric("loss/validation", 0.15, 1)
        framework.log_metric("accuracy/train", 0.95, 1)
        framework.log_metric("accuracy/validation", 0.92, 1)

        # This simulates what add_scalars would capture
        multi_scalar_data = {"train": 0.1, "validation": 0.15, "test": 0.12}

        # Log it as if it came from add_scalars (multi-line plot)
        for key, value in multi_scalar_data.items():
            framework.log_metric(f"loss_comparison/{key}", value, 2)

        # Verify all metrics were captured
        loss_metrics = [m for m in captured_metrics if "loss" in m.name]
        accuracy_metrics = [m for m in captured_metrics if "accuracy" in m.name]

        # Should have individual loss metrics + comparison metrics
        assert len(loss_metrics) >= 5  # 2 original + 3 comparison
        assert len(accuracy_metrics) == 2

        # Check that comparison metrics are properly named (multi-line plot style)
        comparison_metrics = [m for m in captured_metrics if "loss_comparison/" in m.name]
        assert len(comparison_metrics) == 3
