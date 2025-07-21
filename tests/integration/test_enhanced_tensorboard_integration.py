"""Integration tests for enhanced TensorBoard features working together."""

import numpy as np

from tracelet.core.experiment import Experiment, ExperimentConfig
from tracelet.core.orchestrator import MetricType
from tracelet.frameworks.pytorch import PyTorchFramework


class TestEnhancedTensorBoardIntegration:
    """Integration tests for all enhanced TensorBoard features working together."""

    def setup_method(self):
        """Setup test environment"""
        self.experiment = Experiment(
            name="Enhanced TensorBoard Integration Test", config=ExperimentConfig(track_metrics=True)
        )
        self.framework = PyTorchFramework()
        self.framework.initialize(self.experiment)

    def test_complete_ml_workflow_simulation(self):
        """Test a complete ML workflow using all enhanced TensorBoard features"""
        # Mock the experiment's emit_metric method to capture all calls
        captured_metrics = []
        original_emit = self.experiment.emit_metric

        def capture_emit(metric):
            captured_metrics.append(metric)
            return original_emit(metric)

        self.experiment.emit_metric = capture_emit

        # Simulate a complete ML training workflow

        # 1. Log hyperparameters (custom scalar dashboard)
        hparams = {
            "model": "ResNet50",
            "learning_rate": 0.001,
            "batch_size": 32,
            "optimizer": "Adam",
            "weight_decay": 0.0001,
            "epochs": 10,
        }

        initial_metrics = {"initial_accuracy": 0.1, "initial_loss": 2.3, "model_parameters": 25_000_000}

        self.framework.log_enhanced_metric(
            "experiment_config",
            {"hparams": hparams, "metrics": initial_metrics},
            MetricType.HPARAMS,
            iteration=None,
            metadata={"experiment_type": "image_classification", "dataset": "CIFAR-10"},
        )

        # 2. Simulate training epochs with various metric types
        for epoch in range(5):
            # Basic scalar metrics (existing functionality)
            train_acc = 0.1 + (epoch * 0.15)  # Improving accuracy
            train_loss = 2.3 - (epoch * 0.4)  # Decreasing loss

            self.framework.log_metric("accuracy/train", train_acc, epoch)
            self.framework.log_metric("loss/train", train_loss, epoch)
            self.framework.log_metric("accuracy/validation", train_acc - 0.05, epoch)
            self.framework.log_metric("loss/validation", train_loss + 0.1, epoch)

            # 3. Log weight histograms (histogram logging)
            layer_weights = np.random.normal(0, 0.1, (512, 256))
            self.framework.log_enhanced_metric(
                f"weights/conv_layer_{epoch}",
                layer_weights,
                MetricType.HISTOGRAM,
                iteration=epoch,
                metadata={"layer_type": "conv2d", "shape": layer_weights.shape, "bins": 50},
            )

            # 4. Log gradient histograms
            gradients = np.random.normal(0, 0.01, (256, 128))
            self.framework.log_enhanced_metric(
                f"gradients/conv_layer_{epoch}",
                gradients,
                MetricType.HISTOGRAM,
                iteration=epoch,
                metadata={"gradient_type": "backprop", "shape": gradients.shape},
            )

            # 5. Log sample predictions as images (every 2 epochs)
            if epoch % 2 == 0:
                # Simulate batch of predicted images
                prediction_images = np.random.rand(4, 3, 32, 32)  # 4 CIFAR-10 images
                self.framework.log_enhanced_metric(
                    f"predictions/epoch_{epoch}",
                    prediction_images,
                    MetricType.IMAGE,
                    iteration=epoch,
                    metadata={"dataformats": "NCHW", "num_samples": 4, "image_size": [32, 32], "channels": 3},
                )

        # 6. Log training progress text
        progress_text = f"""
# Training Progress Summary

## Model Configuration
- Architecture: {hparams["model"]}
- Learning Rate: {hparams["learning_rate"]}
- Batch Size: {hparams["batch_size"]}

## Final Results
- Final Training Accuracy: {train_acc:.3f}
- Final Training Loss: {train_loss:.3f}
- Total Epochs Completed: 5

## Key Observations
- Model converged successfully
- No overfitting detected
- Ready for inference testing
        """

        self.framework.log_enhanced_metric(
            "training_summary",
            progress_text,
            MetricType.TEXT,
            iteration=5,
            metadata={"format": "markdown", "summary_type": "training_complete"},
        )

        # 7. Log custom visualization data (confusion matrix as figure)
        confusion_matrix_data = {
            "true_labels": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "predicted_labels": [0, 1, 1, 0, 1, 2, 0, 2, 2],
            "class_names": ["cat", "dog", "bird"],
            "accuracy": 0.78,
        }

        self.framework.log_enhanced_metric(
            "confusion_matrix",
            confusion_matrix_data,
            MetricType.FIGURE,
            iteration=5,
            metadata={"plot_type": "confusion_matrix", "num_classes": 3},
        )

        # Verify all metrics were captured correctly

        # Check we have metrics of all enhanced types
        metric_types_captured = {m.type for m in captured_metrics}
        expected_types = {
            MetricType.SCALAR,  # Basic accuracy/loss metrics
            MetricType.HISTOGRAM,  # Weight and gradient distributions
            MetricType.IMAGE,  # Sample predictions
            MetricType.TEXT,  # Training summary
            MetricType.FIGURE,  # Confusion matrix
            MetricType.HPARAMS,  # Hyperparameters dashboard
        }

        # Verify we captured all expected metric types
        for expected_type in expected_types:
            assert expected_type in metric_types_captured, f"Missing metric type: {expected_type}"

        # Verify specific metric counts
        scalar_metrics = [m for m in captured_metrics if m.type == MetricType.SCALAR]
        histogram_metrics = [m for m in captured_metrics if m.type == MetricType.HISTOGRAM]
        image_metrics = [m for m in captured_metrics if m.type == MetricType.IMAGE]
        text_metrics = [m for m in captured_metrics if m.type == MetricType.TEXT]
        figure_metrics = [m for m in captured_metrics if m.type == MetricType.FIGURE]
        hparam_metrics = [m for m in captured_metrics if m.type == MetricType.HPARAMS]

        assert len(scalar_metrics) >= 20  # 4 metrics x 5 epochs
        assert len(histogram_metrics) == 10  # 2 histograms x 5 epochs
        assert len(image_metrics) == 3  # Every 2 epochs: 0, 2, 4
        assert len(text_metrics) == 1  # Training summary
        assert len(figure_metrics) == 1  # Confusion matrix
        assert len(hparam_metrics) == 1  # Hyperparameters

        # Verify metadata is properly preserved
        sample_histogram = histogram_metrics[0]
        assert "layer_type" in sample_histogram.metadata
        assert "shape" in sample_histogram.metadata

        sample_image = image_metrics[0]
        assert sample_image.metadata["dataformats"] == "NCHW"
        assert sample_image.metadata["num_samples"] == 4

        sample_text = text_metrics[0]
        assert sample_text.metadata["format"] == "markdown"

        sample_hparams = hparam_metrics[0]
        assert "hparams" in sample_hparams.value
        assert sample_hparams.value["hparams"]["model"] == "ResNet50"

        print("✅ Complete ML workflow test passed!")
        print(f"   📊 Captured {len(captured_metrics)} total metrics")
        print(f"   📈 {len(scalar_metrics)} scalar metrics")
        print(f"   📊 {len(histogram_metrics)} histogram metrics")
        print(f"   🖼️ {len(image_metrics)} image metrics")
        print(f"   📝 {len(text_metrics)} text metrics")
        print(f"   📋 {len(figure_metrics)} figure metrics")
        print(f"   ⚙️ {len(hparam_metrics)} hyperparameter dashboards")

    def test_metric_routing_and_backend_compatibility(self):
        """Test that enhanced metrics route properly through the orchestrator"""
        # Start the experiment to activate the orchestrator
        self.experiment.start()

        try:
            # Log various metric types
            test_metrics = [
                ("scalar_test", 0.95, MetricType.SCALAR, {}),
                ("histogram_test", np.random.normal(0, 1, 100), MetricType.HISTOGRAM, {"bins": 30}),
                ("image_test", np.random.rand(3, 64, 64), MetricType.IMAGE, {"format": "CHW"}),
                ("text_test", "Integration test message", MetricType.TEXT, {}),
            ]

            for name, value, metric_type, metadata in test_metrics:
                if metric_type == MetricType.SCALAR:
                    self.framework.log_metric(name, value, 1)
                else:
                    self.framework.log_enhanced_metric(name, value, metric_type, 1, metadata)

            # Give the orchestrator time to process
            import time

            time.sleep(0.1)

            # Check orchestrator stats
            stats = self.experiment._orchestrator.get_stats()
            assert stats["running"]
            assert stats["metrics_processed"] >= len(test_metrics)

        finally:
            self.experiment.stop()

    def test_backward_compatibility_with_enhanced_features(self):
        """Test that existing code continues to work with enhanced features enabled"""
        # Capture all metrics
        captured_metrics = []
        original_emit = self.experiment.emit_metric

        def capture_emit(metric):
            captured_metrics.append(metric)
            return original_emit(metric)

        self.experiment.emit_metric = capture_emit

        # Use the framework exactly as before (existing API)
        self.framework.log_metric("accuracy", 0.95, 10)
        self.framework.log_metric("loss", 0.05, 10)

        # These should still work exactly as before
        accuracy_metrics = [m for m in captured_metrics if m.name == "accuracy"]
        loss_metrics = [m for m in captured_metrics if m.name == "loss"]

        assert len(accuracy_metrics) == 1
        assert len(loss_metrics) == 1

        assert accuracy_metrics[0].value == 0.95
        assert accuracy_metrics[0].iteration == 10
        assert loss_metrics[0].value == 0.05
        assert loss_metrics[0].iteration == 10

        # Verify backward compatibility - should be SCALAR type automatically
        assert all(m.type == MetricType.SCALAR for m in [accuracy_metrics[0], loss_metrics[0]])
