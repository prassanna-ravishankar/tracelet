"""
Basic E2E workflow tests for Tracelet

These tests validate that Tracelet works correctly with all backends
using realistic PyTorch training scenarios.
"""

import pytest

from .framework import e2e_framework

# Test fixtures for different backend/workflow combinations


class TestBasicWorkflows:
    """Test basic training workflows across all backends."""

    def test_pytorch_with_mlflow(self):
        """Test PyTorch workflow with MLflow backend."""
        if "mlflow" not in e2e_framework.get_available_backends():
            pytest.skip("MLflow backend not available")

        if "simple_pytorch" not in e2e_framework.get_available_workflows():
            pytest.skip("PyTorch workflow not available")

        with e2e_framework.backend_environment("mlflow") as backend_config:
            results = e2e_framework.run_workflow("simple_pytorch", backend_config)

            assert results["success"], f"Workflow failed: {results.get('error', 'Unknown error')}"
            assert results["epochs_completed"] == 10
            assert len(results["losses"]) == 10
            assert len(results["accuracies"]) == 10
            assert results["execution_time"] > 0

    def test_pytorch_with_clearml(self):
        """Test PyTorch workflow with ClearML backend."""
        if "clearml" not in e2e_framework.get_available_backends():
            pytest.skip("ClearML backend not available")

        if "simple_pytorch" not in e2e_framework.get_available_workflows():
            pytest.skip("PyTorch workflow not available")

        with e2e_framework.backend_environment("clearml") as backend_config:
            results = e2e_framework.run_workflow("simple_pytorch", backend_config)

            # ClearML might fail in CI environments, so we're more lenient
            if "error" in results and "clearml" in results["error"].lower():
                pytest.skip("ClearML requires server connection")

            assert results["success"], f"Workflow failed: {results.get('error', 'Unknown error')}"
            assert results["epochs_completed"] == 10

    def test_pytorch_with_wandb(self):
        """Test PyTorch workflow with W&B backend."""
        if "wandb" not in e2e_framework.get_available_backends():
            pytest.skip("W&B backend not available")

        if "simple_pytorch" not in e2e_framework.get_available_workflows():
            pytest.skip("PyTorch workflow not available")

        with e2e_framework.backend_environment("wandb") as backend_config:
            results = e2e_framework.run_workflow("simple_pytorch", backend_config)

            assert results["success"], f"Workflow failed: {results.get('error', 'Unknown error')}"
            assert results["epochs_completed"] == 10
            assert len(results["losses"]) == 10
            assert len(results["accuracies"]) == 10

    def test_lightning_with_mlflow(self):
        """Test Lightning workflow with MLflow backend."""
        if "mlflow" not in e2e_framework.get_available_backends():
            pytest.skip("MLflow backend not available")

        if "lightning" not in e2e_framework.get_available_workflows():
            pytest.skip("Lightning workflow not available")

        with e2e_framework.backend_environment("mlflow") as backend_config:
            results = e2e_framework.run_workflow("lightning", backend_config)

            assert results["success"], f"Workflow failed: {results.get('error', 'Unknown error')}"
            assert results["epochs_completed"] == 10
            assert results["training_completed"] is True

    def test_lightning_with_clearml(self):
        """Test Lightning workflow with ClearML backend."""
        if "clearml" not in e2e_framework.get_available_backends():
            pytest.skip("ClearML backend not available")

        if "lightning" not in e2e_framework.get_available_workflows():
            pytest.skip("Lightning workflow not available")

        with e2e_framework.backend_environment("clearml") as backend_config:
            results = e2e_framework.run_workflow("lightning", backend_config)

            # ClearML might fail in CI environments
            if "error" in results and "clearml" in results["error"].lower():
                pytest.skip("ClearML requires server connection")

            assert results["success"], f"Workflow failed: {results.get('error', 'Unknown error')}"
            assert results["epochs_completed"] == 10

    def test_lightning_with_wandb(self):
        """Test Lightning workflow with W&B backend."""
        if "wandb" not in e2e_framework.get_available_backends():
            pytest.skip("W&B backend not available")

        if "lightning" not in e2e_framework.get_available_workflows():
            pytest.skip("Lightning workflow not available")

        with e2e_framework.backend_environment("wandb") as backend_config:
            results = e2e_framework.run_workflow("lightning", backend_config)

            assert results["success"], f"Workflow failed: {results.get('error', 'Unknown error')}"
            assert results["epochs_completed"] == 10
            assert results["training_completed"] is True

    def test_pytorch_with_aim(self):
        """Test PyTorch workflow with AIM backend."""
        if "aim" not in e2e_framework.get_available_backends():
            pytest.skip("AIM backend not available")

        if "simple_pytorch" not in e2e_framework.get_available_workflows():
            pytest.skip("PyTorch workflow not available")

        with e2e_framework.backend_environment("aim") as backend_config:
            results = e2e_framework.run_workflow("simple_pytorch", backend_config)

            assert results["success"], f"Workflow failed: {results.get('error', 'Unknown error')}"
            assert results["epochs_completed"] == 10
            assert len(results["losses"]) == 10
            assert len(results["accuracies"]) == 10
            assert results["execution_time"] > 0

    def test_lightning_with_aim(self):
        """Test Lightning workflow with AIM backend."""
        if "aim" not in e2e_framework.get_available_backends():
            pytest.skip("AIM backend not available")

        if "lightning" not in e2e_framework.get_available_workflows():
            pytest.skip("Lightning workflow not available")

        with e2e_framework.backend_environment("aim") as backend_config:
            results = e2e_framework.run_workflow("lightning", backend_config)

            assert results["success"], f"Workflow failed: {results.get('error', 'Unknown error')}"
            assert results["epochs_completed"] == 10
            assert results["training_completed"] is True


class TestCrossBackendCompatibility:
    """Test that the same workflows produce consistent results across backends."""

    @pytest.mark.slow
    def test_pytorch_consistency_across_backends(self):
        """Test that PyTorch workflow produces consistent results across backends."""
        available_backends = e2e_framework.get_available_backends()

        if "simple_pytorch" not in e2e_framework.get_available_workflows():
            pytest.skip("PyTorch workflow not available")

        if len(available_backends) < 2:
            pytest.skip("Need at least 2 backends for consistency testing")

        results = {}

        for backend_name in available_backends:
            try:
                with e2e_framework.backend_environment(backend_name) as backend_config:
                    result = e2e_framework.run_workflow("simple_pytorch", backend_config)

                    if result["success"]:
                        results[backend_name] = result
            except Exception as e:
                # Some backends might not be available in CI
                print(f"Backend {backend_name} failed: {e}")
                continue

        # We need at least 2 successful results to compare
        if len(results) < 2:
            pytest.skip("Need at least 2 successful backend runs for comparison")

        # Compare results across backends
        first_result = next(iter(results.values()))

        for backend_name, result in results.items():
            assert (
                result["epochs_completed"] == first_result["epochs_completed"]
            ), f"Epoch count mismatch: {backend_name} vs first backend"

            assert len(result["losses"]) == len(
                first_result["losses"]
            ), f"Losses count mismatch: {backend_name} vs first backend"

            assert len(result["accuracies"]) == len(
                first_result["accuracies"]
            ), f"Accuracies count mismatch: {backend_name} vs first backend"

    @pytest.mark.slow
    def test_lightning_consistency_across_backends(self):
        """Test that Lightning workflow produces consistent results across backends."""
        available_backends = e2e_framework.get_available_backends()

        if "lightning" not in e2e_framework.get_available_workflows():
            pytest.skip("Lightning workflow not available")

        if len(available_backends) < 2:
            pytest.skip("Need at least 2 backends for consistency testing")

        results = {}

        for backend_name in available_backends:
            try:
                with e2e_framework.backend_environment(backend_name) as backend_config:
                    result = e2e_framework.run_workflow("lightning", backend_config)

                    if result["success"]:
                        results[backend_name] = result
            except Exception as e:
                # Some backends might not be available in CI
                print(f"Backend {backend_name} failed: {e}")
                continue

        # We need at least 2 successful results to compare
        if len(results) < 2:
            pytest.skip("Need at least 2 successful backend runs for comparison")

        # Compare results across backends
        first_result = next(iter(results.values()))

        for backend_name, result in results.items():
            assert (
                result["epochs_completed"] == first_result["epochs_completed"]
            ), f"Epoch count mismatch: {backend_name} vs first backend"

            assert (
                result["training_completed"] == first_result["training_completed"]
            ), f"Training completion mismatch: {backend_name} vs first backend"


class TestComprehensiveE2E:
    """Comprehensive end-to-end test suite."""

    @pytest.mark.slow
    def test_comprehensive_all_backends_workflows(self):
        """Run comprehensive tests across all available backends and workflows."""
        results = e2e_framework.run_comprehensive_test()

        # Print summary for debugging
        summary = results["test_summary"]
        print("\nE2E Test Summary:")
        print(f"Total tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Execution time: {summary['execution_time']:.2f}s")

        # Print detailed results
        for backend_name, backend_results in results["backend_results"].items():
            print(f"\n{backend_name.upper()} Backend:")
            for workflow_name, workflow_result in backend_results.items():
                status = "✓" if workflow_result["success"] else "✗"
                time_taken = workflow_result["execution_time"]
                print(f"  {status} {workflow_name}: {time_taken:.2f}s")
                if not workflow_result["success"] and "error" in workflow_result:
                    print(f"    Error: {workflow_result['error']}")

        # At least some tests should pass (we expect at least MLflow to work)
        assert summary["passed_tests"] > 0, "No tests passed - this suggests a fundamental issue"

        # More than half should pass in a healthy system
        if summary["total_tests"] > 0:
            pass_rate = summary["passed_tests"] / summary["total_tests"]
            print(f"\nOverall pass rate: {pass_rate:.1%}")

            # We expect at least 50% pass rate (some backends might not be available in CI)
            assert pass_rate >= 0.5, f"Pass rate too low: {pass_rate:.1%}"


class TestPerformanceBenchmark:
    """Basic performance benchmarking tests."""

    @pytest.mark.slow
    def test_mlflow_performance(self):
        """Benchmark MLflow backend performance."""
        if "mlflow" not in e2e_framework.get_available_backends():
            pytest.skip("MLflow backend not available")

        if "simple_pytorch" not in e2e_framework.get_available_workflows():
            pytest.skip("PyTorch workflow not available")

        with e2e_framework.backend_environment("mlflow") as backend_config:
            results = e2e_framework.run_workflow("simple_pytorch", backend_config)

            assert results["success"], f"Workflow failed: {results.get('error', 'Unknown error')}"

            # Performance assertions (these are reasonable for simple workflows)
            assert results["execution_time"] < 60, f"Execution too slow: {results['execution_time']:.2f}s"

            print(f"MLflow performance: {results['execution_time']:.2f}s for 10 epochs")

    @pytest.mark.slow
    def test_wandb_performance(self):
        """Benchmark W&B backend performance."""
        if "wandb" not in e2e_framework.get_available_backends():
            pytest.skip("W&B backend not available")

        if "simple_pytorch" not in e2e_framework.get_available_workflows():
            pytest.skip("PyTorch workflow not available")

        with e2e_framework.backend_environment("wandb") as backend_config:
            results = e2e_framework.run_workflow("simple_pytorch", backend_config)

            assert results["success"], f"Workflow failed: {results.get('error', 'Unknown error')}"

            # Performance assertions
            assert results["execution_time"] < 60, f"Execution too slow: {results['execution_time']:.2f}s"

            print(f"W&B performance: {results['execution_time']:.2f}s for 10 epochs")

    @pytest.mark.slow
    def test_clearml_performance(self):
        """Benchmark ClearML backend performance."""
        if "clearml" not in e2e_framework.get_available_backends():
            pytest.skip("ClearML backend not available")

        if "simple_pytorch" not in e2e_framework.get_available_workflows():
            pytest.skip("PyTorch workflow not available")

        with e2e_framework.backend_environment("clearml") as backend_config:
            results = e2e_framework.run_workflow("simple_pytorch", backend_config)

            # ClearML might fail in CI environments
            if "error" in results and "clearml" in results["error"].lower():
                pytest.skip("ClearML requires server connection")

            assert results["success"], f"Workflow failed: {results.get('error', 'Unknown error')}"

            # Performance assertions
            assert results["execution_time"] < 60, f"Execution too slow: {results['execution_time']:.2f}s"

            print(f"ClearML performance: {results['execution_time']:.2f}s for 10 epochs")
