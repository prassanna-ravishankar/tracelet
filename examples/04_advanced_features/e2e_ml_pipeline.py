#!/usr/bin/env python3
"""
Tracelet E2E Testing Demo

This script demonstrates the E2E testing framework by running
sample workflows across available backends.
"""

import sys
from pathlib import Path

# Add parent directory to path to import tracelet
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.e2e.framework import e2e_framework
from tests.e2e.test_runner import E2ETestRunner


def demo_basic_functionality():
    """Demonstrate basic E2E framework functionality."""
    print("🚀 Tracelet E2E Testing Framework Demo")
    print("=" * 50)

    # Check what's available
    backends = e2e_framework.get_available_backends()
    workflows = e2e_framework.get_available_workflows()

    print(f"Available Backends: {backends}")
    print(f"Available Workflows: {workflows}")
    print()

    if not backends:
        print("❌ No backends available. Please install at least one of:")
        print("   - MLflow: pip install mlflow")
        print("   - W&B: pip install wandb")
        print("   - ClearML: pip install clearml")
        return

    if not workflows:
        print("❌ No workflows available. Please install PyTorch:")
        print("   - PyTorch: pip install torch")
        return

    # Run a simple test
    backend_name = backends[0]  # Use first available backend
    workflow_name = workflows[0]  # Use first available workflow

    print(f"🧪 Running demo test: {workflow_name} with {backend_name}")

    try:
        with e2e_framework.backend_environment(backend_name) as backend_config:
            print(f"✅ Backend {backend_name} setup successful")

            results = e2e_framework.run_workflow(workflow_name, backend_config)

            print("📊 Test Results:")
            print(f"   Success: {results['success']}")
            print(f"   Execution Time: {results['execution_time']:.2f}s")

            if results["success"]:
                print(f"   Epochs Completed: {results.get('epochs_completed', 'N/A')}")
                if "losses" in results:
                    print(f"   Final Loss: {results['losses'][-1]:.4f}")
                if "accuracies" in results:
                    print(f"   Final Accuracy: {results['accuracies'][-1]:.4f}")
            else:
                print(f"   Error: {results.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Demo failed: {e}")


def demo_comprehensive_testing():
    """Demonstrate comprehensive testing capabilities."""
    print("\n🔬 Comprehensive Testing Demo")
    print("=" * 40)

    # Only run this if we have multiple backends/workflows available
    backends = e2e_framework.get_available_backends()
    workflows = e2e_framework.get_available_workflows()

    if len(backends) < 1 or len(workflows) < 1:
        print("⏭️  Skipping comprehensive demo (insufficient backends/workflows)")
        return

    # Limit to 2 backends and 2 workflows max for demo
    test_backends = backends[:2]
    test_workflows = workflows[:2]

    print(f"Testing {len(test_backends)} backends x {len(test_workflows)} workflows")

    try:
        results = e2e_framework.run_comprehensive_test(backends=test_backends, workflows=test_workflows)

        summary = results["test_summary"]
        print("📈 Comprehensive Test Results:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed']}")
        print(f"   Failed: {summary['failed_tests']}")
        print(f"   Total Time: {summary['execution_time']:.2f}s")

        if summary["total_tests"] > 0:
            pass_rate = summary["passed_tests"] / summary["total_tests"] * 100
            print(f"   Pass Rate: {pass_rate:.1f}%")

    except Exception as e:
        print(f"❌ Comprehensive testing failed: {e}")


def demo_test_runner():
    """Demonstrate the advanced test runner."""
    print("\n📊 Advanced Test Runner Demo")
    print("=" * 40)

    backends = e2e_framework.get_available_backends()
    workflows = e2e_framework.get_available_workflows()

    if not backends or not workflows:
        print("⏭️  Skipping test runner demo (no backends/workflows available)")
        return

    # Create test runner
    output_dir = Path("./demo_results")
    runner = E2ETestRunner(output_dir)

    # Run limited test suite for demo
    test_backends = backends[:1]  # Only test one backend for demo
    test_workflows = workflows[:1]  # Only test one workflow for demo

    print(f"Running test runner with {test_backends} and {test_workflows}")

    try:
        _report = runner.run_full_test_suite(backends=test_backends, workflows=test_workflows, save_results=True)

        print("✅ Test runner completed successfully!")
        print(f"📁 Results saved to: {output_dir}")

    except Exception as e:
        print(f"❌ Test runner demo failed: {e}")


def main():
    """Main demo function."""
    try:
        demo_basic_functionality()
        demo_comprehensive_testing()
        demo_test_runner()

        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Run the full test suite: python -m pytest tests/e2e/")
        print("2. Run specific backends: python -m pytest tests/e2e/ --e2e-backends=mlflow")
        print("3. Run the test runner: python tests/e2e/test_runner.py")
        print("4. Check the examples in tests/e2e/test_*.py for more complex scenarios")

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed with unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
