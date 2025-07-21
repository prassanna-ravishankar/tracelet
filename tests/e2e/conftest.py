"""
Pytest configuration for E2E tests.
"""

import sys
from pathlib import Path

import pytest

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import after path modification to ensure modules can be found
from .framework import e2e_framework  # noqa: E402


def pytest_configure(config):
    """Configure pytest for E2E tests."""
    # Register custom markers
    config.addinivalue_line("markers", "e2e_basic: Basic end-to-end integration tests")
    config.addinivalue_line("markers", "e2e_advanced: Advanced end-to-end integration tests")
    config.addinivalue_line("markers", "e2e_performance: Performance benchmarking tests")
    config.addinivalue_line("markers", "e2e_comprehensive: Comprehensive cross-backend tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test file and name
        if "test_basic_workflows" in str(item.fspath):
            item.add_marker(pytest.mark.e2e_basic)

        if "test_advanced_workflows" in str(item.fspath):
            item.add_marker(pytest.mark.e2e_advanced)

        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.e2e_performance)

        if "comprehensive" in item.name.lower():
            item.add_marker(pytest.mark.e2e_comprehensive)


@pytest.fixture(scope="session", autouse=True)
def register_advanced_workflows():
    """Register advanced workflows for testing."""
    # Import advanced workflows dynamically to avoid import errors
    try:
        from .test_advanced_workflows import ComputerVisionWorkflow, TimeSeriesWorkflow

        e2e_framework.workflows["computer_vision"] = ComputerVisionWorkflow
        e2e_framework.workflows["time_series"] = TimeSeriesWorkflow
    except ImportError:
        # Advanced workflows not available, skip registration
        pass


@pytest.fixture(scope="session")
def available_backends():
    """Get list of available backends for testing."""
    return e2e_framework.get_available_backends()


@pytest.fixture(scope="session")
def available_workflows():
    """Get list of available workflows for testing."""
    return e2e_framework.get_available_workflows()


@pytest.fixture
def test_output_dir(tmp_path):
    """Provide a temporary directory for test outputs."""
    return tmp_path


# Skip entire test files if no backends are available
def pytest_runtest_setup(item):
    """Setup for each test run."""
    available_backends = e2e_framework.get_available_backends()

    if not available_backends and "e2e" in str(item.fspath):
        pytest.skip("No E2E backends available for testing")


# Custom pytest options
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--e2e-backends",
        action="store",
        default=None,
        help="Comma-separated list of backends to test (e.g., mlflow,wandb)",
    )
    parser.addoption("--e2e-workflows", action="store", default=None, help="Comma-separated list of workflows to test")
    parser.addoption("--e2e-skip-slow", action="store_true", default=False, help="Skip slow E2E tests")


@pytest.fixture(scope="session")
def e2e_test_config(request):
    """Configuration for E2E tests from command line options."""
    config = {}

    backends_option = request.config.getoption("--e2e-backends")
    if backends_option:
        config["backends"] = [b.strip() for b in backends_option.split(",")]

    workflows_option = request.config.getoption("--e2e-workflows")
    if workflows_option:
        config["workflows"] = [w.strip() for w in workflows_option.split(",")]

    config["skip_slow"] = request.config.getoption("--e2e-skip-slow")

    return config


def pytest_report_header(config):
    """Add custom header to pytest report."""
    available_backends = e2e_framework.get_available_backends()
    available_workflows = e2e_framework.get_available_workflows()

    lines = [
        "Tracelet E2E Tests",
        f"Available backends: {', '.join(available_backends) if available_backends else 'None'}",
        f"Available workflows: {', '.join(available_workflows) if available_workflows else 'None'}",
    ]

    return lines
