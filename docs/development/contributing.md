# Contributing to Tracelet

We welcome contributions to Tracelet! This guide will help you get started with contributing to the project.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Development Setup

1. **Fork and Clone**

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/tracelet.git
cd tracelet
```

2. **Install Development Dependencies**

```bash
# Using uv (recommended)
uv sync --all-extras

# Or using pip
pip install -e ".[dev,all]"
```

3. **Install Pre-commit Hooks**

```bash
uv run pre-commit install
```

4. **Verify Installation**

```bash
# Run tests to ensure everything works
uv run pytest tests/unit -v

# Run linting
uv run ruff check
uv run ruff format --check
```

## Development Workflow

### Branch Strategy

- **main**: Production-ready code
- **feature/xxx**: New features
- **fix/xxx**: Bug fixes
- **docs/xxx**: Documentation updates

### Making Changes

1. **Create a Feature Branch**

```bash
git checkout -b feature/my-new-feature
```

2. **Make Your Changes**

- Write code following our style guidelines
- Add tests for new functionality
- Update documentation as needed

3. **Test Your Changes**

```bash
# Run unit tests
uv run pytest tests/unit -v

# Run integration tests (optional)
uv run pytest tests/integration -v

# Run linting
uv run ruff check
uv run ruff format
```

4. **Commit Your Changes**

```bash
git add .
git commit -m "feat: add new awesome feature"
```

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/modifications
- `refactor:` for code refactoring
- `style:` for formatting changes

5. **Push and Create PR**

```bash
git push origin feature/my-new-feature
```

Then create a Pull Request on GitHub.

## Code Style Guidelines

### Python Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check style
uv run ruff check

# Auto-fix issues
uv run ruff check --fix

# Format code
uv run ruff format
```

### Code Organization

```
tracelet/
├── __init__.py           # Main public API
├── core/                 # Core functionality
│   ├── experiment.py     # Experiment management
│   ├── orchestrator.py   # Metric routing
│   └── plugins.py        # Plugin system
├── backends/             # Backend implementations
│   ├── mlflow.py
│   ├── wandb.py
│   └── clearml.py
├── frameworks/           # Framework integrations
│   ├── pytorch.py
│   └── lightning.py
├── collectors/           # Data collectors
│   ├── git.py
│   └── system.py
└── plugins/              # Plugin implementations
    ├── mlflow_backend.py
    └── wandb_backend.py
```

### Naming Conventions

- **Classes**: PascalCase (`ExperimentTracker`)
- **Functions/Variables**: snake_case (`log_metric`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_TIMEOUT`)
- **Private**: Leading underscore (`_internal_method`)

## Testing Guidelines

### Test Structure

```
tests/
├── unit/                 # Unit tests
│   ├── core/
│   ├── backends/
│   └── frameworks/
├── integration/          # Integration tests
│   ├── test_backend_integration.py
│   └── test_e2e_workflows.py
└── e2e/                  # End-to-end tests
    ├── test_basic_workflows.py
    └── test_advanced_workflows.py
```

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch
from tracelet.core.experiment import Experiment

class TestExperiment:
    def test_log_metric(self):
        """Test basic metric logging functionality"""
        experiment = Experiment("test_exp", "test_project")

        # Test the functionality
        experiment.log_metric("accuracy", 0.95, step=100)

        # Assert expected behavior
        assert experiment.metrics["accuracy"][-1] == (0.95, 100)

    @patch('tracelet.backends.mlflow.MLflowBackend')
    def test_backend_integration(self, mock_backend):
        """Test integration with backend"""
        mock_backend.return_value.log_metric = Mock()

        experiment = Experiment("test_exp", "test_project")
        experiment.add_backend(mock_backend.return_value)
        experiment.log_metric("loss", 0.5, step=1)

        mock_backend.return_value.log_metric.assert_called_once_with("loss", 0.5, 1)
```

### Test Categories

**Unit Tests** - Fast, isolated tests:

```bash
uv run pytest tests/unit -v
```

**Integration Tests** - Test component interactions:

```bash
uv run pytest tests/integration -v
```

**E2E Tests** - Full workflow tests (slow):

```bash
uv run pytest tests/e2e -v
```

## Documentation Guidelines

### Code Documentation

Use Google-style docstrings:

```python
def log_metric(self, name: str, value: float, step: int = None) -> None:
    """Log a scalar metric to the experiment.

    Args:
        name: The name of the metric (e.g., 'accuracy', 'loss').
        value: The numeric value to log.
        step: The step/iteration number. If None, auto-incremented.

    Raises:
        ValueError: If value is not a number.

    Example:
        >>> experiment.log_metric("accuracy", 0.95, step=100)
    """
```

### API Documentation

Document all public APIs with:

- Clear description of purpose
- Parameter types and descriptions
- Return value information
- Usage examples
- Related functions/classes

### Adding New Documentation

1. **Create Markdown Files**

```bash
# Add new documentation
touch docs/guides/my-new-guide.md
```

2. **Update Navigation**
   Edit `mkdocs.yml` to include your new documentation:

```yaml
nav:
  - Guides:
      - My New Guide: guides/my-new-guide.md
```

3. **Test Documentation**

```bash
# Build and serve docs locally
uv run mkdocs serve
```

## Adding New Features

### Backend Integration

To add a new backend (e.g., Neptune):

1. **Create Backend Implementation**

```python
# tracelet/backends/neptune.py
from tracelet.core.interfaces import BackendInterface

class NeptuneBackend(BackendInterface):
    def __init__(self, config: dict):
        self.config = config
        self._setup_neptune()

    def log_metric(self, name: str, value: float, step: int):
        # Implementation here
        pass
```

2. **Create Plugin**

```python
# tracelet/plugins/neptune_backend.py
from tracelet.core.plugins import BackendPlugin, PluginMetadata, PluginType

class NeptuneBackendPlugin(BackendPlugin):
    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="neptune",
            version="1.0.0",
            type=PluginType.BACKEND,
            description="Neptune.ai experiment tracking backend"
        )

    def create_backend(self, config: dict):
        from tracelet.backends.neptune import NeptuneBackend
        return NeptuneBackend(config)
```

3. **Add Tests**

```python
# tests/unit/backends/test_neptune.py
# tests/integration/test_neptune_integration.py
```

4. **Update Documentation**

```markdown
# docs/backends/neptune.md
```

### Framework Integration

To add a new framework integration:

1. **Create Framework Module**

```python
# tracelet/frameworks/jax.py
from tracelet.core.interfaces import FrameworkInterface

class JAXFramework(FrameworkInterface):
    def initialize(self, experiment):
        # Patch JAX logging functions
        pass
```

2. **Add Plugin**
3. **Write Tests**
4. **Document Usage**

## Release Process

### Version Management

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Creating a Release

1. **Update Version**

```bash
# Update version in pyproject.toml
# Update CHANGELOG.md
```

2. **Create Release PR**

```bash
git checkout -b release/v1.2.0
git commit -m "chore: prepare release v1.2.0"
```

3. **Tag Release**

```bash
git tag v1.2.0
git push origin v1.2.0
```

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow our Code of Conduct

### Getting Help

- **Discord**: Join our [Discord server](https://discord.gg/tracelet)
- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and community discussions
- **Email**: [maintainers@tracelet.io](mailto:maintainers@tracelet.io)

### Recognition

Contributors are recognized in:

- CONTRIBUTORS.md file
- Release notes
- Documentation acknowledgments
- Social media shoutouts

## Questions?

Don't hesitate to ask questions:

- Open a [GitHub Discussion](https://github.com/prassanna-ravishankar/tracelet/discussions)
- Join our [Discord](https://discord.gg/tracelet)
- Email us at [maintainers@tracelet.io](mailto:maintainers@tracelet.io)

Thank you for contributing to Tracelet! 🚀
