# Claude Code Instructions

## UV Package Manager

This project uses `uv` for dependency management. Key commands:

- `uv sync` - Sync dependencies from pyproject.toml
- `uv run <command>` - Run commands in the project environment
- `uv pip install -e .` - Install project in editable mode
- `uv run pytest` - Run tests in the environment

Always use `uv run` when executing Python code to ensure proper environment.

## Task Master AI Instructions

**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md
