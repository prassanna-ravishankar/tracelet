# Git Collector

::: tracelet.collectors.git.GitCollector
options:
show_source: true
show_bases: true
merge_init_into_class: true
heading_level: 2

## Overview

The Git Collector automatically captures git repository information for experiment reproducibility and version tracking.

## Basic Usage

```python
import tracelet

# Git information is collected automatically when starting experiments
exp = tracelet.start_logging(
    exp_name="git_tracking_demo",
    project="version_control",
    backend="mlflow"
)

# Git info is collected and logged automatically
# No manual intervention needed
```

## Manual Git Collection

```python
from tracelet.collectors.git import GitCollector

# Create git collector
git_collector = GitCollector()

# Initialize and collect git information
git_collector.initialize()
git_info = git_collector.collect()

print("Git Information:")
for key, value in git_info.items():
    print(f"  {key}: {value}")
```

## Collected Information

The Git Collector captures the following repository information:

### Repository State

- **commit_hash**: Current commit SHA
- **branch**: Current branch name
- **remote_url**: Remote repository URL
- **is_dirty**: Whether working directory has uncommitted changes

### Commit Details

- **commit_message**: Latest commit message
- **commit_author**: Commit author name and email
- **commit_date**: Commit timestamp

### Working Directory Status

- **uncommitted_files**: List of modified files (if any)
- **untracked_files**: List of untracked files (if any)

## Configuration Options

### Custom Repository Path

```python
from tracelet.collectors.git import GitCollector

# Specify custom repository path
git_collector = GitCollector(repo_path="/path/to/custom/repo")
git_collector.initialize()
git_info = git_collector.collect()
```

### Integration with Settings

```python
from tracelet.settings import TraceletSettings

# Configure git tracking behavior
settings = TraceletSettings(
    project="git_configured",
    backend=["mlflow"],
    # Git-specific settings would be added here if supported
)

tracelet.start_logging(exp_name="git_exp", settings=settings)
```

## Practical Examples

### Experiment Reproducibility

```python
import tracelet
import json

# Start experiment with automatic git tracking
exp = tracelet.start_logging(
    exp_name="reproducible_experiment",
    project="research",
    backend="mlflow"
)

# Get git information for logging
from tracelet.collectors.git import GitCollector
git_collector = GitCollector()
git_collector.initialize()
git_info = git_collector.collect()

# Log git info as parameters for reproducibility
exp.log_params({
    "git_commit": git_info.get("commit_hash"),
    "git_branch": git_info.get("branch"),
    "git_is_dirty": git_info.get("is_dirty", False)
})

# Save detailed git info as artifact
with open("git_info.json", "w") as f:
    json.dump(git_info, f, indent=2, default=str)

exp.log_artifact("git_info.json", "metadata/git_info.json")

# Your training code here...
tracelet.stop_logging()
```

### Pre-commit Validation

```python
from tracelet.collectors.git import GitCollector
import sys

def validate_git_state():
    """Validate git state before starting experiment."""
    git_collector = GitCollector()
    git_collector.initialize()
    git_info = git_collector.collect()

    # Check for uncommitted changes
    if git_info.get("is_dirty", False):
        print("Warning: Working directory has uncommitted changes!")
        print(f"Modified files: {git_info.get('uncommitted_files', [])}")

        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Experiment cancelled.")
            sys.exit(1)

    # Check if on main/master branch
    current_branch = git_info.get("branch", "")
    if current_branch in ["main", "master"]:
        print(f"Warning: Running experiment on {current_branch} branch!")
        response = input("Are you sure? (y/N): ")
        if response.lower() != 'y':
            print("Experiment cancelled.")
            sys.exit(1)

    return git_info

# Usage
git_info = validate_git_state()
print(f"Starting experiment on branch: {git_info['branch']}")
print(f"Commit: {git_info['commit_hash'][:8]}")

tracelet.start_logging(
    exp_name="validated_experiment",
    project="safe_experiments",
    backend="mlflow"
)
```

### Multi-Repository Projects

```python
from tracelet.collectors.git import GitCollector
import os

def collect_multi_repo_info(repo_paths):
    """Collect git info from multiple repositories."""
    multi_repo_info = {}

    for name, path in repo_paths.items():
        if os.path.exists(path):
            git_collector = GitCollector(repo_path=path)
            git_collector.initialize()
            repo_info = git_collector.collect()
            multi_repo_info[name] = repo_info
        else:
            multi_repo_info[name] = {"error": "Repository not found"}

    return multi_repo_info

# Example usage for projects with multiple repositories
repo_paths = {
    "main_code": ".",
    "data_processing": "../data-pipeline",
    "model_configs": "../model-configs"
}

multi_git_info = collect_multi_repo_info(repo_paths)

# Start experiment
exp = tracelet.start_logging(
    exp_name="multi_repo_experiment",
    project="complex_project",
    backend="mlflow"
)

# Log git info for each repository
for repo_name, git_info in multi_git_info.items():
    if "error" not in git_info:
        exp.log_params({
            f"{repo_name}_commit": git_info.get("commit_hash"),
            f"{repo_name}_branch": git_info.get("branch"),
            f"{repo_name}_is_dirty": git_info.get("is_dirty", False)
        })
```

## Error Handling

### Repository Detection

```python
from tracelet.collectors.git import GitCollector
import logging

def safe_git_collection(repo_path=None):
    """Safely collect git information with error handling."""
    try:
        git_collector = GitCollector(repo_path=repo_path)
        git_collector.initialize()
        return git_collector.collect()

    except Exception as e:
        logging.warning(f"Failed to collect git information: {e}")
        return {
            "error": str(e),
            "git_available": False,
            "fallback_info": {
                "timestamp": str(datetime.now()),
                "hostname": os.uname().nodename if hasattr(os, 'uname') else "unknown"
            }
        }

# Usage
git_info = safe_git_collection()

if git_info.get("git_available", True):
    print(f"Git commit: {git_info.get('commit_hash')}")
else:
    print(f"Git not available: {git_info.get('error')}")
```

### Non-Git Directories

```python
import os
from tracelet.collectors.git import GitCollector

def check_git_repository(path="."):
    """Check if directory is a git repository."""
    git_dir = os.path.join(path, ".git")
    return os.path.exists(git_dir)

# Check before initializing git collector
if check_git_repository():
    git_collector = GitCollector()
    git_collector.initialize()
    git_info = git_collector.collect()
    print("Git repository detected")
else:
    print("No git repository found")
    git_info = {"git_available": False}
```

## Integration Patterns

### Automatic Git Tagging

```python
import subprocess
from tracelet.collectors.git import GitCollector
import tracelet

def create_experiment_tag(exp_name):
    """Create git tag for experiment."""
    tag_name = f"experiment-{exp_name}-{int(time.time())}"

    try:
        subprocess.run(["git", "tag", tag_name], check=True)
        return tag_name
    except subprocess.CalledProcessError as e:
        print(f"Failed to create git tag: {e}")
        return None

# Usage
exp_name = "important_experiment"

# Create git tag before starting experiment
tag_name = create_experiment_tag(exp_name)

exp = tracelet.start_logging(
    exp_name=exp_name,
    project="tagged_experiments",
    backend="mlflow"
)

if tag_name:
    exp.log_params({"git_tag": tag_name})
```

### CI/CD Integration

```python
import os
from tracelet.collectors.git import GitCollector

def get_ci_git_info():
    """Get git information in CI/CD environment."""
    git_info = {}

    # GitHub Actions
    if "GITHUB_ACTIONS" in os.environ:
        git_info.update({
            "ci_provider": "github_actions",
            "commit_hash": os.environ.get("GITHUB_SHA"),
            "branch": os.environ.get("GITHUB_REF_NAME"),
            "repository": os.environ.get("GITHUB_REPOSITORY"),
            "actor": os.environ.get("GITHUB_ACTOR")
        })

    # GitLab CI
    elif "GITLAB_CI" in os.environ:
        git_info.update({
            "ci_provider": "gitlab_ci",
            "commit_hash": os.environ.get("CI_COMMIT_SHA"),
            "branch": os.environ.get("CI_COMMIT_REF_NAME"),
            "repository": os.environ.get("CI_PROJECT_PATH"),
            "pipeline_id": os.environ.get("CI_PIPELINE_ID")
        })

    # Fall back to git collector
    else:
        git_collector = GitCollector()
        git_collector.initialize()
        git_info = git_collector.collect()
        git_info["ci_provider"] = "local"

    return git_info

# Usage in CI/CD pipeline
git_info = get_ci_git_info()

exp = tracelet.start_logging(
    exp_name="ci_experiment",
    project="automated_training",
    backend="mlflow"
)

exp.log_params({
    "ci_provider": git_info.get("ci_provider"),
    "commit_hash": git_info.get("commit_hash"),
    "branch": git_info.get("branch")
})
```

## Best Practices

### Repository Hygiene

```python
from tracelet.collectors.git import GitCollector

def validate_repository_state():
    """Validate repository state before experiment."""
    git_collector = GitCollector()
    git_collector.initialize()
    git_info = git_collector.collect()

    issues = []

    # Check for uncommitted changes
    if git_info.get("is_dirty", False):
        issues.append("Uncommitted changes detected")

    # Check for untracked files
    untracked = git_info.get("untracked_files", [])
    if untracked:
        issues.append(f"Untracked files: {len(untracked)}")

    # Check branch name
    branch = git_info.get("branch", "")
    if not branch or branch == "HEAD":
        issues.append("Detached HEAD state")

    return issues, git_info

# Usage
issues, git_info = validate_repository_state()

if issues:
    print("Repository state issues:")
    for issue in issues:
        print(f"  - {issue}")

    # Log issues as experiment metadata
    exp = tracelet.start_logging(
        exp_name="experiment_with_issues",
        project="debug",
        backend="mlflow"
    )

    exp.log_params({
        "git_issues": ", ".join(issues),
        "git_validation": "failed"
    })
else:
    print("Repository state is clean")
```

### Experiment Lineage

```python
import json
from tracelet.collectors.git import GitCollector

def track_experiment_lineage(parent_commit=None):
    """Track experiment lineage through git history."""
    git_collector = GitCollector()
    git_collector.initialize()
    git_info = git_collector.collect()

    lineage_info = {
        "current_commit": git_info.get("commit_hash"),
        "current_branch": git_info.get("branch"),
        "parent_commit": parent_commit,
        "timestamp": str(datetime.now())
    }

    # Save lineage information
    with open("experiment_lineage.json", "w") as f:
        json.dump(lineage_info, f, indent=2)

    return lineage_info

# Usage
lineage = track_experiment_lineage(parent_commit="abc123def456")

exp = tracelet.start_logging(
    exp_name="lineage_tracked",
    project="experiment_lineage",
    backend="mlflow"
)

exp.log_params(lineage)
exp.log_artifact("experiment_lineage.json", "metadata/lineage.json")
```
