"""AIM backend plugin for Tracelet."""

# Import the actual implementation from backends
from tracelet.backends.aim import AimBackend

# Re-export the class so plugin discovery can find it
__all__ = ["AimBackend"]
