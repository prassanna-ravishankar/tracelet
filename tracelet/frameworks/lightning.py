from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from ..core.interfaces import FrameworkInterface

if TYPE_CHECKING:
    from ..core.experiment import Experiment


class LightningFramework(FrameworkInterface):
    """PyTorch Lightning framework integration"""

    def __init__(self):
        self._experiment = None
        self._original_log_metrics = None
        self._lightning_available = self._check_lightning()

    @staticmethod
    def _check_lightning():
        """Check if pytorch_lightning is available"""
        try:
            importlib.import_module("pytorch_lightning")
        except ImportError:
            return False
        else:
            return True

    def initialize(self, experiment: Experiment):
        self._experiment = experiment
        if self._lightning_available:
            self._patch_lightning_logging()

    def start_tracking(self):
        pass  # Nothing specific needed for Lightning

    def stop_tracking(self):
        if self._lightning_available:
            self._unpatch_lightning_logging()

    def log_metric(self, name: str, value: Any, iteration: int):
        if self._experiment:
            self._experiment.log_metric(name, value, iteration)

    def _patch_lightning_logging(self):
        """Patch Lightning's logging system to capture metrics"""
        if not self._original_log_metrics and self._lightning_available:
            try:
                from pytorch_lightning.core.module import LightningModule
            except ImportError:
                # Lightning not available, update flag and return
                self._lightning_available = False
                return

            # Store original method - patch LightningModule.log instead of Trainer.log_metrics
            self._original_log_metrics = LightningModule.log

            def wrapped_log(module_self, name: str, value, *args, **kwargs):
                # Call original logging method
                result = self._original_log_metrics(module_self, name, value, *args, **kwargs)

                # Get the current step from trainer if available
                current_step = 0
                try:
                    if hasattr(module_self, "trainer") and module_self.trainer:
                        current_step = getattr(module_self.trainer, "global_step", 0)
                except RuntimeError:
                    # Lightning module not attached to a trainer yet
                    current_step = 0

                # Log to our experiment (only if value is a scalar)
                if self._experiment and isinstance(value, (int, float)):
                    self.log_metric(name, float(value), current_step)

                return result

            # Apply the patch
            LightningModule.log = wrapped_log

    def _unpatch_lightning_logging(self):
        """Restore original Lightning logging methods"""
        if self._original_log_metrics and self._lightning_available:
            try:
                from pytorch_lightning.core.module import LightningModule

                LightningModule.log = self._original_log_metrics
                self._original_log_metrics = None
            except ImportError:
                # Lightning not available, just clear the reference
                self._original_log_metrics = None
