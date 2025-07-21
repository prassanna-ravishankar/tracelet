from __future__ import annotations

import functools
import importlib
from typing import Any, TYPE_CHECKING

from ..core.interfaces import FrameworkInterface

if TYPE_CHECKING:
    from ..core.experiment import Experiment


class PyTorchFramework(FrameworkInterface):
    """PyTorch framework integration that patches tensorboard for metric tracking"""

    def __init__(self, patch_tensorboard: bool = True):
        self._experiment = None
        self._original_add_scalar = None
        self._original_add_scalars = None
        self._patch_tensorboard = patch_tensorboard
        self._tensorboard_available = self._check_tensorboard()

    @staticmethod
    def _check_tensorboard():
        """Check if tensorboard is available"""
        try:
            importlib.import_module("torch.utils.tensorboard")
        except ImportError:
            return False
        else:
            return True

    def initialize(self, experiment: Experiment):
        self._experiment = experiment
        if self._patch_tensorboard and self._tensorboard_available:
            self._patch_tensorboard_writer()

    def start_tracking(self):
        pass  # Nothing specific needed for PyTorch

    def stop_tracking(self):
        if self._patch_tensorboard and self._tensorboard_available:
            self._unpatch_tensorboard_writer()

    def log_metric(self, name: str, value: Any, iteration: int):
        if self._experiment:
            self._experiment.log_metric(name, value, iteration)

    def _patch_tensorboard_writer(self):
        """Patch tensorboard's SummaryWriter to capture metrics"""
        if not self._original_add_scalar:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                # TensorBoard not available, update flag and return
                self._tensorboard_available = False
                return

            self._original_add_scalar = SummaryWriter.add_scalar
            self._original_add_scalars = SummaryWriter.add_scalars

            @functools.wraps(SummaryWriter.add_scalar)
            def wrapped_add_scalar(
                writer_self, tag: str, scalar_value: float, global_step: int | None = None, *args, **kwargs
            ):
                # Call original method
                result = self._original_add_scalar(writer_self, tag, scalar_value, global_step, *args, **kwargs)
                # Log to our experiment
                self.log_metric(tag, scalar_value, global_step)
                return result

            @functools.wraps(SummaryWriter.add_scalars)
            def wrapped_add_scalars(
                writer_self,
                main_tag: str,
                tag_scalar_dict: dict[str, float],
                global_step: int | None = None,
                *args,
                **kwargs,
            ):
                # Call original method
                result = self._original_add_scalars(
                    writer_self, main_tag, tag_scalar_dict, global_step, *args, **kwargs
                )
                # Log each metric
                for tag, scalar in tag_scalar_dict.items():
                    metric_name = f"{main_tag}/{tag}"
                    self.log_metric(metric_name, scalar, global_step)
                return result

            SummaryWriter.add_scalar = wrapped_add_scalar
            SummaryWriter.add_scalars = wrapped_add_scalars

    def _unpatch_tensorboard_writer(self):
        """Restore original tensorboard methods"""
        if self._original_add_scalar and self._tensorboard_available:
            try:
                from torch.utils.tensorboard import SummaryWriter

                SummaryWriter.add_scalar = self._original_add_scalar
                SummaryWriter.add_scalars = self._original_add_scalars
                self._original_add_scalar = None
                self._original_add_scalars = None
            except ImportError:
                # TensorBoard not available, just clear references
                self._original_add_scalar = None
                self._original_add_scalars = None
