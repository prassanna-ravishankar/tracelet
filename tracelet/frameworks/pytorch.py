from typing import Any, Dict, Optional
import functools
import importlib
from ..core.interfaces import FrameworkInterface

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
            importlib.import_module('torch.utils.tensorboard')
            return True
        except ImportError:
            return False
        
    def initialize(self, experiment: 'Experiment'):
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
            from torch.utils.tensorboard import SummaryWriter
            self._original_add_scalar = SummaryWriter.add_scalar
            self._original_add_scalars = SummaryWriter.add_scalars
            
            @functools.wraps(SummaryWriter.add_scalar)
            def wrapped_add_scalar(
                writer_self,
                tag: str,
                scalar_value: float,
                global_step: Optional[int] = None,
                *args,
                **kwargs
            ):
                # Call original method
                result = self._original_add_scalar(
                    writer_self, tag, scalar_value, global_step, *args, **kwargs
                )
                # Log to our experiment
                self.log_metric(tag, scalar_value, global_step)
                return result
                
            @functools.wraps(SummaryWriter.add_scalars)
            def wrapped_add_scalars(
                writer_self,
                main_tag: str,
                tag_scalar_dict: Dict[str, float],
                global_step: Optional[int] = None,
                *args,
                **kwargs
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
            from torch.utils.tensorboard import SummaryWriter
            SummaryWriter.add_scalar = self._original_add_scalar
            SummaryWriter.add_scalars = self._original_add_scalars
            self._original_add_scalar = None
            self._original_add_scalars = None