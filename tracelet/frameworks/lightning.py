from typing import Any, Dict, Optional
import importlib
from ..core.interfaces import FrameworkInterface

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
            importlib.import_module('pytorch_lightning')
            return True
        except ImportError:
            return False
        
    def initialize(self, experiment: 'Experiment'):
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
            import pytorch_lightning as pl
            
            # Store original method
            self._original_log_metrics = pl.Trainer.log_metrics
            
            def wrapped_log_metrics(trainer_self, metrics: Dict[str, float], step: Optional[int] = None):
                # Call original logging method
                self._original_log_metrics(trainer_self, metrics, step)
                
                # Get the current epoch from trainer
                current_epoch = trainer_self.current_epoch if hasattr(trainer_self, 'current_epoch') else 0
                
                # Log each metric to our experiment
                for name, value in metrics.items():
                    # Lightning already prefixes with training/validation
                    self.log_metric(name, value, step or current_epoch)
                    
            # Apply the patch
            pl.Trainer.log_metrics = wrapped_log_metrics
            
    def _unpatch_lightning_logging(self):
        """Restore original Lightning logging methods"""
        if self._original_log_metrics and self._lightning_available:
            import pytorch_lightning as pl
            pl.Trainer.log_metrics = self._original_log_metrics
            self._original_log_metrics = None 