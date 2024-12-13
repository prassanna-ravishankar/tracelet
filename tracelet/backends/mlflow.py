from typing import Any, Dict, Optional
import os
import mlflow
from ..core.interfaces import BackendInterface

class MLflowBackend(BackendInterface):
    """MLflow backend integration"""
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._run = None
        
    def initialize(self):
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
            
        if self.experiment_name:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
            else:
                experiment_id = experiment.experiment_id
            mlflow.set_experiment(experiment_id)
            
        self._run = mlflow.start_run()
        
    def log_metric(self, name: str, value: Any, iteration: int):
        mlflow.log_metric(name, value, step=iteration)
        
    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)
        
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        mlflow.log_artifact(local_path, artifact_path)
        
    def save_experiment(self, experiment_data: Dict[str, Any]):
        # Log experiment metadata
        mlflow.set_tags(experiment_data)
        
        # If there's git info, log it as a separate set of tags
        if "git" in experiment_data:
            git_info = experiment_data["git"]
            if isinstance(git_info, dict):
                for key, value in git_info.items():
                    if isinstance(value, (str, bool, int, float)):
                        mlflow.set_tag(f"git.{key}", str(value))
                        
        # Log system info
        if "system" in experiment_data:
            system_info = experiment_data["system"]
            if isinstance(system_info, dict):
                for key, value in system_info.items():
                    if isinstance(value, (str, bool, int, float)):
                        mlflow.set_tag(f"system.{key}", str(value))
                        
    def __enter__(self):
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._run:
            mlflow.end_run() 