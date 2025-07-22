"""
Automatic monitors for training progress and system resources.
"""

import threading
import warnings
import weakref

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import GPUtil

    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ..core.experiment import Experiment


class TrainingMonitor:
    """Monitor training loops and automatically detect metrics."""

    def __init__(self, config):
        self.config = config
        self._active_experiments: dict[str, weakref.ref] = {}
        self._monitor_threads: dict[str, threading.Thread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._lock = threading.RLock()

        # Training detection state
        self._training_state: dict[str, dict] = {}

    def start(self, experiment: Experiment) -> None:
        """Start monitoring for an experiment."""
        with self._lock:
            exp_id = experiment.id
            if exp_id in self._active_experiments:
                return  # Already monitoring

            self._active_experiments[exp_id] = weakref.ref(experiment)
            self._training_state[exp_id] = {
                "epoch": 0,
                "step": 0,
                "loss_history": [],
                "last_loss_time": None,
                "training_started": False,
            }

            # Start monitoring thread
            stop_event = threading.Event()
            self._stop_events[exp_id] = stop_event

            monitor_thread = threading.Thread(target=self._monitor_loop, args=(exp_id, stop_event), daemon=True)
            self._monitor_threads[exp_id] = monitor_thread
            monitor_thread.start()

    def stop(self, experiment_id: str) -> None:
        """Stop monitoring for an experiment."""
        with self._lock:
            if experiment_id in self._stop_events:
                self._stop_events[experiment_id].set()

            if experiment_id in self._monitor_threads:
                thread = self._monitor_threads[experiment_id]
                if thread.is_alive():
                    thread.join(timeout=1.0)  # Wait up to 1 second
                del self._monitor_threads[experiment_id]

            # Clean up state
            self._active_experiments.pop(experiment_id, None)
            self._stop_events.pop(experiment_id, None)
            self._training_state.pop(experiment_id, None)

    def cleanup(self) -> None:
        """Stop all monitoring."""
        with self._lock:
            for exp_id in list(self._active_experiments.keys()):
                self.stop(exp_id)

    def _monitor_loop(self, experiment_id: str, stop_event: threading.Event) -> None:
        """Main monitoring loop."""
        while not stop_event.wait(1.0):  # Check every second
            try:
                experiment_ref = self._active_experiments.get(experiment_id)
                if not experiment_ref:
                    break

                experiment = experiment_ref()
                if not experiment:
                    break  # Experiment was garbage collected

                self._detect_training_patterns(experiment_id, experiment)

            except Exception as e:
                warnings.warn(f"Error in training monitor for {experiment_id}: {e}", stacklevel=2)
                break

    def _detect_training_patterns(self, experiment_id: str, experiment: Experiment) -> None:
        """Detect training patterns and log derived metrics."""
        state = self._training_state.get(experiment_id, {})

        # Check if training has started by looking for recent metrics
        recent_metrics = self._get_recent_metrics(experiment)

        if recent_metrics and not state.get("training_started"):
            state["training_started"] = True
            experiment.log_metric("training_started", 1)

        # Detect epochs and steps
        self._detect_epoch_changes(experiment_id, experiment, recent_metrics)

        # Analyze loss trends
        self._analyze_loss_trends(experiment_id, experiment, recent_metrics)

    def _get_recent_metrics(self, experiment: Experiment) -> dict[str, float]:
        """Get recent metrics from the experiment (simplified)."""
        # This would need to integrate with the actual metric storage
        # For now, return empty dict as placeholder
        return {}

    def _detect_epoch_changes(self, experiment_id: str, experiment: Experiment, metrics: dict[str, float]) -> None:
        """Detect epoch changes and log epoch-level metrics."""
        # Look for common epoch indicators in metrics
        epoch_indicators = ["epoch", "train_loss", "val_loss", "accuracy"]

        for indicator in epoch_indicators:
            if indicator in metrics:
                # Simple heuristic: if we see these metrics, increment epoch
                state = self._training_state[experiment_id]
                current_epoch = state.get("epoch", 0)
                state["epoch"] = current_epoch + 1
                experiment.log_metric("detected_epoch", state["epoch"])
                break

    def _analyze_loss_trends(self, experiment_id: str, experiment: Experiment, metrics: dict[str, float]) -> None:
        """Analyze loss trends and detect convergence."""
        state = self._training_state[experiment_id]

        # Look for loss metrics
        loss_metrics = [k for k in metrics if "loss" in k.lower()]

        for loss_key in loss_metrics:
            loss_value = metrics[loss_key]
            loss_history = state.setdefault("loss_history", [])

            loss_history.append(loss_value)

            # Keep only recent history
            if len(loss_history) > 100:
                loss_history.pop(0)

            # Detect trends
            if len(loss_history) >= 10:
                recent_avg = sum(loss_history[-10:]) / 10
                older_avg = sum(loss_history[-20:-10]) / 10 if len(loss_history) >= 20 else recent_avg

                # Calculate trend
                if older_avg > 0:
                    trend = (recent_avg - older_avg) / older_avg
                    experiment.log_metric(f"{loss_key}_trend", trend)

                    # Detect convergence
                    if abs(trend) < 0.001:  # Less than 0.1% change
                        experiment.log_metric(f"{loss_key}_converged", 1)


class ResourceMonitor:
    """Monitor system resources during training."""

    def __init__(self, config):
        self.config = config
        self._active_experiments: dict[str, weakref.ref] = {}
        self._monitor_threads: dict[str, threading.Thread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._lock = threading.RLock()

    def start(self, experiment: Experiment) -> None:
        """Start resource monitoring for an experiment."""
        if not (HAS_PSUTIL or HAS_GPUTIL):
            warnings.warn("Neither psutil nor GPUtil available, resource monitoring disabled", stacklevel=2)
            return

        with self._lock:
            exp_id = experiment.id
            if exp_id in self._active_experiments:
                return  # Already monitoring

            self._active_experiments[exp_id] = weakref.ref(experiment)

            # Start monitoring thread
            stop_event = threading.Event()
            self._stop_events[exp_id] = stop_event

            monitor_thread = threading.Thread(target=self._monitor_loop, args=(exp_id, stop_event), daemon=True)
            self._monitor_threads[exp_id] = monitor_thread
            monitor_thread.start()

    def stop(self, experiment_id: str) -> None:
        """Stop resource monitoring for an experiment."""
        with self._lock:
            if experiment_id in self._stop_events:
                self._stop_events[experiment_id].set()

            if experiment_id in self._monitor_threads:
                thread = self._monitor_threads[experiment_id]
                if thread.is_alive():
                    thread.join(timeout=1.0)
                del self._monitor_threads[experiment_id]

            self._active_experiments.pop(experiment_id, None)
            self._stop_events.pop(experiment_id, None)

    def cleanup(self) -> None:
        """Stop all resource monitoring."""
        with self._lock:
            for exp_id in list(self._active_experiments.keys()):
                self.stop(exp_id)

    def _monitor_loop(self, experiment_id: str, stop_event: threading.Event) -> None:
        """Main resource monitoring loop."""
        while not stop_event.wait(5.0):  # Monitor every 5 seconds
            try:
                experiment_ref = self._active_experiments.get(experiment_id)
                if not experiment_ref:
                    break

                experiment = experiment_ref()
                if not experiment:
                    break

                self._log_system_metrics(experiment)
                self._log_gpu_metrics(experiment)

            except Exception as e:
                warnings.warn(f"Error in resource monitor for {experiment_id}: {e}", stacklevel=2)
                break

    def _log_system_metrics(self, experiment: Experiment) -> None:
        """Log CPU and memory metrics."""
        if not (HAS_PSUTIL and self.config.monitor_cpu_usage):
            return

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            experiment.log_metric("cpu_usage_percent", cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            experiment.log_metric("memory_usage_percent", memory.percent)
            experiment.log_metric("memory_used_gb", memory.used / (1024**3))
            experiment.log_metric("memory_available_gb", memory.available / (1024**3))

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                experiment.log_metric("disk_read_mb", disk_io.read_bytes / (1024**2))
                experiment.log_metric("disk_write_mb", disk_io.write_bytes / (1024**2))

            # Network I/O
            net_io = psutil.net_io_counters()
            if net_io:
                experiment.log_metric("network_sent_mb", net_io.bytes_sent / (1024**2))
                experiment.log_metric("network_recv_mb", net_io.bytes_recv / (1024**2))

        except Exception as e:
            warnings.warn(f"Error logging system metrics: {e}", stacklevel=2)

    def _log_gpu_metrics(self, experiment: Experiment) -> None:
        """Log GPU metrics."""
        if not (self.config.monitor_gpu_memory and (HAS_GPUTIL or HAS_TORCH)):
            return

        try:
            # Try PyTorch first if available
            if HAS_TORCH and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    # GPU memory
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                    memory_cached = torch.cuda.memory_reserved(i) / (1024**3)  # GB

                    experiment.log_metric(f"gpu_{i}_memory_allocated_gb", memory_allocated)
                    experiment.log_metric(f"gpu_{i}_memory_cached_gb", memory_cached)

                    # GPU utilization (if available)
                    try:
                        import nvidia_ml_py3 as nvml

                        nvml.nvmlInit()
                        handle = nvml.nvmlDeviceGetHandleByIndex(i)
                        util = nvml.nvmlDeviceGetUtilizationRates(handle)
                        experiment.log_metric(f"gpu_{i}_utilization_percent", util.gpu)
                        experiment.log_metric(f"gpu_{i}_memory_utilization_percent", util.memory)
                    except ImportError:
                        pass

            # Fallback to GPUtil
            elif HAS_GPUTIL:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    experiment.log_metric(f"gpu_{i}_utilization_percent", gpu.load * 100)
                    experiment.log_metric(f"gpu_{i}_memory_used_gb", gpu.memoryUsed / 1024)
                    experiment.log_metric(f"gpu_{i}_memory_total_gb", gpu.memoryTotal / 1024)
                    experiment.log_metric(f"gpu_{i}_temperature_c", gpu.temperature)

        except Exception as e:
            warnings.warn(f"Error logging GPU metrics: {e}", stacklevel=2)
