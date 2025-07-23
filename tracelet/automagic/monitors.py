"""
Automatic monitors for training progress and system resources.
"""

import threading
import warnings
import weakref
from typing import Any, Optional

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

try:
    import nvidia_ml_py3 as nvml

    HAS_NVML = True
except ImportError:
    HAS_NVML = False

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

        # Metric history for recent metric access (experiment_id -> metrics history)
        self._metric_history: dict[str, list[dict]] = {}

        # Store original log_metric methods to restore later
        self._original_log_metric: dict[str, callable] = {}

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
                "last_epoch_value": None,  # Track explicit epoch metric values
                "last_val_metric_time": None,  # Track when validation metrics were seen
                "train_metrics_since_val": 0,  # Count training metrics since last validation
            }

            # Initialize metric history for this experiment
            self._metric_history[exp_id] = []

            # Hook into the experiment's log_metric method to capture metrics
            self._wrap_log_metric(experiment)

            # Start monitoring thread
            stop_event = threading.Event()
            self._stop_events[exp_id] = stop_event

            monitor_thread = threading.Thread(target=self._monitor_loop, args=(exp_id, stop_event), daemon=True)
            self._monitor_threads[exp_id] = monitor_thread
            monitor_thread.start()

    def _wrap_log_metric(self, experiment: Experiment) -> None:
        """Wrap the experiment's log_metric method to capture metrics."""
        exp_id = experiment.id
        if hasattr(experiment, "_original_log_metric_wrapped"):
            return  # Already wrapped

        # Store the original method
        original_log_metric = experiment.log_metric
        self._original_log_metric[exp_id] = original_log_metric

        def wrapped_log_metric(name: str, value: Any, iteration: Optional[int] = None):
            # Call the original log_metric method
            result = original_log_metric(name, value, iteration)

            # Only capture numeric metrics for monitoring
            if isinstance(value, (int, float)):
                self.capture_metric(exp_id, name, float(value), iteration)

            return result

        # Replace the method and mark as wrapped
        experiment.log_metric = wrapped_log_metric
        experiment._original_log_metric_wrapped = True

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

            # Restore original log_metric method if we wrapped it
            if experiment_id in self._original_log_metric:
                experiment_ref = self._active_experiments.get(experiment_id)
                if experiment_ref:
                    experiment = experiment_ref()
                    if experiment and hasattr(experiment, "_original_log_metric_wrapped"):
                        experiment.log_metric = self._original_log_metric[experiment_id]
                        delattr(experiment, "_original_log_metric_wrapped")
                del self._original_log_metric[experiment_id]

            # Clean up state
            self._active_experiments.pop(experiment_id, None)
            self._stop_events.pop(experiment_id, None)
            self._training_state.pop(experiment_id, None)
            self._metric_history.pop(experiment_id, None)

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

    def capture_metric(self, experiment_id: str, name: str, value: float, iteration: Optional[int] = None) -> None:
        """Capture a metric for monitoring purposes."""
        import time

        with self._lock:
            if experiment_id not in self._metric_history:
                return

            metric_entry = {"name": name, "value": value, "iteration": iteration, "timestamp": time.time()}

            # Add to history
            self._metric_history[experiment_id].append(metric_entry)

            # Keep only recent metrics (last 100 entries)
            if len(self._metric_history[experiment_id]) > 100:
                self._metric_history[experiment_id].pop(0)

    def _get_recent_metrics(self, experiment: Experiment) -> dict[str, float]:
        """Get recent metrics from the experiment."""
        import time

        exp_id = experiment.id
        if exp_id not in self._metric_history:
            return {}

        # Get metrics from the last 10 seconds
        current_time = time.time()
        recent_threshold = current_time - 10.0  # 10 seconds ago

        recent_metrics = {}
        for metric_entry in reversed(self._metric_history[exp_id]):
            if metric_entry["timestamp"] < recent_threshold:
                break

            # Use the most recent value for each metric name
            name = metric_entry["name"]
            if name not in recent_metrics:
                recent_metrics[name] = metric_entry["value"]

        return recent_metrics

    def _detect_epoch_changes(self, experiment_id: str, experiment: Experiment, metrics: dict[str, float]) -> None:
        """Detect epoch changes using robust heuristics."""
        state = self._training_state[experiment_id]
        import time

        current_time = time.time()

        # Strategy 1: Look for explicit epoch metric
        if "epoch" in metrics:
            epoch_value = metrics["epoch"]
            last_epoch_value = state.get("last_epoch_value")

            # Only increment if epoch value actually increased
            if last_epoch_value is None or epoch_value > last_epoch_value:
                state["last_epoch_value"] = epoch_value
                state["epoch"] = int(epoch_value)
                experiment.log_metric("detected_epoch", state["epoch"])
                return

        # Strategy 2: Infer epoch from validation metrics pattern
        # Look for validation metrics (typically logged once per epoch)
        val_metrics = [k for k in metrics if any(pattern in k.lower() for pattern in ["val_", "valid_", "test_"])]
        train_metrics = [k for k in metrics if any(pattern in k.lower() for pattern in ["train_", "training_"])]

        if val_metrics:
            # Reset training metric counter when we see validation metrics
            state["train_metrics_since_val"] = 0
            last_val_time = state.get("last_val_metric_time", 0)

            # Only count as new epoch if sufficient time has passed since last validation
            # This prevents multiple validation metrics in same epoch from being counted separately
            if current_time - last_val_time > self.config.epoch_detection_threshold_seconds:
                state["last_val_metric_time"] = current_time
                state["epoch"] = state.get("epoch", 0) + 1
                experiment.log_metric("detected_epoch", state["epoch"])

        elif train_metrics:
            # Track training metrics to help detect epoch boundaries
            state["train_metrics_since_val"] = state.get("train_metrics_since_val", 0) + 1

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
        self._nvml_initialized = False
        self._active_monitor_count = 0  # Track number of active monitors

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
            self._active_monitor_count += 1

            # Initialize NVML when the first monitor starts
            if self._active_monitor_count == 1 and HAS_NVML and not self._nvml_initialized:
                try:
                    nvml.nvmlInit()
                    self._nvml_initialized = True
                except Exception as e:
                    warnings.warn(f"Failed to initialize NVML: {e}", stacklevel=2)

            # Start monitoring thread
            stop_event = threading.Event()
            self._stop_events[exp_id] = stop_event

            monitor_thread = threading.Thread(target=self._monitor_loop, args=(exp_id, stop_event), daemon=True)
            self._monitor_threads[exp_id] = monitor_thread
            monitor_thread.start()

    def stop(self, experiment_id: str) -> None:
        """Stop resource monitoring for an experiment."""
        with self._lock:
            # Only proceed if this experiment is actually being monitored
            if experiment_id not in self._active_experiments:
                return

            if experiment_id in self._stop_events:
                self._stop_events[experiment_id].set()

            if experiment_id in self._monitor_threads:
                thread = self._monitor_threads[experiment_id]
                if thread.is_alive():
                    thread.join(timeout=1.0)
                del self._monitor_threads[experiment_id]

            # Clean up experiment tracking
            self._active_experiments.pop(experiment_id, None)
            self._stop_events.pop(experiment_id, None)
            self._active_monitor_count -= 1

            # Shutdown NVML when the last monitor stops
            if self._active_monitor_count == 0 and self._nvml_initialized:
                try:
                    nvml.nvmlShutdown()
                    self._nvml_initialized = False
                except Exception as e:
                    warnings.warn(f"Failed to shutdown NVML: {e}", stacklevel=2)

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
                    if HAS_NVML and self._nvml_initialized:
                        try:
                            handle = nvml.nvmlDeviceGetHandleByIndex(i)
                            util = nvml.nvmlDeviceGetUtilizationRates(handle)
                            experiment.log_metric(f"gpu_{i}_utilization_percent", util.gpu)
                            experiment.log_metric(f"gpu_{i}_memory_utilization_percent", util.memory)
                        except Exception as e:
                            warnings.warn(f"Error accessing GPU {i} utilization via nvml: {e}", stacklevel=2)

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
