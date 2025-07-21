import contextlib
import logging
import platform
import threading
import time
from typing import Any

import psutil

from ..core.interfaces import CollectorInterface

logger = logging.getLogger(__name__)

try:
    import pynvml

    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False


class SystemMetricsCollector(CollectorInterface):
    """Collector for system metrics including CPU, memory, and GPU usage"""

    def __init__(self, collect_interval: float = 10.0):
        self.collect_interval = collect_interval
        self._stop_event = threading.Event()
        self._collection_thread = None
        self._metrics = {}
        self._nvml_initialized = False

    def initialize(self):
        if NVIDIA_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
            except Exception:
                self._nvml_initialized = False

    def collect(self) -> dict[str, Any]:
        """Collect current system information and metrics"""
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "memory_total": psutil.virtual_memory().total,
            "hostname": platform.node(),
        }

        # Add current metrics
        metrics = self._collect_current_metrics()
        system_info.update(metrics)

        return system_info

    def _collect_current_metrics(self) -> dict[str, Any]:
        """Collect current CPU, memory and GPU metrics"""
        metrics = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available": psutil.virtual_memory().available,
        }

        # Collect GPU metrics if available
        if self._nvml_initialized:
            gpu_metrics = {}
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                    gpu_metrics[f"gpu_{i}"] = {
                        "memory_total": info.total,
                        "memory_used": info.used,
                        "memory_free": info.free,
                        "utilization": utilization.gpu,
                    }

                metrics["gpu"] = gpu_metrics
            except Exception as e:
                metrics["gpu_error"] = str(e)

        return metrics

    def _collection_loop(self):
        """Background collection loop"""
        while not self._stop_event.is_set():
            try:
                metrics = self._collect_current_metrics()
                self._metrics = metrics
            except Exception:
                logger.exception("Failed to collect system metrics")
            time.sleep(self.collect_interval)

    def start(self):
        """Start the background collection thread"""
        if self._collection_thread is None:
            self._stop_event.clear()
            self._collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
            self._collection_thread.start()

    def stop(self):
        """Stop the background collection thread"""
        if self._collection_thread is not None:
            self._stop_event.set()
            self._collection_thread.join()
            self._collection_thread = None

        if self._nvml_initialized:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
