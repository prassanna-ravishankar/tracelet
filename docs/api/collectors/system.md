# System Metrics Collector

::: tracelet.collectors.system.SystemMetricsCollector
options:
show_source: true
show_bases: true
merge_init_into_class: true
heading_level: 2

## Overview

The System Metrics Collector monitors system performance during experiment execution, providing insights into resource utilization and system health.

## Basic Usage

```python
import tracelet

# System metrics are collected automatically with default settings
exp = tracelet.start_logging(
    exp_name="system_monitoring_demo",
    project="performance_tracking",
    backend="mlflow"
)

# System metrics collected in background automatically
# Train your model here...

tracelet.stop_logging()
```

## Manual System Collection

```python
from tracelet.collectors.system import SystemMetricsCollector

# Create system collector
system_collector = SystemMetricsCollector(collect_interval=5.0)

# Initialize and start background collection
system_collector.initialize()
system_collector.start()

# Collect current metrics snapshot
current_metrics = system_collector.collect()

print("Current System Metrics:")
for category, metrics in current_metrics.items():
    print(f"  {category}: {metrics}")

# Stop collection
system_collector.stop()
```

## Configuration Options

### Collection Interval

```python
from tracelet.settings import TraceletSettings

# Configure system metrics collection
settings = TraceletSettings(
    project="performance_monitoring",
    backend=["mlflow"],
    track_system=True,        # Enable system tracking
    metrics_interval=10.0,    # Collect every 10 seconds
    track_gpu=True,          # Include GPU metrics (if available)
    track_disk=True,         # Include disk I/O metrics
    track_network=True       # Include network I/O metrics
)

tracelet.start_logging(
    exp_name="configured_monitoring",
    settings=settings
)
```

### Custom Collection Parameters

```python
from tracelet.collectors.system import SystemMetricsCollector

# Custom collector configuration
collector = SystemMetricsCollector(
    collect_interval=2.0,     # Collect every 2 seconds
    include_per_cpu=True,     # Include per-CPU metrics
    include_processes=True,   # Include top processes
    process_limit=5          # Limit to top 5 processes
)
```

## Collected Metrics

### CPU Metrics

- **cpu_percent**: Overall CPU utilization percentage
- **cpu_count_logical**: Number of logical CPU cores
- **cpu_count_physical**: Number of physical CPU cores
- **cpu_per_core**: Per-core utilization (if enabled)
- **load_average**: System load averages (1, 5, 15 minutes)

### Memory Metrics

- **memory_total**: Total system memory (bytes)
- **memory_available**: Available memory (bytes)
- **memory_used**: Used memory (bytes)
- **memory_percent**: Memory utilization percentage
- **swap_total**: Total swap space (bytes)
- **swap_used**: Used swap space (bytes)

### Disk Metrics

- **disk_total**: Total disk space (bytes)
- **disk_used**: Used disk space (bytes)
- **disk_free**: Free disk space (bytes)
- **disk_percent**: Disk utilization percentage
- **disk_read_bytes**: Cumulative bytes read
- **disk_write_bytes**: Cumulative bytes written

### Network Metrics

- **network_bytes_sent**: Cumulative bytes sent
- **network_bytes_recv**: Cumulative bytes received
- **network_packets_sent**: Cumulative packets sent
- **network_packets_recv**: Cumulative packets received

### GPU Metrics (if available)

- **gpu_count**: Number of available GPUs
- **gpu_utilization**: GPU utilization per device
- **gpu_memory_used**: GPU memory usage per device
- **gpu_memory_total**: Total GPU memory per device
- **gpu_temperature**: GPU temperature per device

## Practical Examples

### Performance Monitoring During Training

```python
import time
import tracelet
from tracelet.collectors.system import SystemMetricsCollector

# Start experiment with system monitoring
exp = tracelet.start_logging(
    exp_name="performance_monitored_training",
    project="ml_performance",
    backend="mlflow"
)

# Get system collector for manual snapshots
collector = SystemMetricsCollector(collect_interval=1.0)
collector.initialize()

# Training simulation with periodic monitoring
for epoch in range(100):
    # Simulate training work
    time.sleep(0.1)

    # Log training metrics
    train_loss = 1.0 / (epoch + 1)
    exp.log_metric("train_loss", train_loss, iteration=epoch)

    # Periodic system snapshots
    if epoch % 10 == 0:
        system_snapshot = collector.collect()

        # Log key system metrics
        exp.log_metric("cpu_percent", system_snapshot.get("cpu_percent", 0), iteration=epoch)
        exp.log_metric("memory_percent", system_snapshot.get("memory_percent", 0), iteration=epoch)

        # Log GPU metrics if available
        if "gpu" in system_snapshot:
            for i, gpu_info in enumerate(system_snapshot["gpu"]):
                exp.log_metric(f"gpu_{i}_utilization", gpu_info.get("utilization", 0), iteration=epoch)
                exp.log_metric(f"gpu_{i}_memory_percent", gpu_info.get("memory_percent", 0), iteration=epoch)

tracelet.stop_logging()
```

### Resource Usage Analysis

```python
import tracelet
import matplotlib.pyplot as plt
from tracelet.collectors.system import SystemMetricsCollector

# Detailed resource analysis
collector = SystemMetricsCollector(collect_interval=0.5)
collector.initialize()

# Start experiment
exp = tracelet.start_logging(
    exp_name="resource_analysis",
    project="performance_analysis",
    backend="mlflow"
)

# Collect metrics during workload
metrics_history = []
start_time = time.time()

# Simulate varying workload
for i in range(120):  # 60 seconds of collection
    # Simulate different workload intensities
    if i < 40:
        # Light workload
        time.sleep(0.1)
    elif i < 80:
        # Heavy workload simulation
        _ = [x**2 for x in range(10000)]
        time.sleep(0.3)
    else:
        # Cool down
        time.sleep(0.2)

    # Collect metrics
    metrics = collector.collect()
    metrics["timestamp"] = time.time() - start_time
    metrics_history.append(metrics)

# Analyze and plot resource usage
timestamps = [m["timestamp"] for m in metrics_history]
cpu_usage = [m.get("cpu_percent", 0) for m in metrics_history]
memory_usage = [m.get("memory_percent", 0) for m in metrics_history]

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(timestamps, cpu_usage, label="CPU %")
plt.ylabel("CPU Usage (%)")
plt.legend()
plt.title("System Resource Usage During Experiment")

plt.subplot(2, 1, 2)
plt.plot(timestamps, memory_usage, label="Memory %", color="red")
plt.ylabel("Memory Usage (%)")
plt.xlabel("Time (seconds)")
plt.legend()

plt.tight_layout()
plt.savefig("resource_usage.png")

# Log analysis results
exp.log_artifact("resource_usage.png", "analysis/resource_usage.png")

# Log summary statistics
avg_cpu = sum(cpu_usage) / len(cpu_usage)
max_cpu = max(cpu_usage)
avg_memory = sum(memory_usage) / len(memory_usage)
max_memory = max(memory_usage)

exp.log_params({
    "avg_cpu_usage": avg_cpu,
    "max_cpu_usage": max_cpu,
    "avg_memory_usage": avg_memory,
    "max_memory_usage": max_memory
})

tracelet.stop_logging()
```

### GPU Monitoring

```python
import tracelet
from tracelet.collectors.system import SystemMetricsCollector

def monitor_gpu_training():
    """Monitor GPU utilization during training."""

    exp = tracelet.start_logging(
        exp_name="gpu_monitored_training",
        project="gpu_performance",
        backend="mlflow"
    )

    collector = SystemMetricsCollector(collect_interval=1.0)
    collector.initialize()

    # Simulate GPU training
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")

        # Create model and data on GPU
        model = torch.nn.Linear(1000, 100).to(device)
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(50):
            # Simulate training batch
            data = torch.randn(128, 1000).to(device)
            target = torch.randn(128, 100).to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()

            # Log training metrics
            exp.log_metric("train_loss", loss.item(), iteration=epoch)

            # Log GPU metrics
            system_metrics = collector.collect()
            if "gpu" in system_metrics:
                for i, gpu_info in enumerate(system_metrics["gpu"]):
                    exp.log_metric(f"gpu_{i}_utilization", gpu_info.get("utilization", 0), iteration=epoch)
                    exp.log_metric(f"gpu_{i}_memory_used", gpu_info.get("memory_used", 0), iteration=epoch)
                    exp.log_metric(f"gpu_{i}_temperature", gpu_info.get("temperature", 0), iteration=epoch)
    else:
        print("No GPU available for monitoring")

    tracelet.stop_logging()

# Run GPU monitoring
monitor_gpu_training()
```

## Advanced Features

### Custom Metric Collection

```python
from tracelet.collectors.system import SystemMetricsCollector
import psutil
import json

class ExtendedSystemCollector(SystemMetricsCollector):
    """Extended system collector with custom metrics."""

    def collect(self):
        """Collect standard metrics plus custom ones."""
        # Get base metrics
        metrics = super().collect()

        # Add custom metrics
        custom_metrics = self._collect_custom_metrics()
        metrics.update(custom_metrics)

        return metrics

    def _collect_custom_metrics(self):
        """Collect additional custom system metrics."""
        custom = {}

        # Process count by state
        processes = list(psutil.process_iter(['status']))
        status_counts = {}
        for proc in processes:
            try:
                status = proc.info['status']
                status_counts[status] = status_counts.get(status, 0) + 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        custom["process_counts"] = status_counts
        custom["total_processes"] = len(processes)

        # Open file descriptors
        try:
            current_process = psutil.Process()
            custom["open_files"] = len(current_process.open_files())
        except (psutil.AccessDenied, AttributeError):
            custom["open_files"] = -1

        # System uptime
        boot_time = psutil.boot_time()
        uptime = time.time() - boot_time
        custom["system_uptime_hours"] = uptime / 3600

        return custom

# Usage
extended_collector = ExtendedSystemCollector(collect_interval=5.0)
extended_collector.initialize()

metrics = extended_collector.collect()
print("Extended metrics:", json.dumps(metrics, indent=2))
```

### Threshold-Based Alerts

```python
import tracelet
from tracelet.collectors.system import SystemMetricsCollector
import time

class AlertingSystemCollector(SystemMetricsCollector):
    """System collector with alerting capabilities."""

    def __init__(self, collect_interval=10.0, thresholds=None):
        super().__init__(collect_interval)
        self.thresholds = thresholds or {
            "cpu_percent": 90.0,
            "memory_percent": 85.0,
            "disk_percent": 95.0
        }
        self.alerts = []

    def collect(self):
        """Collect metrics and check thresholds."""
        metrics = super().collect()
        self._check_thresholds(metrics)
        return metrics

    def _check_thresholds(self, metrics):
        """Check if any metrics exceed thresholds."""
        timestamp = time.time()

        for metric_name, threshold in self.thresholds.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                if value > threshold:
                    alert = {
                        "timestamp": timestamp,
                        "metric": metric_name,
                        "value": value,
                        "threshold": threshold,
                        "severity": "warning" if value < threshold * 1.1 else "critical"
                    }
                    self.alerts.append(alert)
                    print(f"ALERT: {metric_name} = {value:.1f}% (threshold: {threshold}%)")

    def get_alerts(self):
        """Get all alerts since initialization."""
        return self.alerts.copy()

# Usage with alerting
exp = tracelet.start_logging(
    exp_name="monitored_with_alerts",
    project="system_alerting",
    backend="mlflow"
)

alerting_collector = AlertingSystemCollector(
    collect_interval=2.0,
    thresholds={
        "cpu_percent": 80.0,
        "memory_percent": 75.0
    }
)

alerting_collector.initialize()
alerting_collector.start()

# Simulate workload
time.sleep(30)

# Check for alerts
alerts = alerting_collector.get_alerts()
if alerts:
    print(f"Generated {len(alerts)} alerts during experiment")

    # Log alerts as experiment metadata
    exp.log_params({
        "alert_count": len(alerts),
        "max_cpu_alert": max([a["value"] for a in alerts if a["metric"] == "cpu_percent"], default=0),
        "critical_alerts": len([a for a in alerts if a["severity"] == "critical"])
    })

alerting_collector.stop()
tracelet.stop_logging()
```

## Error Handling

### Platform Compatibility

```python
import platform
from tracelet.collectors.system import SystemMetricsCollector

def create_platform_aware_collector():
    """Create system collector appropriate for current platform."""

    system = platform.system().lower()

    try:
        if system == "linux":
            # Linux-specific configuration
            collector = SystemMetricsCollector(
                collect_interval=5.0,
                include_per_cpu=True,
                include_processes=True
            )
        elif system == "darwin":  # macOS
            # macOS-specific configuration
            collector = SystemMetricsCollector(
                collect_interval=5.0,
                include_per_cpu=False  # Some limitations on macOS
            )
        elif system == "windows":
            # Windows-specific configuration
            collector = SystemMetricsCollector(
                collect_interval=10.0  # Slower collection on Windows
            )
        else:
            # Default configuration for unknown platforms
            collector = SystemMetricsCollector(collect_interval=10.0)

        collector.initialize()
        return collector

    except Exception as e:
        print(f"Failed to create system collector: {e}")
        return None

# Usage
collector = create_platform_aware_collector()
if collector:
    metrics = collector.collect()
    print(f"Platform: {platform.system()}")
    print(f"Available metrics: {list(metrics.keys())}")
else:
    print("System metrics collection not available on this platform")
```

### Graceful Degradation

```python
from tracelet.collectors.system import SystemMetricsCollector
import logging

def safe_system_collection():
    """Safely collect system metrics with fallback options."""

    try:
        # Try full system collection
        collector = SystemMetricsCollector(collect_interval=5.0)
        collector.initialize()
        metrics = collector.collect()

        # Validate critical metrics
        required_metrics = ["cpu_percent", "memory_percent"]
        missing_metrics = [m for m in required_metrics if m not in metrics]

        if missing_metrics:
            logging.warning(f"Missing critical metrics: {missing_metrics}")

        return metrics

    except ImportError as e:
        logging.error(f"Missing system monitoring dependencies: {e}")
        return {"error": "psutil not available", "basic_info": {"platform": platform.system()}}

    except Exception as e:
        logging.error(f"System collection failed: {e}")
        # Fallback to basic information
        return {
            "error": str(e),
            "fallback_metrics": {
                "timestamp": time.time(),
                "platform": platform.system(),
                "python_version": platform.python_version()
            }
        }

# Usage
system_info = safe_system_collection()

exp = tracelet.start_logging(
    exp_name="safe_monitoring",
    project="robust_experiments",
    backend="mlflow"
)

if "error" not in system_info:
    # Log successful metrics
    exp.log_params({
        "system_monitoring": "enabled",
        "cpu_cores": system_info.get("cpu_count_logical", "unknown"),
        "memory_gb": round(system_info.get("memory_total", 0) / (1024**3), 2)
    })
else:
    # Log error information
    exp.log_params({
        "system_monitoring": "failed",
        "error": system_info["error"]
    })

tracelet.stop_logging()
```

## Best Practices

### Resource-Aware Collection

```python
from tracelet.collectors.system import SystemMetricsCollector
import psutil

def configure_adaptive_collection():
    """Configure collection based on system resources."""

    # Get system information
    memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count()

    # Adapt collection based on system capacity
    if memory_gb < 4:  # Low memory system
        interval = 30.0  # Less frequent collection
        include_processes = False
    elif memory_gb < 8:  # Medium memory system
        interval = 15.0
        include_processes = True
    else:  # High memory system
        interval = 5.0
        include_processes = True

    # Adapt based on CPU count
    if cpu_count <= 2:
        include_per_cpu = False
    else:
        include_per_cpu = True

    collector = SystemMetricsCollector(
        collect_interval=interval,
        include_per_cpu=include_per_cpu,
        include_processes=include_processes
    )

    print(f"Configured system collection:")
    print(f"  Interval: {interval}s")
    print(f"  Per-CPU metrics: {include_per_cpu}")
    print(f"  Process metrics: {include_processes}")

    return collector

# Usage
adaptive_collector = configure_adaptive_collection()
adaptive_collector.initialize()
```

### Experiment Performance Impact

```python
import time
from tracelet.collectors.system import SystemMetricsCollector

def measure_collection_overhead():
    """Measure the performance impact of system collection."""

    # Baseline measurement without collection
    start_time = time.time()
    for i in range(1000):
        _ = [x**2 for x in range(1000)]  # Simulate work
    baseline_time = time.time() - start_time

    # Measurement with system collection
    collector = SystemMetricsCollector(collect_interval=0.1)  # Aggressive collection
    collector.initialize()
    collector.start()

    start_time = time.time()
    for i in range(1000):
        _ = [x**2 for x in range(1000)]  # Same work
    collection_time = time.time() - start_time

    collector.stop()

    overhead_percent = ((collection_time - baseline_time) / baseline_time) * 100

    print(f"Baseline time: {baseline_time:.3f}s")
    print(f"With collection: {collection_time:.3f}s")
    print(f"Overhead: {overhead_percent:.1f}%")

    return overhead_percent

# Measure and decide on collection strategy
overhead = measure_collection_overhead()

if overhead > 5.0:  # More than 5% overhead
    print("High overhead detected, using conservative collection")
    collection_interval = 30.0
else:
    print("Low overhead, using frequent collection")
    collection_interval = 5.0
```
