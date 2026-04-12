"""
System Metrics Collection for Raspberry Pi
"""

import platform
import subprocess
from typing import Any

import psutil


class SystemMetrics:
    """Collect system metrics on Raspberry Pi."""

    def __init__(self):
        self._tpu_checked = False
        self._tpu_available = False

    def get_device_info(self) -> dict[str, Any]:
        """Get static device information."""
        return {
            'hostname': platform.node(),
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_mb': round(psutil.virtual_memory().total / (1024 * 1024), 1),
            'tpu_detected': self.check_tpu(),
            'tflite_version': self._get_tflite_version(),
        }

    def get_current(self) -> dict[str, Any]:
        """Get current system metrics."""
        mem = psutil.virtual_memory()

        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_used_mb': round(mem.used / (1024 * 1024), 1),
            'memory_percent': mem.percent,
            'cpu_temp_celsius': self._get_cpu_temp(),
        }

    def collect_during_benchmark(self, duration_seconds: float) -> dict[str, Any]:
        """Collect metrics during benchmark execution."""
        import time

        cpu_samples = []
        mem_samples = []

        interval = min(0.1, duration_seconds / 10)
        samples = int(duration_seconds / interval)

        for _ in range(samples):
            cpu_samples.append(psutil.cpu_percent(interval=None))
            mem_samples.append(psutil.virtual_memory().used / (1024 * 1024))
            time.sleep(interval)

        return {
            'cpu_percent_mean': round(sum(cpu_samples) / len(cpu_samples), 1)
            if cpu_samples
            else 0,
            'cpu_percent_max': round(max(cpu_samples), 1) if cpu_samples else 0,
            'memory_mb_mean': round(sum(mem_samples) / len(mem_samples), 1)
            if mem_samples
            else 0,
            'memory_mb_max': round(max(mem_samples), 1) if mem_samples else 0,
            'cpu_temp_celsius': self._get_cpu_temp(),
            'tpu_detected': self.check_tpu(),
        }

    def detect_tpu_devices(self) -> list[str]:
        """Return list of available Edge TPU device paths."""
        import glob

        # PCIe Apex devices
        devices = glob.glob('/dev/apex_*')
        if devices:
            return sorted(devices)

        # USB devices via pycoral
        try:
            from pycoral.utils.edgetpu import list_edge_tpus

            tpus = list_edge_tpus()
            if tpus:
                return [t.get('path', f'usb:{i}') for i, t in enumerate(tpus)]
        except ImportError:
            pass

        # Fallback: check lsusb for Coral USB Accelerator
        try:
            result = subprocess.run(
                ['lsusb'], capture_output=True, text=True, timeout=5
            )
            count = result.stdout.count('Google') + result.stdout.count('Global Unichip')
            if count > 0:
                return [f'usb:{i}' for i in range(count)]
        except Exception:
            pass

        return []

    def check_tpu(self) -> bool:
        """Check if Edge TPU is available."""
        if self._tpu_checked:
            return self._tpu_available

        self._tpu_checked = True
        self._tpu_available = len(self.detect_tpu_devices()) > 0
        return self._tpu_available

    def _get_cpu_temp(self) -> float | None:
        """Get CPU temperature on Raspberry Pi."""
        # Method 1: thermal zone
        try:
            with open('/sys/class/thermal/thermal_zone0/temp') as f:
                temp = int(f.read().strip()) / 1000.0
                return round(temp, 1)
        except Exception:
            pass

        # Method 2: vcgencmd (Raspberry Pi specific)
        try:
            result = subprocess.run(
                ['vcgencmd', 'measure_temp'],
                capture_output=True,
                text=True,
                timeout=2,
            )
            # Output: temp=45.0'C
            temp_str = result.stdout.replace('temp=', '').replace("'C", '').strip()
            return float(temp_str)
        except Exception:
            pass

        return None

    def _get_tflite_version(self) -> str | None:
        """Get TFLite runtime version."""
        try:
            import tflite_runtime.interpreter as tflite

            return getattr(tflite, '__version__', 'unknown')
        except ImportError:
            pass

        try:
            import tensorflow as tf

            return tf.__version__
        except ImportError:
            pass

        return None
