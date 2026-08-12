"""
System Metrics Collection for Raspberry Pi
"""

import os
import platform
import subprocess
import time
from typing import Any

import psutil

_TPU_CACHE_TTL = (
    30.0  # seconds — USB TPU connected after agent start is visible within this window
)


class SystemMetrics:
    """Collect system metrics on Raspberry Pi."""

    def __init__(self):
        self._tpu_last_check: float = 0.0
        self._tpu_available: bool = False

    def get_device_info(self) -> dict[str, Any]:
        """Get static device information."""
        return {
            'hostname': platform.node(),
            'platform': platform.platform(),
            'kernel_version': platform.release(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_mb': round(psutil.virtual_memory().total / (1024 * 1024), 1),
            'tpu_detected': self.check_tpu(),
            'tflite_version': self._get_tflite_version(),
            'libedgetpu_version': self._get_libedgetpu_version(),
            'cpu_governor': self._get_cpu_governor(),
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
        cpu_samples: list[float] = []
        # Process RSS — measures this process's actual resident memory, not total system usage.
        rss_samples: list[float] = []
        temp_samples: list[float] = []
        freq_samples: list[float] = []

        interval = min(0.1, duration_seconds / 10)
        samples = max(1, int(duration_seconds / interval))

        _proc = psutil.Process(os.getpid())

        for _ in range(samples):
            cpu_samples.append(psutil.cpu_percent(interval=None))
            rss_samples.append(_proc.memory_info().rss / (1024 * 1024))

            temp = self._get_cpu_temp()
            if temp is not None:
                temp_samples.append(temp)

            freq = psutil.cpu_freq()
            if freq:
                freq_samples.append(freq.current)

            time.sleep(interval)

        return {
            'cpu_percent_mean': round(sum(cpu_samples) / len(cpu_samples), 1)
            if cpu_samples
            else 0,
            'cpu_percent_max': round(max(cpu_samples), 1) if cpu_samples else 0,
            # process_rss_mb: RSS of this process (not total system memory).
            'process_rss_mb_mean': round(sum(rss_samples) / len(rss_samples), 1)
            if rss_samples
            else 0,
            'process_rss_mb_max': round(max(rss_samples), 1) if rss_samples else 0,
            # Final temp (backward compat) + max for throttle detection
            'cpu_temp_celsius': round(temp_samples[-1], 1) if temp_samples else None,
            'cpu_temp_max': round(max(temp_samples), 1) if temp_samples else None,
            # Min freq during benchmark for throttle detection
            'cpu_freq_mhz_min': round(min(freq_samples), 0) if freq_samples else None,
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
            count = result.stdout.count('Google') + result.stdout.count(
                'Global Unichip'
            )
            if count > 0:
                return [f'usb:{i}' for i in range(count)]
        except Exception:
            pass

        return []

    def check_tpu(self) -> bool:
        """Check if Edge TPU is available.

        Result is cached for _TPU_CACHE_TTL seconds so a USB TPU plugged in
        after agent startup is detected within one TTL window without the cost
        of running lsusb on every call.
        """
        now = time.monotonic()
        if now - self._tpu_last_check < _TPU_CACHE_TTL:
            return self._tpu_available
        self._tpu_last_check = now
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

    def _get_cpu_governor(self) -> str | None:
        """Read the cpufreq scaling governor for cpu0."""
        try:
            with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor') as f:
                return f.read().strip()
        except Exception:
            return None

    def _get_libedgetpu_version(self) -> str | None:
        """Return libedgetpu1-std package version via dpkg, or None if unavailable."""
        try:
            result = subprocess.run(
                ['dpkg-query', '-W', '-f=${Version}', 'libedgetpu1-std'],
                capture_output=True,
                text=True,
                timeout=5,
            )
            v = result.stdout.strip()
            return v if v else None
        except Exception:
            return None

    def _get_tflite_version(self) -> str | None:
        """Get the version of whichever TFLite runtime is installed."""
        from tflite_backend import TFLiteBackendError, backend_version, resolve_backend

        try:
            _, _, source = resolve_backend()
        except TFLiteBackendError:
            return None
        return backend_version(source)
