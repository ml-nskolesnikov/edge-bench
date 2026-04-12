"""
Pydantic Models for API
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DeviceStatus(str, Enum):
    ONLINE = 'online'
    OFFLINE = 'offline'
    BUSY = 'busy'
    ERROR = 'error'


class ExperimentStatus(str, Enum):
    QUEUED = 'queued'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


class Backend(str, Enum):
    CPU = 'cpu'
    EDGETPU = 'edgetpu'


# Device models
class DeviceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    ip: str
    port: int = 8001
    description: str | None = None


class DeviceResponse(BaseModel):
    id: str
    name: str
    ip: str
    port: int
    status: DeviceStatus
    description: str | None
    last_seen: datetime | None
    created_at: datetime


class DeviceInfo(BaseModel):
    hostname: str
    platform: str
    python_version: str
    cpu_count: int
    memory_total_mb: float
    tpu_detected: bool
    tflite_version: str | None


# Experiment models
class ExperimentParams(BaseModel):
    backend: Backend = Backend.CPU
    batch_size: int = 1
    num_threads: int = 4
    warmup_runs: int = 10
    benchmark_runs: int = 100
    timeout_seconds: int = 600


class ExperimentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    device_id: str
    model_path: str
    script_path: str | None = 'benchmark_tflite.py'
    params: ExperimentParams = Field(default_factory=ExperimentParams)
    description: str | None = None


class ExperimentBatchCreate(BaseModel):
    """Create multiple experiments at once."""

    models: list[str]
    device: str
    backends: list[Backend] = [Backend.CPU]
    params: ExperimentParams = Field(default_factory=ExperimentParams)


class ExperimentResponse(BaseModel):
    id: str
    name: str
    device_id: str
    model_name: str
    model_path: str
    script_path: str
    params: dict[str, Any]
    status: ExperimentStatus
    logs: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


# Metrics models
class LatencyMetrics(BaseModel):
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float


class ThroughputMetrics(BaseModel):
    fps: float
    images_per_second: float


class ColdStartMetrics(BaseModel):
    model_load_ms: float
    first_inference_ms: float


class SystemMetrics(BaseModel):
    cpu_percent_mean: float
    cpu_percent_max: float
    memory_mb_mean: float
    memory_mb_max: float
    cpu_temp_celsius: float | None
    tpu_detected: bool


class ModelInfo(BaseModel):
    name: str
    hash: str
    size_bytes: int
    quantization: str | None


class BenchmarkResult(BaseModel):
    experiment_id: str
    device: str
    model: ModelInfo
    params: dict[str, Any]
    latency: LatencyMetrics
    throughput: ThroughputMetrics
    cold_start: ColdStartMetrics
    system: SystemMetrics
    device_info: DeviceInfo
    timestamp: datetime
    duration_seconds: float
    status: str


# File models
class FileType(str, Enum):
    MODEL = 'model'
    SCRIPT = 'script'
    OTHER = 'other'


class FileResponse(BaseModel):
    id: str
    name: str
    type: FileType
    size_bytes: int
    hash: str
    created_at: datetime
