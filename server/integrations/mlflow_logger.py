"""
MLflow Logger for Edge-Bench Benchmark Results
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MLflowLogger:
    """Log benchmark results to MLflow tracking server."""

    def __init__(self, tracking_uri: str, experiment_name: str = 'edge-bench'):
        try:
            import mlflow

            self._mlflow = mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._enabled = True
            logger.info(f'MLflow logger initialized: {tracking_uri}')
        except ImportError:
            self._enabled = False
            logger.warning('mlflow not installed; logging disabled')

    @property
    def enabled(self) -> bool:
        return self._enabled

    def log_experiment(self, result: dict[str, Any]) -> str | None:
        """Log a benchmark result dict to MLflow. Returns run_id or None."""
        if not self._enabled:
            return None

        mlflow = self._mlflow
        latency = result.get('latency', {})
        throughput = result.get('throughput', {})
        system = result.get('system', {})
        model_info = result.get('model', {})
        params = result.get('params', {})
        device_info = result.get('device_info', {})

        try:
            with mlflow.start_run() as run:
                # Tags
                mlflow.set_tags(
                    {
                        'experiment_id': result.get('experiment_id', ''),
                        'model_name': model_info.get('name', ''),
                        'backend': params.get('backend', 'cpu'),
                        'device': result.get('device', ''),
                        'tpu_detected': str(system.get('tpu_detected', False)),
                        'quantization': model_info.get('quantization', ''),
                    }
                )

                # Parameters
                mlflow.log_params(
                    {
                        'num_threads': params.get('num_threads', 4),
                        'benchmark_runs': params.get('benchmark_runs', 100),
                        'warmup_runs': params.get('warmup_runs', 10),
                        'batch_size': params.get('batch_size', 1),
                        'model_size_bytes': model_info.get('size_bytes', 0),
                        'cpu_count': device_info.get('cpu_count', 0),
                    }
                )

                # Metrics
                metrics: dict[str, float] = {}
                if latency:
                    metrics.update(
                        {
                            'latency_mean_ms': latency.get('mean_ms', 0),
                            'latency_std_ms': latency.get('std_ms', 0),
                            'latency_p50_ms': latency.get('p50_ms', 0),
                            'latency_p90_ms': latency.get('p90_ms', 0),
                            'latency_p95_ms': latency.get('p95_ms', 0),
                            'latency_p99_ms': latency.get('p99_ms', 0),
                        }
                    )
                if throughput:
                    metrics['fps'] = throughput.get('fps', 0)
                if system:
                    metrics.update(
                        {
                            'cpu_percent_mean': system.get('cpu_percent_mean', 0),
                            'memory_mb_mean': system.get('memory_mb_mean', 0),
                        }
                    )
                    if system.get('cpu_temp_celsius') is not None:
                        metrics['cpu_temp_celsius'] = system['cpu_temp_celsius']

                mlflow.log_metrics(metrics)
                run_id = run.info.run_id
                logger.info(f'MLflow: logged run {run_id}')
                return run_id

        except Exception as e:
            logger.error(f'MLflow logging failed: {e}')
            return None
