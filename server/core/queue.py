"""
Task Queue for Experiment Execution with Retry Support
"""

import asyncio
from datetime import UTC, datetime
import json

import httpx

from server.core.config import settings
from server.core.models import ExperimentStatus
from server.core.ws_manager import ws_manager
from server.db.database import get_db

# Retry settings
MAX_RETRIES = 5
RETRY_DELAY_BASE = 10  # seconds, exponential backoff
AGENT_BUSY_DELAY = 30  # wait before retry when agent is busy


class TaskQueue:
    """Async task queue for experiments with retry support."""

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._current_task: str | None = None
        self._retry_counts: dict[str, int] = {}

    async def add_task(self, experiment_id: str, priority: bool = False):
        """Add experiment to queue."""
        if priority:
            # For retries, add to front (will be processed after current)
            self._retry_counts[experiment_id] = self._retry_counts.get(experiment_id, 0)
        await self._queue.put(experiment_id)

    async def restore_pending_tasks(self):
        """Restore queued experiments from database on server restart."""
        async with get_db() as db:
            cursor = await db.execute(
                """SELECT id FROM experiments
                   WHERE status IN (?, ?)
                   ORDER BY created_at ASC""",
                (ExperimentStatus.QUEUED.value, ExperimentStatus.RUNNING.value),
            )
            pending = await cursor.fetchall()

            # Reset running experiments to queued
            await db.execute(
                """UPDATE experiments SET status = ? WHERE status = ?""",
                (ExperimentStatus.QUEUED.value, ExperimentStatus.RUNNING.value),
            )
            await db.commit()

        count = 0
        for row in pending:
            await self._queue.put(row['id'])
            count += 1

        if count > 0:
            print(f'[Queue] Restored {count} pending experiments from database')

        return count

    async def process_queue(self):
        """Process tasks from queue."""
        self._running = True

        # Restore pending tasks on startup
        await self.restore_pending_tasks()

        while self._running:
            try:
                experiment_id = await asyncio.wait_for(self._queue.get(), timeout=5.0)
                self._current_task = experiment_id

                success = await self._execute_experiment(experiment_id)

                if not success:
                    # Check if we should retry
                    retry_count = self._retry_counts.get(experiment_id, 0)
                    if retry_count < MAX_RETRIES:
                        self._retry_counts[experiment_id] = retry_count + 1
                        delay = RETRY_DELAY_BASE * (2**retry_count)
                        print(
                            f'[Queue] Retry {retry_count + 1}/{MAX_RETRIES} for {experiment_id} in {delay}s'
                        )

                        # Schedule retry
                        asyncio.create_task(self._delayed_retry(experiment_id, delay))
                    else:
                        print(f'[Queue] Max retries reached for {experiment_id}')
                        del self._retry_counts[experiment_id]
                else:
                    # Success - remove from retry tracking
                    if experiment_id in self._retry_counts:
                        del self._retry_counts[experiment_id]

                self._current_task = None
                self._queue.task_done()

            except TimeoutError:
                # Check for any failed experiments that need retry
                await self._check_failed_experiments()
                continue
            except Exception as e:
                print(f'[Queue] Error: {e}')
                if self._current_task:
                    self._current_task = None

    async def _delayed_retry(self, experiment_id: str, delay: float):
        """Schedule a delayed retry."""
        await asyncio.sleep(delay)

        # Check if experiment is still in queued/failed state
        async with get_db() as db:
            cursor = await db.execute(
                'SELECT status FROM experiments WHERE id = ?', (experiment_id,)
            )
            row = await cursor.fetchone()

            if row and row['status'] in (
                ExperimentStatus.QUEUED.value,
                ExperimentStatus.FAILED.value,
            ):
                # Reset to queued
                await db.execute(
                    'UPDATE experiments SET status = ?, error_message = NULL WHERE id = ?',
                    (ExperimentStatus.QUEUED.value, experiment_id),
                )
                await db.commit()

                # Re-add to queue
                await self._queue.put(experiment_id)
                print(f'[Queue] Re-queued experiment {experiment_id}')

    async def _check_failed_experiments(self):
        """Periodically check for failed experiments that can be retried."""
        async with get_db() as db:
            # Get recently failed experiments (within last hour)
            cursor = await db.execute(
                """SELECT id, error_message FROM experiments
                   WHERE status = ?
                   AND completed_at > datetime('now', '-1 hour')
                   AND (error_message LIKE '%busy%' OR error_message LIKE '%timeout%' OR error_message LIKE '%connection%')
                   LIMIT 5""",
                (ExperimentStatus.FAILED.value,),
            )
            failed = await cursor.fetchall()

        for row in failed:
            exp_id = row['id']
            if exp_id not in self._retry_counts:
                self._retry_counts[exp_id] = 0
                await self._delayed_retry(exp_id, RETRY_DELAY_BASE)

    def stop(self):
        """Stop queue processing."""
        self._running = False

    async def _get_experiment_and_device(self, experiment_id: str):
        """Get experiment and device info. Returns (experiment, device) or (None, None)."""
        async with get_db() as db:
            cursor = await db.execute(
                'SELECT * FROM experiments WHERE id = ?', (experiment_id,)
            )
            experiment = await cursor.fetchone()

            if not experiment:
                return None, None

            cursor = await db.execute(
                'SELECT * FROM devices WHERE id = ?', (experiment['device_id'],)
            )
            device = await cursor.fetchone()

        return experiment, device

    async def _check_device_health(self, agent_url: str) -> tuple[bool, str]:
        """Check if device is online. Returns (is_online, error_message)."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f'{agent_url}/health')
                if resp.status_code != 200:
                    return False, 'Device offline'
                return True, ''
        except Exception as e:
            return False, f'Device unreachable: {e}'

    async def _run_on_agent(
        self,
        agent_url: str,
        experiment: dict,
        stream_callback_url: str | None = None,
    ) -> dict:
        """Send task to agent and get result."""
        params = json.loads(experiment['params'])

        payload: dict = {
            'experiment_id': experiment['id'],
            'model_path': experiment['model_path'],
            'script': experiment['script_path'],
            'params': params,
        }
        if stream_callback_url:
            payload['stream_callback_url'] = stream_callback_url

        async with httpx.AsyncClient(timeout=settings.TASK_TIMEOUT_SECONDS) as client:
            response = await client.post(f'{agent_url}/execute', json=payload)

            if response.status_code == 503:
                raise httpx.HTTPStatusError(
                    'Agent busy', request=None, response=response
                )

            if response.status_code != 200:
                raise Exception(f'Agent error: {response.text}')

            return response.json()

    async def _save_results(self, experiment_id: str, result: dict):
        """Save experiment results to database and log to integrations."""
        async with get_db() as db:
            await db.execute(
                """INSERT OR REPLACE INTO results
                (id, experiment_id, metrics, created_at)
                VALUES (?, ?, ?, ?)""",
                (
                    f'res_{experiment_id}',
                    experiment_id,
                    json.dumps(result),
                    datetime.now(UTC).isoformat(),
                ),
            )

            await db.execute(
                """UPDATE experiments
                SET status = ?, completed_at = ?, logs = ?, error_message = NULL
                WHERE id = ?""",
                (
                    ExperimentStatus.COMPLETED.value,
                    datetime.now(UTC).isoformat(),
                    result.get('logs', ''),
                    experiment_id,
                ),
            )
            await db.commit()

        # Log to MLflow if configured
        await self._log_to_integrations(experiment_id, result)

    async def _log_to_integrations(self, experiment_id: str, result: dict) -> None:
        """Log result to enabled integrations (MLflow, W&B)."""
        try:
            # Load settings from DB (may be overridden via settings page)
            async with get_db() as db:
                cursor = await db.execute('SELECT key, value FROM settings')
                saved = {row['key']: row['value'] for row in await cursor.fetchall()}

            mlflow_uri = saved.get('mlflow_uri', settings.MLFLOW_TRACKING_URI)

            if mlflow_uri:
                from server.integrations.mlflow_logger import MLflowLogger

                loop = __import__('asyncio').get_event_loop()
                mlflow_exp = saved.get(
                    'mlflow_experiment', settings.MLFLOW_EXPERIMENT_NAME
                )
                logger_obj = MLflowLogger(mlflow_uri, mlflow_exp)
                if logger_obj.enabled:
                    await loop.run_in_executor(
                        None, logger_obj.log_experiment, result
                    )
        except Exception as e:
            print(f'[Queue] Integration logging error: {e}')

    async def _execute_experiment(self, experiment_id: str) -> bool:
        """Execute a single experiment. Returns True on success."""
        experiment, device = await self._get_experiment_and_device(experiment_id)

        if not experiment:
            return True  # Don't retry non-existent experiments

        if experiment['status'] in (
            ExperimentStatus.COMPLETED.value,
            ExperimentStatus.CANCELLED.value,
        ):
            return True

        if not device:
            await self._mark_failed(experiment_id, 'Device not found')
            return True  # Don't retry - device doesn't exist

        agent_url = f'http://{device["ip"]}:{device["port"]}'

        is_online, error = await self._check_device_health(agent_url)
        if not is_online:
            await self._mark_failed(experiment_id, error)
            return False  # Retry later

        # Update status to running
        async with get_db() as db:
            await db.execute(
                'UPDATE experiments SET status = ?, started_at = ? WHERE id = ?',
                (
                    ExperimentStatus.RUNNING.value,
                    datetime.now(UTC).isoformat(),
                    experiment_id,
                ),
            )
            await db.commit()

        # Broadcast "running" status to WebSocket clients
        await ws_manager.broadcast(
            experiment_id, {'type': 'status', 'status': ExperimentStatus.RUNNING.value}
        )

        # Build a callback URL so the agent can stream metrics back
        stream_callback_url = (
            f'http://{settings.HOST}:{settings.PORT}'
            f'/api/experiments/{experiment_id}/metric'
        )
        # Use 127.0.0.1 when HOST is 0.0.0.0 (loopback for agent → server)
        if settings.HOST == '0.0.0.0':
            stream_callback_url = (
                f'http://127.0.0.1:{settings.PORT}'
                f'/api/experiments/{experiment_id}/metric'
            )

        try:
            result = await self._run_on_agent(
                agent_url, dict(experiment), stream_callback_url
            )
            await self._save_results(experiment_id, result)
            # Broadcast "completed"
            await ws_manager.broadcast(
                experiment_id,
                {'type': 'status', 'status': ExperimentStatus.COMPLETED.value},
            )
            await ws_manager.broadcast(experiment_id, {'type': 'done'})
            return True

        except httpx.HTTPStatusError:
            await self._mark_failed(experiment_id, 'Agent busy')
            return False
        except httpx.TimeoutException:
            await self._mark_failed(experiment_id, 'Execution timeout')
            return False
        except httpx.ConnectError as e:
            await self._mark_failed(experiment_id, f'Connection error: {e}')
            return False
        except Exception as e:
            await self._mark_failed(experiment_id, str(e))
            return True  # Don't retry unknown errors

    async def _mark_failed(self, experiment_id: str, error: str):
        """Mark experiment as failed (may be retried)."""
        async with get_db() as db:
            await db.execute(
                """UPDATE experiments
                SET status = ?, error_message = ?, completed_at = ?
                WHERE id = ?""",
                (
                    ExperimentStatus.FAILED.value,
                    error,
                    datetime.now(UTC).isoformat(),
                    experiment_id,
                ),
            )
            await db.commit()
        await ws_manager.broadcast(
            experiment_id,
            {'type': 'status', 'status': ExperimentStatus.FAILED.value, 'error': error},
        )
        print(f'[Queue] Experiment {experiment_id} failed: {error}')

    def get_queue_status(self) -> dict:
        """Get current queue status."""
        return {
            'queue_size': self._queue.qsize(),
            'current_task': self._current_task,
            'retry_counts': dict(self._retry_counts),
            'running': self._running,
        }


# Global task queue instance
task_queue = TaskQueue()
