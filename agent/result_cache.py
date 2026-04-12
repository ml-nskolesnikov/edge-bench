"""
Local Result Cache for Edge-Bench Agent.

Persists benchmark results to disk so they survive server outages.
A background sync task periodically pushes cached results to the server.
"""

import asyncio
from datetime import datetime
import json
from pathlib import Path
import time

from config import settings
import httpx

CACHE_DIR = Path(settings.INSTALL_DIR) / 'cache'
SYNC_INTERVAL = 30  # seconds between sync attempts
SYNC_TIMEOUT = 15  # seconds per HTTP request to server
MAX_CACHE_AGE_DAYS = 7  # auto-cleanup old entries


class ResultCache:
    """Local file-based cache for benchmark results."""

    def __init__(self):
        self._cache_dir = CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._syncing = False

    def save(self, experiment_id: str, result: dict) -> Path:
        """Save a result to local cache. Returns the cache file path."""
        entry = {
            'experiment_id': experiment_id,
            'result': result,
            'cached_at': datetime.utcnow().isoformat(),
            'synced': False,
        }

        filename = f'{experiment_id}.json'
        filepath = self._cache_dir / filename

        with open(filepath, 'w') as f:
            json.dump(entry, f, indent=2, default=str)

        return filepath

    def mark_synced(self, experiment_id: str) -> None:
        """Mark a cached result as successfully synced."""
        filepath = self._cache_dir / f'{experiment_id}.json'
        if filepath.exists():
            filepath.unlink()

    def get_unsynced(self) -> list[dict]:
        """Get all unsynced results from cache."""
        results = []
        for filepath in sorted(self._cache_dir.glob('*.json')):
            try:
                with open(filepath) as f:
                    entry = json.load(f)
                if not entry.get('synced', False):
                    results.append(entry)
            except (json.JSONDecodeError, OSError):
                continue
        return results

    def count_unsynced(self) -> int:
        """Count unsynced results."""
        return len(list(self._cache_dir.glob('*.json')))

    def cleanup_old(self) -> int:
        """Remove cache entries older than MAX_CACHE_AGE_DAYS."""
        cutoff = time.time() - (MAX_CACHE_AGE_DAYS * 86400)
        removed = 0
        for filepath in self._cache_dir.glob('*.json'):
            if filepath.stat().st_mtime < cutoff:
                filepath.unlink()
                removed += 1
        return removed

    async def sync_to_server(self, server_url: str) -> dict:
        """Push all unsynced results to the server.

        Returns dict with counts: {synced, failed, total}.
        """
        if self._syncing:
            return {'synced': 0, 'failed': 0, 'total': 0, 'status': 'already_syncing'}

        self._syncing = True
        unsynced = self.get_unsynced()
        synced = 0
        failed = 0

        try:
            if not unsynced:
                return {'synced': 0, 'failed': 0, 'total': 0}

            async with httpx.AsyncClient(timeout=SYNC_TIMEOUT) as client:
                for entry in unsynced:
                    exp_id = entry.get('experiment_id', '')
                    result = entry.get('result', {})

                    try:
                        resp = await client.post(
                            f'{server_url}/api/results/report',
                            json={
                                'experiment_id': exp_id,
                                'result': result,
                            },
                        )

                        if resp.status_code in (200, 201):
                            self.mark_synced(exp_id)
                            synced += 1
                            print(f'[Cache] Synced: {exp_id}')
                        elif resp.status_code == 409:
                            # Already exists on server, remove from cache
                            self.mark_synced(exp_id)
                            synced += 1
                            print(f'[Cache] Already on server: {exp_id}')
                        else:
                            failed += 1
                            print(
                                f'[Cache] Sync failed for {exp_id}: '
                                f'HTTP {resp.status_code}'
                            )
                    except (httpx.ConnectError, httpx.TimeoutException) as e:
                        failed += 1
                        print(f'[Cache] Server unreachable: {e}')
                        break  # Don't try more if server is down
                    except Exception as e:
                        failed += 1
                        print(f'[Cache] Sync error for {exp_id}: {e}')

        finally:
            self._syncing = False

        return {'synced': synced, 'failed': failed, 'total': len(unsynced)}


# Global instance
result_cache = ResultCache()


async def background_sync_loop(server_url: str):
    """Background task that periodically syncs cached results to the server."""
    if not server_url:
        print('[Cache] No SERVER_URL configured, background sync disabled')
        return

    print(f'[Cache] Background sync started (interval: {SYNC_INTERVAL}s)')
    print(f'[Cache] Server: {server_url}')

    # Initial cleanup
    removed = result_cache.cleanup_old()
    if removed:
        print(f'[Cache] Cleaned up {removed} old cache entries')

    while True:
        await asyncio.sleep(SYNC_INTERVAL)

        unsynced_count = result_cache.count_unsynced()
        if unsynced_count == 0:
            continue

        print(f'[Cache] {unsynced_count} unsynced results, attempting sync...')
        stats = await result_cache.sync_to_server(server_url)

        if stats.get('synced', 0) > 0:
            remaining = result_cache.count_unsynced()
            print(
                f'[Cache] Sync complete: {stats["synced"]} sent, '
                f'{stats["failed"]} failed, {remaining} remaining'
            )
