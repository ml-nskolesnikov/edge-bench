"""
Agent Configuration
"""

import os

# Agent version - increment when making changes
AGENT_VERSION = '1.1.0'


class Settings:
    """Agent settings loaded from environment."""

    PORT: int = int(os.environ.get('EDGEBENCH_AGENT_PORT', '8001'))
    SERVER_URL: str = os.environ.get('EDGEBENCH_SERVER', '')

    # Auth: shared secret verified via X-Agent-Secret header.
    # Leave empty to disable auth (development / trusted LAN only).
    AGENT_SECRET: str = os.environ.get('EDGEBENCH_AGENT_SECRET', '')

    # Debug mode: enables /execute/code endpoint.
    # NEVER set to true in production.
    DEBUG: bool = os.environ.get('EDGEBENCH_DEBUG', '').lower() in ('1', 'true', 'yes')

    # Execution limits
    MAX_TIMEOUT: int = 3600  # 1 hour
    MAX_MEMORY_MB: int = 1024  # 1 GB

    # Paths
    MODELS_DIR: str = os.path.expanduser('~/models')
    INSTALL_DIR: str = os.path.expanduser('~/edge-bench-agent')
    STATE_FILE: str = '/var/lib/edgebench/state.json'
    CACHE_DIR: str = os.path.expanduser('~/edge-bench-agent/cache')


settings = Settings()
