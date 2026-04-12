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

    # Execution limits
    MAX_TIMEOUT: int = 3600  # 1 hour
    MAX_MEMORY_MB: int = 1024  # 1 GB

    # Paths
    MODELS_DIR: str = os.path.expanduser('~/models')
    INSTALL_DIR: str = os.path.expanduser('~/edge-bench-agent')
    STATE_FILE: str = '/var/lib/edgebench/state.json'
    CACHE_DIR: str = os.path.expanduser('~/edge-bench-agent/cache')


settings = Settings()
