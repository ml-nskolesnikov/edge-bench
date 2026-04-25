"""
Server Configuration
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Current agent version - should match agent/config.py AGENT_VERSION
AGENT_VERSION = '1.1.0'


class Settings(BaseSettings):
    """Application settings."""

    # Server
    HOST: str = '0.0.0.0'
    PORT: int = 8000
    DEBUG: bool = False

    # Database
    DATABASE_PATH: Path = Path('data/edgebench.db')

    # Files storage
    UPLOAD_DIR: Path = Path('data/uploads')
    MODELS_DIR: Path = Path('data/models')
    SCRIPTS_DIR: Path = Path('data/scripts')

    # Task queue
    MAX_CONCURRENT_TASKS: int = 1
    TASK_TIMEOUT_SECONDS: int = 3600  # 1 hour

    # Agent communication
    AGENT_TIMEOUT_SECONDS: int = 30
    AGENT_DEFAULT_PORT: int = 8001

    # Integrations (empty string = disabled)
    MLFLOW_TRACKING_URI: str = ''
    MLFLOW_EXPERIMENT_NAME: str = 'edge-bench'
    WANDB_API_KEY: str = ''

    model_config = SettingsConfigDict(
        env_prefix='EDGEBENCH_',
        env_file='.env',
    )


settings = Settings()

# Ensure directories exist
settings.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
