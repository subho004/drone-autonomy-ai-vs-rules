"""Application configuration.

Exposes a typed `Settings` object loaded from environment variables
(and an optional `.env` file) via `pydantic-settings`.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings.

    Values are read from the environment (case-insensitive) and from a
    `.env` file if present. Unknown keys in `.env` are ignored.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "YOUR Backend"
    env: str = "development"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: Annotated[list[str], NoDecode] = ["*"]
    database_url: str = ""

    # Local data (SQLite + Parquet trajectories); repo-root relative.
    data_dir: str = "data"
    db_path: str = "data/drone_jev.db"

    # LLM / decision-model credentials (never logged; read from .env).
    openrouter_api_key: SecretStr = SecretStr("")
    gemini_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    budget_usd: float = 15.0

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, value: object) -> object:
        """Allow a comma-separated string for `CORS_ORIGINS` env vars."""

        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


# Single settings instance used across the app
settings = Settings()


__all__ = ["Settings", "settings"]
