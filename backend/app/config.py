from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "mediq_wait_events"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    data_file: Path = Path("../data/raw/TTSH Oct 25 - 04 May 26 - WaitTimeAdded.xlsx")
    anthropic_api_key: str = ""
    llm_model: str = "claude-haiku-4-5-20251001"
    cors_origins: str = "http://localhost:3000"

    # ingestion data-quality bounds (spec DQ-2)
    max_wait_sec: float = 14_400.0  # 4 hours
    min_wait_sec: float = 0.0


settings = Settings()
