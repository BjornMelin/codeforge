# coding=utf-8
"""Configuration management for CodeForge AI using Pydantic Settings.

This module defines the app settings loaded from env vars with validation.
"""

from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        use_async: Toggle for async features in DB clients.
        use_sparse: Toggle for sparse embeddings in RAG.
        use_gpu: Toggle for GPU acceleration.
        use_structured: Toggle for structured outputs in routing.
        qdrant_url: URL for Qdrant service.
        neo4j_uri: URI for Neo4j service.
        tavily_api_key: API key for Tavily search.
        openrouter_api_key: API key for OpenRouter.
    """

    model_config = SettingsConfigDict(
        env_prefix="CF_", case_sensitive=False, nested_model_default_partial_update=True
    )

    use_async: bool = Field(default=False, description="Toggle async DB operations.")
    use_sparse: bool = Field(default=False, description="Toggle sparse embeddings.")
    use_gpu: bool = Field(default=False, description="Toggle GPU usage.")
    use_structured: bool = Field(
        default=False, description="Toggle structured outputs."
    )
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    tavily_api_key: Optional[str] = Field(default=None, alias="TAVILY_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")

    @field_validator(
        "use_async", "use_sparse", "use_gpu", "use_structured", mode="before"
    )
    @classmethod
    def parse_bool(cls, v: str) -> bool:
        """Parse boolean from string per Pydantic best practices."""
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return bool(v)

    @model_validator(mode="after")
    def check_api_keys(self) -> "Settings":
        """Validate required API keys per Pydantic 2.11.7 best practices."""
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is required.")
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required.")
        return self


settings = Settings()  # Global instance for app-wide use
