"""Application configuration via pydantic-settings. No secrets in code."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    anthropic_api_key: str = ""
    intent_model: str = "claude-haiku-4-5"
    synthesis_model: str = "claude-haiku-4-5"  # thrift: all-Haiku for now

    # Datastores
    database_url: str = "postgresql+asyncpg://locale:locale@localhost:5432/locale"
    redis_url: str = "redis://localhost:6379/0"

    # External sources
    nominatim_user_agent: str = "locale-agent/0.1 (you@example.com)"
    nominatim_base_url: str = "https://nominatim.openstreetmap.org"
    overpass_base_url: str = "https://overpass-api.de/api/interpreter"
    wikipedia_base_url: str = "https://en.wikipedia.org/w/api.php"
    gdelt_base_url: str = "https://api.gdeltproject.org/api/v2/doc/doc"

    # Reddit — free "script" app at https://www.reddit.com/prefs/apps. Optional;
    # the adapter no-ops gracefully when these are blank.
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "locale-agent/0.1 by u/locale_dev"

    # Apify — third-party scraping vendor used to source Reddit (and later
    # Nextdoor), whose official APIs are gated. Token: apify.com → Settings →
    # Integrations → API token. Optional; the adapter no-ops without it.
    apify_token: str = ""
    apify_reddit_actor: str = "practicaltools/apify-reddit-api"  # API-based, fast (~seconds)
    apify_base_url: str = "https://api.apify.com/v2"
    apify_timeout_s: int = 60

    # Guardrails
    cost_cap_external_calls: int = 12

    # Logging
    log_level: str = "INFO"

    @property
    def has_llm(self) -> bool:
        """True when a real Anthropic key is configured (gates real LLM vs. fallback)."""
        key = self.anthropic_api_key
        return bool(key and key.startswith("sk-ant-") and "REPLACE_ME" not in key)

    @property
    def has_reddit(self) -> bool:
        """True when Reddit app credentials are configured."""
        return bool(self.reddit_client_id and self.reddit_client_secret)

    @property
    def has_apify(self) -> bool:
        """True when an Apify token is configured (gates Reddit/Nextdoor scraping)."""
        return bool(self.apify_token)


@lru_cache
def get_settings() -> Settings:
    return Settings()
