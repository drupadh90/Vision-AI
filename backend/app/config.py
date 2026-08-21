"""Central configuration for Vision AI.

Every knob is environment-driven (see `.env.example`) and every knob has a
default that works with **no API keys at all**, so the full stack boots offline.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

LLMProviderName = Literal["openai", "anthropic", "ollama", "mock"]
EmbeddingProviderName = Literal["openai", "hash"]
TranscriberName = Literal["whisper_api", "captions", "auto"]
ExecutorName = Literal["docker", "subprocess", "disabled"]
GuardianMode = Literal["enforce", "monitor"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
        protected_namespaces=(),
    )

    # ---- LLM routing -------------------------------------------------------
    llm_provider: LLMProviderName = Field("mock", alias="VISION_LLM_PROVIDER")
    llm_model: str = Field("gpt-4o-mini", alias="VISION_LLM_MODEL")
    planner_model: str = Field("", alias="VISION_PLANNER_MODEL")
    worker_model: str = Field("", alias="VISION_WORKER_MODEL")
    guardian_model: str = Field("", alias="VISION_GUARDIAN_MODEL")
    reflection_model: str = Field("", alias="VISION_REFLECTION_MODEL")
    llm_temperature: float = Field(0.2, alias="VISION_LLM_TEMPERATURE")
    llm_max_tokens: int = Field(2048, alias="VISION_LLM_MAX_TOKENS")
    llm_timeout_seconds: float = Field(90.0, alias="VISION_LLM_TIMEOUT_SECONDS")

    # ---- credentials -------------------------------------------------------
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    openai_base_url: str = Field("", alias="OPENAI_BASE_URL")
    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")

    # ---- embeddings --------------------------------------------------------
    embedding_provider: EmbeddingProviderName = Field("hash", alias="VISION_EMBEDDING_PROVIDER")
    embedding_model: str = Field("text-embedding-3-small", alias="VISION_EMBEDDING_MODEL")
    embedding_dim: int = Field(1536, alias="VISION_EMBEDDING_DIM")

    # ---- vector memory -----------------------------------------------------
    chroma_path: Path = Field(REPO_ROOT / "data" / "chroma", alias="VISION_CHROMA_PATH")
    twin_collection: str = Field("vision_digital_twin", alias="VISION_TWIN_COLLECTION")
    skill_collection: str = Field("vision_skills", alias="VISION_SKILL_COLLECTION")

    # ---- youtube skills ----------------------------------------------------
    transcriber: TranscriberName = Field("auto", alias="VISION_TRANSCRIBER")
    whisper_model: str = Field("whisper-1", alias="VISION_WHISPER_MODEL")
    skill_chunk_chars: int = Field(1200, alias="VISION_SKILL_CHUNK_CHARS")
    skill_chunk_overlap: int = Field(200, alias="VISION_SKILL_CHUNK_OVERLAP")
    skill_match_threshold: float = Field(0.14, alias="VISION_SKILL_MATCH_THRESHOLD")
    audio_cache: Path = Field(REPO_ROOT / "data" / "audio", alias="VISION_AUDIO_CACHE")

    # ---- goal mode ---------------------------------------------------------
    max_subtasks: int = Field(12, alias="VISION_MAX_SUBTASKS")
    max_concurrency: int = Field(4, alias="VISION_MAX_CONCURRENCY")
    max_task_retries: int = Field(2, alias="VISION_MAX_TASK_RETRIES")
    goal_wall_clock_seconds: float = Field(1800.0, alias="VISION_GOAL_WALL_CLOCK_SECONDS")

    # ---- guardian ----------------------------------------------------------
    guardian_enabled: bool = Field(True, alias="VISION_GUARDIAN_ENABLED")
    guardian_mode: GuardianMode = Field("enforce", alias="VISION_GUARDIAN_MODE")
    guardian_llm_review: bool = Field(True, alias="VISION_GUARDIAN_LLM_REVIEW")

    # ---- execution ---------------------------------------------------------
    executor: ExecutorName = Field("docker", alias="VISION_EXECUTOR")
    docker_image: str = Field("python:3.11-slim", alias="VISION_DOCKER_IMAGE")
    exec_timeout_seconds: float = Field(120.0, alias="VISION_EXEC_TIMEOUT_SECONDS")
    exec_memory_limit: str = Field("512m", alias="VISION_EXEC_MEMORY_LIMIT")
    exec_cpu_limit: float = Field(1.0, alias="VISION_EXEC_CPU_LIMIT")
    exec_network: str = Field("none", alias="VISION_EXEC_NETWORK")
    workspace_dir: Path = Field(REPO_ROOT / "data" / "workspace", alias="VISION_WORKSPACE_DIR")

    # ---- proactive ---------------------------------------------------------
    proactive_enabled: bool = Field(True, alias="VISION_PROACTIVE_ENABLED")
    proactive_min_confidence: float = Field(0.55, alias="VISION_PROACTIVE_MIN_CONFIDENCE")
    proactive_cooldown_seconds: float = Field(180.0, alias="VISION_PROACTIVE_COOLDOWN_SECONDS")

    # ---- server ------------------------------------------------------------
    host: str = Field("0.0.0.0", alias="VISION_HOST")
    port: int = Field(8000, alias="VISION_PORT")
    cors_origins: str = Field("*", alias="VISION_CORS_ORIGINS")
    log_level: str = Field("INFO", alias="VISION_LOG_LEVEL")

    # ---- derived helpers ---------------------------------------------------
    @field_validator("chroma_path", "audio_cache", "workspace_dir", mode="after")
    @classmethod
    def _absolutise(cls, v: Path) -> Path:
        return v if v.is_absolute() else (REPO_ROOT / v).resolve()

    @property
    def cors_origin_list(self) -> list[str]:
        raw = self.cors_origins.strip()
        return ["*"] if raw in {"", "*"} else [o.strip() for o in raw.split(",") if o.strip()]

    def model_for(self, role: str) -> str:
        """Resolve the model for a role, falling back to the global default."""
        override = {
            "planner": self.planner_model,
            "worker": self.worker_model,
            "guardian": self.guardian_model,
            "reflection": self.reflection_model,
        }.get(role, "")
        return override or self.llm_model

    def effective_llm_provider(self) -> LLMProviderName:
        """Downgrade to `mock` when the configured provider has no credentials.

        This is what lets a fresh clone run end-to-end with an empty `.env`
        instead of exploding on the first LLM call.
        """
        if self.llm_provider == "openai" and not self.openai_api_key:
            return "mock"
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            return "mock"
        return self.llm_provider

    def effective_embedding_provider(self) -> EmbeddingProviderName:
        if self.embedding_provider == "openai" and not self.openai_api_key:
            return "hash"
        return self.embedding_provider

    def ensure_dirs(self) -> None:
        for p in (self.chroma_path, self.audio_cache, self.workspace_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


settings = get_settings()
