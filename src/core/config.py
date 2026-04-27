from __future__ import annotations
from dataclasses import dataclass
from dotenv import load_dotenv
import os


@dataclass(frozen=True)
class Config:
    chat_api_key: str
    chat_base_url: str
    chat_model: str
    chat_timeout: float
    chat_temperature: float

    embedding_provider: str
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str | None = None
    embedding_timeout: float | None = None

    @classmethod
    def from_env(cls) -> Config:
        load_dotenv()

        chat_api_key = os.getenv("FOX_AGENT_CHAT_API_KEY").strip()
        if not chat_api_key:
            raise ValueError("FOX_AGENT_CHAT_API_KEY is required")
        chat_base_url = os.getenv("FOX_AGENT_CHAT_BASE_URL").strip()
        if not chat_base_url:
            raise ValueError("FOX_AGENT_CHAT_BASE_URL is required")
        chat_model = os.getenv("FOX_AGENT_CHAT_MODEL").strip()
        if not chat_model:
            raise ValueError("FOX_AGENT_CHAT_MODEL is required")
        chat_timeout = float(os.getenv("FOX_AGENT_CHAT_TIMEOUT", "60.0").strip())
        chat_temperature = float(os.getenv("FOX_AGENT_CHAT_TEMPERATURE", "0.7").strip())

        embedding_provider = os.getenv("FOX_AGENT_EMBEDDING_PROVIDER").strip()
        embedding_api_key = os.getenv("FOX_AGENT_EMBEDDING_API_KEY", "").strip()
        embedding_base_url = os.getenv("FOX_AGENT_EMBEDDING_BASE_URL", "").strip()
        embedding_model = os.getenv("FOX_AGENT_EMBEDDING_MODEL", "").strip()
        embedding_timeout = float(
            os.getenv("FOX_AGENT_EMBEDDING_TIMEOUT", "60.0").strip()
        )

        return cls(
            chat_api_key=chat_api_key,
            chat_base_url=chat_base_url,
            chat_model=chat_model,
            chat_timeout=chat_timeout,
            chat_temperature=chat_temperature,
            embedding_provider=embedding_provider,
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
            embedding_model=embedding_model,
            embedding_timeout=embedding_timeout,
        )
