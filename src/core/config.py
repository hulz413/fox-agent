from __future__ import annotations
from dataclasses import dataclass
from dotenv import load_dotenv
import os


@dataclass(frozen=True)
class Config:
    api_key: str
    base_url: str
    model: str
    timeout: float
    temperature: float

    @classmethod
    def from_env(cls) -> Config:
        load_dotenv()

        api_key = os.getenv("FOX_AGENT_CHAT_API_KEY").strip()
        if not api_key:
            raise ValueError("FOX_AGENT_CHAT_API_KEY is required")
        base_url = os.getenv("FOX_AGENT_CHAT_BASE_URL").strip()
        if not base_url:
            raise ValueError("FOX_AGENT_CHAT_BASE_URL is required")
        model = os.getenv("FOX_AGENT_CHAT_MODEL").strip()
        if not model:
            raise ValueError("FOX_AGENT_CHAT_MODEL is required")
        timeout = float(os.getenv("FOX_AGENT_CHAT_TIMEOUT", "60.0").strip())
        temperature = float(os.getenv("FOX_AGENT_CHAT_TEMPERATURE", "0.7").strip())

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            temperature=temperature,
        )
