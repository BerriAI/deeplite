import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    model: str
    gateway_url: str
    exa_key: str
    langsmith_endpoint: str
    langsmith_key: str
    langsmith_project: str
    litellm_endpoint: str
    litellm_key: str


def configure() -> Settings:
    project_root: Final = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    model: Final = os.getenv("DEEPLITE_MODEL", "claude-sonnet-5-5")
    gateway_url: Final = os.getenv("LITELLM_GATEWAY_URL", "https://gateway.litellm-sandbox.ai").rstrip("/")
    langsmith_endpoint: Final = os.getenv("LANGSMITH_OTLP_TRACES_ENDPOINT", "")
    litellm_endpoint: Final = os.getenv("LITELLM_OTLP_TRACES_ENDPOINT", "")
    for endpoint in (langsmith_endpoint, litellm_endpoint):
        parsed: Final = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or not parsed.path.endswith("/v1/traces"):
            raise ValueError("Set both OTLP traces endpoints to full HTTP URLs ending in /v1/traces")
    gateway: Final = urlparse(gateway_url)
    if gateway.scheme != "https" or not gateway.netloc:
        raise ValueError("Set LITELLM_GATEWAY_URL to an HTTPS base URL in .env")
    exa_key: Final = os.getenv("EXA_API_KEY", "")
    if not exa_key:
        raise ValueError("Set EXA_API_KEY in .env")
    langsmith_key: Final = os.getenv("LANGSMITH_API_KEY", "")
    litellm_key: Final = os.getenv("LITELLM_API_KEY", "")
    if not langsmith_key or not litellm_key:
        raise ValueError("Set LANGSMITH_API_KEY and LITELLM_API_KEY in .env")
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_TRACING_MODE"] = "otel"
    return Settings(
        model=model,
        gateway_url=gateway_url,
        exa_key=exa_key,
        langsmith_endpoint=langsmith_endpoint,
        langsmith_key=langsmith_key,
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "deeplite"),
        litellm_endpoint=litellm_endpoint,
        litellm_key=litellm_key,
    )
