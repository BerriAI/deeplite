import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    model: str
    prod_base: str
    prod_key: str
    exa_key: str
    langsmith_endpoint: str
    langsmith_key: str
    langsmith_project: str
    dev_base: str
    dev_key: str


def configure() -> Settings:
    project_root: Final = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    model: Final = os.getenv("DEEPLITE_MODEL", "claude-sonnet-5-5")
    prod_base: Final = os.getenv("LITELLM_PROD_BASE", "").rstrip("/")
    langsmith_endpoint: Final = os.getenv("LANGSMITH_OTLP_TRACES_ENDPOINT", "")
    dev_base: Final = os.getenv("LITELLM_DEV_BASE", "")
    for endpoint in (langsmith_endpoint, dev_base):
        parsed: Final = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or not parsed.path.endswith("/v1/traces"):
            raise ValueError("Set both OTLP traces endpoints to full HTTP URLs ending in /v1/traces")
    gateway: Final = urlparse(prod_base)
    if gateway.scheme != "https" or not gateway.netloc:
        raise ValueError("Set LITELLM_PROD_BASE to an HTTPS base URL in .env")
    exa_key: Final = os.getenv("EXA_API_KEY", "")
    if not exa_key:
        raise ValueError("Set EXA_API_KEY in .env")
    langsmith_key: Final = os.getenv("LANGSMITH_API_KEY", "")
    prod_key: Final = os.getenv("LITELLM_PROD_KEY", "")
    dev_key: Final = os.getenv("LITELLM_DEV_KEY", "")
    if not langsmith_key or not prod_key or not dev_key:
        raise ValueError("Set LANGSMITH_API_KEY, LITELLM_PROD_KEY, and LITELLM_DEV_KEY in .env")
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_TRACING_MODE"] = "otel"
    return Settings(
        model=model,
        prod_base=prod_base,
        prod_key=prod_key,
        exa_key=exa_key,
        langsmith_endpoint=langsmith_endpoint,
        langsmith_key=langsmith_key,
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "deeplite"),
        dev_base=dev_base,
        dev_key=dev_key,
    )
