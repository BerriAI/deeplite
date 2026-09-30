import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

from dotenv import load_dotenv
from exa_py import Exa
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, tool
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


@dataclass(frozen=True, slots=True)
class Settings:
    model: str
    exa_key: str
    langsmith_endpoint: str
    langsmith_key: str
    langsmith_project: str
    litellm_endpoint: str
    litellm_key: str


def configure() -> Settings:
    project_root: Final = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    model: Final = os.getenv("DEEPLITE_MODEL", "openai:gpt-6-luna")
    langsmith_endpoint: Final = os.getenv("LANGSMITH_OTLP_TRACES_ENDPOINT", "")
    litellm_endpoint: Final = os.getenv("LITELLM_OTLP_TRACES_ENDPOINT", "")
    for endpoint in (langsmith_endpoint, litellm_endpoint):
        parsed: Final = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or not parsed.path.endswith("/v1/traces"):
            raise ValueError("Set both OTLP traces endpoints to full HTTP URLs ending in /v1/traces")
    if model.startswith("openai:") and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in .env")
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
        exa_key=exa_key,
        langsmith_endpoint=langsmith_endpoint,
        langsmith_key=langsmith_key,
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "deeplite"),
        litellm_endpoint=litellm_endpoint,
        litellm_key=litellm_key,
    )


def configure_tracing(settings: Settings) -> TracerProvider:
    provider: Final = TracerProvider(resource=Resource.create({"service.name": "deeplite"}))
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=settings.langsmith_endpoint,
                headers={"x-api-key": settings.langsmith_key, "Langsmith-Project": settings.langsmith_project},
            )
        )
    )
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=settings.litellm_endpoint,
                headers={"Authorization": f"Bearer {settings.litellm_key}"},
            )
        )
    )
    trace.set_tracer_provider(provider)
    return provider


def make_search_tool(exa: Exa) -> BaseTool:
    @tool("web_search", description="Search the web with Exa for current facts and return excerpts with source URLs")
    def web_search(query: str) -> str:
        result: Final = exa.search(query, type="auto", num_results=5, contents={"highlights": True})
        return json.dumps(
            [{"title": item.title, "url": item.url, "highlights": item.highlights} for item in result.results]
        )

    return web_search


def build_agent(model: str | BaseChatModel, search_tool: BaseTool):
    from deepagents import create_deep_agent

    return create_deep_agent(
        model=model,
        system_prompt=(
            "You coordinate two subagents. Delegate the user's task to drafter, then send its draft "
            "to reviewer. Use both subagents before answering. Preserve source URLs from the draft. "
            "Return the improved result only."
        ),
        subagents=[
            {
                "name": "drafter",
                "description": "Research with Exa and draft a response to the delegated task",
                "system_prompt": (
                    "Use web_search for factual claims. Produce a concise draft for the assigned task "
                    "with source URLs. Return only the draft."
                ),
                "tools": [search_tool],
            },
            {
                "name": "reviewer",
                "description": "Review a draft for errors, omissions, and clarity",
                "system_prompt": "Review the draft against the original task. Return specific corrections or say it is ready.",
                "tools": [],
            },
        ],
    )


def main() -> None:
    parser: Final = argparse.ArgumentParser(description="Run a Deep Agent with two subagents and OTLP tracing")
    parser.add_argument("task", help="Task for the agent")
    args: Final = parser.parse_args()
    try:
        settings: Final = configure()
    except ValueError as error:
        parser.error(str(error))
    provider: Final = configure_tracing(settings)
    try:
        agent: Final = build_agent(settings.model, make_search_tool(Exa(api_key=settings.exa_key)))
        result: Final = agent.invoke({"messages": [{"role": "user", "content": args.task}]})
        answer: Final = next(message for message in reversed(result["messages"]) if isinstance(message, AIMessage))
        print(answer.text)
    finally:
        provider.shutdown()


if __name__ == "__main__":
    main()
