from typing import Final

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from deeplite.config import Settings


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
