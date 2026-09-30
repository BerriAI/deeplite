# deeplite

Copy `.env.example` to `.env` and set the API keys and OTLP endpoints. The LiteLLM proxy needs `general_settings.tracing.store: clickhouse` to expose `/v1/traces`

```sh
uv run deeplite "is litellm good?"
```

The researcher, skeptic, verifier, red team, and editor share one conversation and can hand control to one another. Search runs through Exa. The run stops after at most 40 graph steps, and its trace is exported to both configured OTLP destinations
