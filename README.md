# deeplite

Copy `.env.example` to `.env` and set the API keys and OTLP endpoints. `LITELLM_PROD_*` configures model calls, while `LITELLM_DEV_*` configures local trace export. The LiteLLM proxy needs `general_settings.tracing.store: clickhouse` to expose `/v1/traces`

```sh
uv run deeplite "is litellm good?"
```

The researcher, skeptic, verifier, red team, and editor share one conversation and can hand control to one another. Search runs through Exa. The run stops after at most 40 graph steps, and its trace is exported to both configured OTLP destinations

Agents share an in-memory virtual filesystem. All agents can read it; only the editor can write or edit files. The store lasts for the lifetime of the agent instance and does not write to disk
