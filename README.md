# deeplite

Copy `.env.example` to `.env` and set the API keys and OTLP endpoints. `LITELLM_PROD_BASE` and `LITELLM_PROD_KEY` send model calls to the production gateway. `LITELLM_DEV_BASE` and `LITELLM_DEV_KEY` export traces to a separate LiteLLM proxy

```sh
uv run deeplite "is litellm good?"
```

The researcher, skeptic, verifier, red team, and editor share one conversation and can hand control to one another. Search runs through Exa. The run stops after at most 40 graph steps, and its trace is exported to both configured OTLP destinations

Agents share an in-memory virtual filesystem. All agents can read it; only the editor can write or edit files. The store lasts for the lifetime of the agent instance and does not write to disk

## Export traces to LiteLLM Lens

Enable tracing on the receiving LiteLLM proxy in `config.yaml` and set `CLICKHOUSE_URL` in that proxy's environment. The [tracing Compose stack](https://github.com/BerriAI/litellm/blob/main/docker/docker-compose.tracing.yml) provides a local example on port 4002

```yaml
general_settings:
  tracing:
    store: clickhouse
```

For that Compose stack, set these values in DeepLite's `.env`

```dotenv
LITELLM_DEV_BASE=http://localhost:4002/v1/traces
LITELLM_DEV_KEY=local-tracing-master-key
```

DeepLite sends OTLP/HTTP protobuf to the full `LITELLM_DEV_BASE` URL with `Authorization: Bearer <LITELLM_DEV_KEY>`. It also exports the same spans to the configured LangSmith endpoint. The CLI flushes both exporters before exiting. Open `http://localhost:4002/ui/?page=logs` to inspect the run; see the [Lens API docs](https://docs.litellm.ai/docs/proxy/lens) for trace read endpoints
