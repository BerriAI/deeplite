# deeplite

A small Deep Agents CLI with a coordinating agent, a drafter, and a reviewer. The drafter can search the web through Exa and returns source URLs. The coordinator passes work through Deep Agents' `task` tool. One OpenTelemetry tracer sends the same spans to LangSmith and LiteLLM

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`, `EXA_API_KEY`, `LANGSMITH_API_KEY`, and `LITELLM_API_KEY`. The example endpoints point to LangSmith and a LiteLLM proxy on `localhost:4000`. Both values must be full OTLP/HTTP traces URLs

Run from this directory:

```sh
uv run deeplite "Write a three-sentence summary of OpenTelemetry"
```

`uv run` installs the locked dependencies on the first run. The command prints the final answer and sends the agent trace to both endpoints. Open the `deeplite` LangSmith project or LiteLLM's agent tracing UI to inspect the coordinator and subagent runs. LiteLLM requires its `tracing` setting and ClickHouse to ingest OTLP traces
