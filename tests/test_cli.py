import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Final
from unittest.mock import Mock

from deepagents.backends.store import StoreBackend
from exa_py import Exa
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.store.memory import InMemoryStore
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from deeplite.agent import build_agent, make_filesystem_middleware, make_search_tool
from deeplite.cli import run_task
from deeplite.config import Settings
from deeplite.tracing import configure_tracing


class ScriptedModel(BaseChatModel):
    stop_at_red_team: bool = False

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        system: Final = " ".join(message.text for message in messages if isinstance(message, SystemMessage))
        history: Final = " ".join(message.text for message in messages)
        if "You are the researcher" in system:
            if any(isinstance(message, ToolMessage) and message.tool_call_id == "research-search" for message in messages):
                response: Final = AIMessage(
                    content="Research cites https://example.com/source",
                    tool_calls=[{"name": "transfer_to_skeptic", "args": {}, "id": "research-handoff"}],
                )
            else:
                response = AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "web_search", "args": {"query": "research evidence"}, "id": "research-search"}
                    ],
                )
        elif "You are the skeptic" in system:
            assert "Research cites https://example.com/source" in history
            response = AIMessage(
                content="Skeptic asks for independent verification",
                tool_calls=[{"name": "transfer_to_verifier", "args": {}, "id": "skeptic-handoff"}],
            )
        elif "You are the verifier" in system:
            assert "Skeptic asks for independent verification" in history
            if any(isinstance(message, ToolMessage) and message.tool_call_id == "verify-search" for message in messages):
                response = AIMessage(
                    content="Verifier confirms the source",
                    tool_calls=[{"name": "transfer_to_red_team", "args": {}, "id": "verifier-handoff"}],
                )
            else:
                response = AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "web_search", "args": {"query": "verify evidence"}, "id": "verify-search"}
                    ],
                )
        elif "You are red_team" in system:
            assert "Verifier confirms the source" in history
            response = (
                AIMessage(content="Red team answer cites https://example.com/source")
                if self.stop_at_red_team
                else AIMessage(
                    content="Red team finds no remaining objection",
                    tool_calls=[{"name": "transfer_to_editor", "args": {}, "id": "red-team-handoff"}],
                )
            )
        else:
            assert "Red team finds no remaining objection" in history
            response = AIMessage(content="Final answer cites https://example.com/source")
        return ChatResult(generations=[ChatGeneration(message=response)])


def test_swarm_handoffs_share_context_and_one_trace_across_both_otlp_destinations(monkeypatch):
    received: Final = ([], [])

    def start_server(index: int):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length: Final = int(self.headers["Content-Length"])
                received[index].append((self.path, self.headers, self.rfile.read(length)))
                self.send_response(200)
                self.end_headers()

            def log_message(self, format, *args):
                pass

        server: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    langsmith: Final = start_server(0)
    litellm: Final = start_server(1)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_TRACING_MODE", "otel")
    settings: Final = Settings(
        model="scripted",
        gateway_url="https://gateway.example.com",
        exa_key="test-exa-key",
        langsmith_endpoint=f"http://127.0.0.1:{langsmith.server_port}/otel/v1/traces",
        langsmith_key="test-langsmith-key",
        langsmith_project="deeplite-test",
        litellm_endpoint=f"http://127.0.0.1:{litellm.server_port}/v1/traces",
        litellm_key="test-litellm-key",
    )
    provider: Final = configure_tracing(settings)
    exa: Final = Mock(spec=Exa)
    exa.search.return_value = SimpleNamespace(
        results=[SimpleNamespace(title="Source", url="https://example.com/source", highlights=["Relevant finding"])]
    )
    try:
        answer: Final = run_task(build_agent(ScriptedModel(), make_search_tool(exa)), "Answer a question")
    finally:
        provider.shutdown()
        langsmith.shutdown()
        litellm.shutdown()

    assert answer == "Final answer cites https://example.com/source"
    assert exa.search.call_count == 2
    assert [call.args[0] for call in exa.search.call_args_list] == ["research evidence", "verify evidence"]
    assert received[0] and received[1]
    assert all(path == "/otel/v1/traces" for path, _, _ in received[0])
    assert all(path == "/v1/traces" for path, _, _ in received[1])
    assert all(headers["x-api-key"] == "test-langsmith-key" for _, headers, _ in received[0])
    assert all(headers["Authorization"] == "Bearer test-litellm-key" for _, headers, _ in received[1])
    langsmith_spans: Final = tuple(
        span
        for _, _, body in received[0]
        for resource in ExportTraceServiceRequest.FromString(body).resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    )
    litellm_spans: Final = tuple(
        span
        for _, _, body in received[1]
        for resource in ExportTraceServiceRequest.FromString(body).resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    )
    assert len({span.trace_id for span in langsmith_spans}) == 1
    assert {span.trace_id for span in langsmith_spans} == {span.trace_id for span in litellm_spans}
    span_ids: Final = frozenset(span.span_id for span in langsmith_spans)
    assert all(not span.parent_span_id or span.parent_span_id in span_ids for span in langsmith_spans)
    assert {
        "transfer_to_skeptic",
        "transfer_to_verifier",
        "transfer_to_red_team",
        "transfer_to_editor",
    } <= {span.name for span in langsmith_spans}


def test_swarm_can_finish_with_a_non_editor_agent():
    exa: Final = Mock(spec=Exa)
    exa.search.return_value = SimpleNamespace(
        results=[SimpleNamespace(title="Source", url="https://example.com/source", highlights=["Relevant finding"])]
    )
    agent: Final = build_agent(ScriptedModel(stop_at_red_team=True), make_search_tool(exa))

    assert run_task(agent, "Answer a question") == "Red team answer cites https://example.com/source"


def test_only_editor_writes_shared_virtual_files():
    store: Final = InMemoryStore()
    editor_backend: Final = StoreBackend(store=store, namespace=lambda _runtime: ("deeplite",))
    reader_backend: Final = StoreBackend(store=store, namespace=lambda _runtime: ("deeplite",))
    editor_tools: Final = {tool.name for tool in make_filesystem_middleware(editor_backend, writable=True)[0].tools}
    reader_tools: Final = {tool.name for tool in make_filesystem_middleware(reader_backend, writable=False)[0].tools}

    assert {"read_file", "write_file", "edit_file"} <= editor_tools
    assert "read_file" in reader_tools
    assert not {"write_file", "edit_file", "delete", "execute"} & reader_tools
    assert editor_backend.write("/answer.md", "Final shared answer").error is None
    answer: Final = reader_backend.read("/answer.md")
    assert answer.file_data is not None
    assert answer.file_data["content"] == "Final shared answer"
