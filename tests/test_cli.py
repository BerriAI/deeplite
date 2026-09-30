import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Final
from unittest.mock import Mock

from exa_py import Exa
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from deeplite.cli import Settings, build_agent, configure_tracing, make_search_tool


class ScriptedModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        system: Final = " ".join(message.text for message in messages if isinstance(message, SystemMessage))
        if "Use web_search for factual claims" in system:
            search_results: Final = tuple(message for message in messages if isinstance(message, ToolMessage))
            if search_results:
                response: Final = AIMessage(content=f"Draft answer citing {search_results[-1].text}")
            else:
                response = AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "web_search", "args": {"query": "evidence for the answer"}, "id": "search-task"}
                    ],
                )
        elif "Review the draft" in system:
            response = AIMessage(content="Add the missing detail")
        else:
            completed: Final = sum(isinstance(message, ToolMessage) for message in messages)
            if completed == 0:
                response = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {"subagent_type": "drafter", "description": "Draft an answer to the user's task"},
                            "id": "draft-task",
                        }
                    ],
                )
            elif completed == 1:
                response = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "subagent_type": "reviewer",
                                "description": "Review Draft answer against the original task",
                            },
                            "id": "review-task",
                        }
                    ],
                )
            else:
                response = AIMessage(content="Final answer with the missing detail")
        return ChatResult(generations=[ChatGeneration(message=response)])


def test_subagents_share_one_trace_across_both_otlp_destinations(monkeypatch):
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
        result: Final = build_agent(ScriptedModel(), make_search_tool(exa)).invoke(
            {"messages": [{"role": "user", "content": "Answer a question"}]}
        )
    finally:
        provider.shutdown()
        langsmith.shutdown()
        litellm.shutdown()

    tool_results: Final = tuple(message.text for message in result["messages"] if isinstance(message, ToolMessage))
    assert any("Draft answer" in text for text in tool_results)
    assert any("https://example.com/source" in text for text in tool_results)
    assert any("Add the missing detail" in text for text in tool_results)
    assert result["messages"][-1].text == "Final answer with the missing detail"
    exa.search.assert_called_once_with(
        "evidence for the answer", type="auto", num_results=5, contents={"highlights": True}
    )
    assert received[0] and received[1]
    assert all(path == "/otel/v1/traces" for path, _, _ in received[0])
    assert all(path == "/v1/traces" for path, _, _ in received[1])
    assert all(headers["x-api-key"] == "test-langsmith-key" for _, headers, _ in received[0])
    assert all(headers["Authorization"] == "Bearer test-litellm-key" for _, headers, _ in received[1])
    langsmith_trace_ids: Final = {
        span.trace_id
        for _, _, body in received[0]
        for resource in ExportTraceServiceRequest.FromString(body).resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    }
    litellm_trace_ids: Final = {
        span.trace_id
        for _, _, body in received[1]
        for resource in ExportTraceServiceRequest.FromString(body).resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    }
    assert len(langsmith_trace_ids) == 1
    assert langsmith_trace_ids == litellm_trace_ids
