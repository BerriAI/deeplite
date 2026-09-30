import json
from typing import Final

from deepagents.backends.store import StoreBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.unsupported_content import UnsupportedContentMiddleware
from exa_py import Exa
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph_swarm import create_handoff_tool, create_swarm


def make_search_tool(exa: Exa) -> BaseTool:
    @tool("web_search", description="Search the web with Exa for current facts and return excerpts with source URLs")
    def web_search(query: str) -> str:
        result: Final = exa.search(query, type="auto", num_results=5, contents={"highlights": True})
        return json.dumps(
            [{"title": item.title, "url": item.url, "highlights": item.highlights} for item in result.results]
        )

    return web_search


def make_filesystem_middleware(
    backend: StoreBackend, writable: bool
) -> tuple[FilesystemMiddleware, UnsupportedContentMiddleware]:
    tools: Final = ["ls", "read_file", "glob", "grep", "write_file", "edit_file"] if writable else [
        "ls", "read_file", "glob", "grep"
    ]
    return (
        FilesystemMiddleware(
            backend=backend,
            tools=tools,
            tool_token_limit_before_evict=None,
            human_message_token_limit_before_evict=None,
        ),
        UnsupportedContentMiddleware(),
    )


def build_agent(
    model: BaseChatModel, search_tool: BaseTool, file_store: BaseStore | None = None
) -> CompiledStateGraph:
    store: Final = file_store if file_store is not None else InMemoryStore()
    backend: Final = StoreBackend(store=store, namespace=lambda _runtime: ("deeplite",))

    researcher: Final = create_agent(
        model,
        middleware=make_filesystem_middleware(backend, writable=False),
        tools=[
            search_tool,
            create_handoff_tool(agent_name="skeptic", description="Send findings to the skeptic for critique"),
            create_handoff_tool(agent_name="verifier", description="Send revised findings for source verification"),
        ],
        system_prompt=(
            "You are the researcher. Search for evidence and share source URLs. "
            "You may read shared virtual files, but only the editor writes them. "
            "Send initial findings to the skeptic. If another agent returns with corrections, "
            "revise your findings and send them to the verifier. Do not give the final answer."
        ),
        name="researcher",
    )
    skeptic: Final = create_agent(
        model,
        middleware=make_filesystem_middleware(backend, writable=False),
        tools=[
            create_handoff_tool(agent_name="researcher", description="Ask the researcher to address a concrete gap"),
            create_handoff_tool(agent_name="verifier", description="Send critique for independent verification"),
            create_handoff_tool(agent_name="red_team", description="Send checked claims for adversarial review"),
        ],
        system_prompt=(
            "You are the skeptic. Challenge the research, name unsupported claims, and send your critique "
            "to the verifier. If the editor sends a revision back, ask the researcher to fix concrete gaps "
            "or send the result to red_team. Do not give the final answer."
        ),
        name="skeptic",
    )
    verifier: Final = create_agent(
        model,
        middleware=make_filesystem_middleware(backend, writable=False),
        tools=[
            search_tool,
            create_handoff_tool(agent_name="researcher", description="Request a correction from the researcher"),
            create_handoff_tool(agent_name="red_team", description="Send verified evidence for adversarial review"),
            create_handoff_tool(agent_name="editor", description="Send verified findings to the editor"),
        ],
        system_prompt=(
            "You are the verifier. Independently search to check claims and source URLs. "
            "If evidence is weak, send the issue to the researcher. Otherwise send your verdict to red_team. "
            "Do not give the final answer."
        ),
        name="verifier",
    )
    red_team: Final = create_agent(
        model,
        middleware=make_filesystem_middleware(backend, writable=False),
        tools=[
            create_handoff_tool(agent_name="skeptic", description="Return an unresolved objection to the skeptic"),
            create_handoff_tool(agent_name="editor", description="Send adversarial findings to the editor"),
        ],
        system_prompt=(
            "You are red_team. Find the strongest remaining objection to the verified findings. "
            "Send unresolved issues to the skeptic, or send your assessment to the editor. "
            "Do not give the final answer."
        ),
        name="red_team",
    )
    editor: Final = create_agent(
        model,
        middleware=make_filesystem_middleware(backend, writable=True),
        tools=[
            create_handoff_tool(agent_name="researcher", description="Request missing evidence from the researcher"),
            create_handoff_tool(agent_name="skeptic", description="Request another critique from the skeptic"),
            create_handoff_tool(agent_name="verifier", description="Request another source check from the verifier"),
        ],
        system_prompt=(
            "You are the editor. Use the shared conversation to write one concise answer with source URLs. "
            "Write the final answer to /answer.md in the shared virtual filesystem before replying. "
            "If important issues remain, hand off to the right agent before answering."
        ),
        name="editor",
    )
    return create_swarm(
        [researcher, skeptic, verifier, red_team, editor],
        default_active_agent="researcher",
    ).compile()
