import argparse
from typing import Final

from exa_py import Exa
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tracers.langchain import wait_for_all_tracers
from opentelemetry import trace

from deeplite.agent import build_agent, make_search_tool
from deeplite.config import configure
from deeplite.tracing import configure_tracing


def run_task(agent: Runnable, task: str) -> str:
    tracer: Final = trace.get_tracer("deeplite")
    with tracer.start_as_current_span("deeplite.run") as span:
        span.set_attribute("langsmith.span.kind", "chain")
        span.set_attribute("input.value", task)
        try:
            result: Final = agent.invoke(
                {"messages": [{"role": "user", "content": task}]}, config={"recursion_limit": 40}
            )
            answer: Final = next(message for message in reversed(result["messages"]) if isinstance(message, AIMessage))
            span.set_attribute("output.value", answer.text)
            return answer.text
        finally:
            wait_for_all_tracers()


def main() -> None:
    parser: Final = argparse.ArgumentParser(description="Run a LangGraph swarm with OTLP tracing")
    parser.add_argument("task", help="Task for the agent")
    args: Final = parser.parse_args()
    try:
        settings: Final = configure()
    except ValueError as error:
        parser.error(str(error))
    provider: Final = configure_tracing(settings)
    try:
        model: Final = ChatAnthropic(
            model=settings.model,
            api_key=settings.prod_key,
            base_url=settings.prod_base,
        )
        agent: Final = build_agent(model, make_search_tool(Exa(api_key=settings.exa_key)))
        print(run_task(agent, args.task))
    finally:
        provider.shutdown()


if __name__ == "__main__":
    main()
