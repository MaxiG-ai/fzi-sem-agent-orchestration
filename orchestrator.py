# orchestrator/router.py
import os
from datetime import datetime
from dotenv import load_dotenv
from typing import TypedDict, Annotated, cast
from langchain_core.runnables import RunnableConfig

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agents.utils import get_azure_llm
from agents.langfuse_config import get_langfuse_handler
from langfuse import observe, get_client, Evaluation, propagate_attributes

# === Agents importieren ===
from agents.statistics_agent import run_statistics_agent
from agents.plot_agent import run_plot_agent
from agents.physics_agent import run_physics_agent

class RouterState(TypedDict):
    """
    State for the router graph.
    messages is required, langfuse_handler is optional (accessed via .get()).
    """
    messages: Annotated[list, add_messages]
    # Note: langfuse_handler is optional and accessed via state.get("langfuse_handler")

# TOOLS (Wrapper for the sub-agents)
@tool
def call_statistics_agent(query: str) -> str:
    """Calls the Statistics Agent to calculate max, min, or outliers."""
    return str(run_statistics_agent(query))

@tool
def call_plot_agent(query: str) -> str:
    """Calls the Plot Agent to generate charts and diagrams."""
    return str(run_plot_agent(query))

@tool
def call_physics_agent(query: str) -> str:
    """Calls the Physics Agent to analyze physical relationships and correlations."""
    return str(run_physics_agent(query))


router_tools = [call_statistics_agent, call_plot_agent, call_physics_agent]

# ROUTER MODEL

def router_decision(state: RouterState):
    # Get Langfuse handler from state if available
    langfuse_handler = state.get("langfuse_handler")
    callbacks = [langfuse_handler] if langfuse_handler else None

    model = get_azure_llm(callbacks=callbacks)
    model = model.bind_tools(router_tools)

    response = model.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: RouterState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END  

# BUILD GRAPH

graph = StateGraph(RouterState)

graph.add_node("router", router_decision)
graph.add_node("tools", ToolNode(router_tools))

graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    should_continue,
    {
        "tools": "tools",
        END: END,
    }
)
graph.add_edge("tools", "router")

router_graph = graph.compile()

# PUBLIC RUN FUNCTION

@observe()
def run_router(query: str, system_prompt: str) -> str:
    """
    Run the router agent with Langfuse tracking.

    Args:
        query: The user's query
        system_prompt: The system prompt to use

    Returns:
        The agent's response
    """
    # Setup Langfuse handler
    langfuse_handler = get_langfuse_handler()

    initial = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ],
        "langfuse_handler": langfuse_handler,
    }

    # Configure callbacks and metadata for the graph execution (v3 style)
    config = {
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_run_name": "router_agent",
            "agent_type": "router",
            "query": query,
            "available_agents": ["statistics_agent", "plot_agent", "physics_agent"],
            "orchestration_type": "langgraph",
        }
    } if langfuse_handler else {}

    final_state = {}
    for event in router_graph.stream(cast(RouterState, initial), config=cast(RunnableConfig, config)):
        final_state = event

    messages = []
    if final_state and "router" in final_state:
        messages = final_state["router"]["messages"]
    elif final_state and "__end__" in final_state:
        messages = final_state["__end__"]["messages"]

    for m in reversed(messages):
        if hasattr(m, "content") and m.content:
            result = m.content
            return result

    return "Keine Antwort gefunden."


if __name__ == "__main__":
    load_dotenv()
    langfuse = get_client()
    print("Router Agent Experiment")
    
    # 1. Get Dataset
    dataset = langfuse.get_dataset("fzi_sem_ds")
    
    # 2. Get Prompt
    prompt = langfuse.get_prompt("router_agent")
    compiled_prompt = prompt.compile()

    # 3. Define Task
    def experiment_task(*, item, **kwargs):
        return run_router(item.input, compiled_prompt)

    # 4. Define Evaluator
    def basic_evaluator(*, input, output, expected_output, **kwargs):
        return Evaluation(
            name="has_response",
            value=1 if output and len(output) > 5 else 0
        )

    # 5. Run Experiment
    print("Running experiment...")
    with propagate_attributes(session_id=f"experiment-{datetime.now().strftime('%Y-%m-%d_%H:%M')}"):
        langfuse.run_experiment(
            name="router_experiment_v1",
            data=dataset.items[::5],
            task=experiment_task,
            evaluators=[basic_evaluator]
        )
