# orchestrator/router.py
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated



from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agents.utils import get_azure_llm
from agents.langfuse_config import get_langfuse_handler
from langfuse import observe, get_client

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
def run_router(query: str) -> str:
    """
    Run the router agent with Langfuse tracking.

    Args:
        query: The user's query

    Returns:
        The agent's response
    """
    # Setup Langfuse handler
    langfuse_handler = get_langfuse_handler()

    initial = {
        "messages": [
            SystemMessage(content="""\
Du bist ein intelligenter Router-Agent.

Wähle basierend auf der Nutzeranfrage einen der folgenden Agents:
1) Statistik-Agent → Max, Min, Mittelwert, Ausreißer, Kennzahlen
2) Plot-Agent → Zeitreihen, Histogramm, Scatter, Korrelationsmatrix
3) Physik-Agent → datenbasierte Korrelationen + physikalische Beziehung/Interpretation

Nutze IMMER ein Tool.
"""),
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

    final_state = None
    for event in router_graph.stream(initial, config=config):
        final_state = event

    messages = None
    if "router" in final_state:
        messages = final_state["router"]["messages"]
    elif "__end__" in final_state:
        messages = final_state["__end__"]["messages"]

    for m in reversed(messages):
        if hasattr(m, "content") and m.content:
            result = m.content
            return result

    return "Keine Antwort gefunden."


if __name__ == "__main__":
    load_dotenv()
    langfuse = get_client()
    print("Router Agent Test")
    dataset = langfuse.get_dataset("fzi_sem_ds")
    print(langfuse.get_prompt("router_agent"))
    for i, item in enumerate(dataset.items):
        if i >= 5:
            break
        run_router(item.input)