# orchestrator/router.py
from typing import TypedDict, Annotated, Any

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agents.utils import get_azure_llm
from agents.langfuse_config import get_langfuse_handler, flush_langfuse_handler
from langfuse.decorators import observe

# === Agents importieren ===
from agents.statistics_agent import run_statistics_agent
from agents.plot_agent import run_plot_agent
from agents.physics_agent import run_physics_agent

class RouterState(TypedDict, total=False):
    """
    State for the router graph.
    Using total=False allows optional fields like langfuse_handler.
    """
    messages: Annotated[list, add_messages]
    langfuse_handler: Any  # Optional: Langfuse callback handler

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
    # Setup Langfuse handler with comprehensive metadata
    langfuse_handler = get_langfuse_handler(
        trace_name="router_agent",
        metadata={
            "agent_type": "router",
            "query": query,
            "available_agents": ["statistics_agent", "plot_agent", "physics_agent"],
            "orchestration_type": "langgraph"
        },
        tags=["agent", "router", "orchestration"],
    )
    
    initial = {
        "messages": [
            SystemMessage(
                content="""\
You are an intelligent Router Agent.

Select one of the following agents based on the user query:

1. Statistics Agent -> Max, Min, Outliers
2. Plot Agent -> Diagrams, Charts, Visualization
3. Physics Agent -> Physical relationships, formulas, correlations

ALWAYS use a tool.
"""
            ),
            HumanMessage(content=query),
        ],
        "langfuse_handler": langfuse_handler,
    }

    # Configure callbacks for the graph execution
    config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
    
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
            # Flush Langfuse handler
            flush_langfuse_handler(langfuse_handler)
            return result

    # Flush Langfuse handler
    flush_langfuse_handler(langfuse_handler)
    return "Keine Antwort gefunden."


if __name__ == "__main__":
    print("Router Agent Test")
    run_router("How is the correlation between temperature and pressure?")