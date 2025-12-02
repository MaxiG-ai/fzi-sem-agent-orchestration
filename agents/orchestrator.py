# orchestrator/router.py

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# === Agents importieren ===
from agents.statistics_agent import run_statistics_agent
from agents.plot_agent import run_plot_agent
from agents.physics_agent import run_physics_agent

# Load environment variables from .env file
load_dotenv()

AZURE_AI_CREDENTIAL = os.getenv("AZURE_AI_CREDENTIAL")
AZURE_AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT")
AZURE_AI_MODEL_NAME = os.getenv("AZURE_AI_MODEL_NAME")
AZURE_AI_DEPLOYMENT = os.getenv("AZURE_AI_DEPLOYMENT", "o4-mini")
AZURE_AI_API_VERSION = os.getenv("AZURE_AI_API_VERSION", "2024-12-01-preview")

if not AZURE_AI_CREDENTIAL:
    raise ValueError(
        "AZURE_AI_CREDENTIAL not found in .env file. Please create a .env file with your API key."
    )


class RouterState(TypedDict):
    messages: Annotated[list, add_messages]

# TOOLS (SUB-AGENTS)

@tool
def call_statistics_agent(query: str) -> str:
    """Ruft den Statistik-Agenten auf und gibt dessen Ergebnis zurück."""
    return str(run_statistics_agent(query))


@tool
def call_plot_agent(query: str) -> str:
    """Ruft den Plot-Agenten auf und erzeugt ein Diagramm."""
    return str(run_plot_agent(query))


@tool
def call_physics_agent(query: str) -> str:
    """Ruft den Physik-Agenten auf."""
    return str(run_physics_agent(query))


router_tools = [call_statistics_agent, call_plot_agent, call_physics_agent]

# ROUTER MODEL

def router_decision(state: RouterState):
    model = AzureChatOpenAI(
        azure_deployment=AZURE_AI_DEPLOYMENT,
        model_name=AZURE_AI_MODEL_NAME,
        api_version=AZURE_AI_API_VERSION,
        azure_endpoint=AZURE_AI_ENDPOINT,
        api_key=AZURE_AI_CREDENTIAL,
    )
    
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

def run_router(query: str) -> str:
    initial = {
        "messages": [
            SystemMessage(content="""\
Du bist ein intelligenter Router-Agent.

Wähle basierend auf der Nutzeranfrage einen der folgenden Agents:

1. Statistik-Agent → Max, Min, Ausreißer
2. Plot-Agent → Diagramme
3. Physik-Agent → physikalische Zusammenhänge

Nutze IMMER ein Tool.
"""),
            HumanMessage(content=query)
        ]
    }

    final_state = None
    for event in router_graph.stream(initial):
        final_state = event

    messages = None
    if "router" in final_state:
        messages = final_state["router"]["messages"]
    elif "__end__" in final_state:
        messages = final_state["__end__"]["messages"]

    for m in reversed(messages):
        if hasattr(m, "content") and m.content:
            return m.content

    return "Keine Antwort gefunden."

# CLI MODE für die bisschen bessere Bedienbarkeit :) 

def main():
    print("🚀 Router gestartet – stelle deine Fragen!")
    print("Gib 'exit' ein zum Beenden.\n")

    while True:
        user = input("> ").strip()
        if user.lower() in ("exit", "quit"):
            break

        try:
            print("\n🧠 Antwort:")
            print(run_router(user))
        except Exception as e:
            print("\n❌ Fehler:", e)


if __name__ == "__main__":
    main()