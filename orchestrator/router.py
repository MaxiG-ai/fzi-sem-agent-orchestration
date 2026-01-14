# orchestrator/router.py
import os

from orchestrator.observability import setup_observability, get_langfuse_callbacks
setup_observability("agent_fzi_test")
LANGFUSE_CALLBACKS = get_langfuse_callbacks()

import pandas as pd
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langsmith import traceable
from orchestrator.observability import setup_observability, get_langfuse_callbacks, flush_langfuse

load_dotenv()
setup_observability("agent_fzi_test")
LANGFUSE_CALLBACKS = get_langfuse_callbacks()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY fehlt in .env")

from agents.statistics_agent import run_statistics_agent
from agents.plot_agent import run_plot_agent
from agents.physics_agent import run_physics_agent


class RouterState(TypedDict):
    messages: Annotated[list, add_messages]


# =============================================================================
# Tools = Agent Calls
# =============================================================================
@tool(description="Leitet die Anfrage an den Statistik-Agenten weiter (CSV-basiert).")
def call_statistics_agent(query: str) -> str:
    return str(run_statistics_agent(query))


@tool(description="Leitet die Anfrage an den Plot-Agenten weiter (CSV-basiert).")
def call_plot_agent(query: str) -> str:
    return str(run_plot_agent(query))


@tool(description="Leitet die Anfrage an den Physik-Agenten weiter (CSV-basiert, datengetrieben).")
def call_physics_agent(query: str) -> str:
    df = pd.read_csv("data/sample_sensor_data.csv")
    return str(run_physics_agent(query, df))


router_tools = [call_statistics_agent, call_plot_agent, call_physics_agent]


def router_decision(state: RouterState):
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY,
    ).bind_tools(router_tools)

    if LANGFUSE_CALLBACKS:
        response = model.invoke(state["messages"], config={"callbacks": LANGFUSE_CALLBACKS})
    else:
        response = model.invoke(state["messages"])

    return {"messages": [response]}


def should_continue(state: RouterState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


graph = StateGraph(RouterState)
graph.add_node("router", router_decision)
graph.add_node("tools", ToolNode(router_tools))

graph.add_edge(START, "router")
graph.add_conditional_edges("router", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "router")

router_graph = graph.compile()


@traceable(name="run_router")
def run_router(query: str) -> str:
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
        ]
    }

    final_state = None
    if LANGFUSE_CALLBACKS:
        stream_iter = router_graph.stream(initial, config={"callbacks": LANGFUSE_CALLBACKS})
    else:
        stream_iter = router_graph.stream(initial)

    for event in stream_iter:
        final_state = event

    if final_state is None:
        return "Keine Antwort gefunden (final_state=None)."

    messages = None
    if "router" in final_state:
        messages = final_state["router"]["messages"]
    elif "__end__" in final_state:
        messages = final_state["__end__"]["messages"]

    if not messages:
        return "Keine Antwort gefunden (messages leer)."

    for m in reversed(messages):
        if getattr(m, "content", None):
            return m.content

    return "Keine Antwort gefunden."


def main():
    print("🚀 Router gestartet – stelle deine Fragen!")
    print("🧩 Langfuse:", "aktiv ✅" if LANGFUSE_CALLBACKS else "nicht aktiv")
    print("Gib 'exit' ein, um zu beenden.\n")

    try:
        while True:
            user = input("> ").strip()
            if user.lower() in ("exit", "quit", "stop"):
                print("👋 Router beendet.")
                break
            ans = run_router(user)
            print("\n🧠 Antwort:\n", ans, "\n")
    finally:
        flush_langfuse()


if __name__ == "__main__":
    main()