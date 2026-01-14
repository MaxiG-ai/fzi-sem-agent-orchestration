# agents/statistics_agent.py
import os
from typing import TypedDict, Annotated

import pandas as pd
from dotenv import load_dotenv

from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from orchestrator.observability import setup_observability, get_langfuse_callbacks

load_dotenv()
setup_observability("agent_fzi_test")
LANGFUSE_CALLBACKS = get_langfuse_callbacks()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY fehlt in .env")

DATA_PATH = "data/sample_sensor_data.csv"


def _load_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def _normalize_col(df: pd.DataFrame, col: str) -> str:
    target = col.strip().lower()
    for c in df.columns:
        if c.lower() == target or target in c.lower():
            return c
    raise ValueError(f"Spalte '{col}' nicht gefunden. Verfügbar: {list(df.columns)}")


@tool(description="Gibt den Maximalwert der angegebenen Spalte zurück.")
def max_value(column: str) -> str:
    df = _load_df()
    c = _normalize_col(df, column)
    val = df[c].max()
    return f"Das Maximum von {c} beträgt {val}."


@tool(description="Gibt den Minimalwert der angegebenen Spalte zurück.")
def min_value(column: str) -> str:
    df = _load_df()
    c = _normalize_col(df, column)
    val = df[c].min()
    return f"Das Minimum von {c} beträgt {val}."


@tool(description="Gibt den Mittelwert der angegebenen Spalte zurück.")
def mean_value(column: str) -> str:
    df = _load_df()
    c = _normalize_col(df, column)
    val = df[c].mean()
    return f"Der Mittelwert von {c} beträgt {val}."


@tool(description="Listet Messwerte auf, die oberhalb des Schwellwerts (threshold) liegen.")
def outliers_above(column: str, threshold: float) -> str:
    df = _load_df()
    c = _normalize_col(df, column)
    cols = [c] + (["timestamp"] if "timestamp" in df.columns else [])
    hits = df[df[c] > threshold][cols]
    if hits.empty:
        return f"Keine Werte von {c} über {threshold} gefunden."
    return hits.tail(50).to_json(orient="records", indent=2, date_format="iso")


@tool(description="Listet Messwerte auf, die unterhalb des Schwellwerts (threshold) liegen.")
def outliers_below(column: str, threshold: float) -> str:
    df = _load_df()
    c = _normalize_col(df, column)
    cols = [c] + (["timestamp"] if "timestamp" in df.columns else [])
    hits = df[df[c] < threshold][cols]
    if hits.empty:
        return f"Keine Werte von {c} unter {threshold} gefunden."
    return hits.tail(50).to_json(orient="records", indent=2, date_format="iso")


TOOLS = [max_value, min_value, mean_value, outliers_above, outliers_below]


class State(TypedDict):
    messages: Annotated[list, add_messages]


def _should_continue(state: State):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def _call_model(state: State):
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY,
    ).bind_tools(TOOLS)

    system = SystemMessage(content=f"""\
Du bist ein Statistik-Agent für einen CSV-Datensatz.
Verfügbare Spalten: {", ".join(_load_df().columns)}

Regeln:
- Nutze IMMER ein Tool.
- Wenn Nutzer "Maximum/Minimum/Mittelwert" fragt → passendes Tool.
- Wenn Nutzer "Ausreißer über X / unter X" fragt → outliers_above/outliers_below.
""")

    messages = [system] + state["messages"]

    if LANGFUSE_CALLBACKS:
        resp = model.invoke(messages, config={"callbacks": LANGFUSE_CALLBACKS})
    else:
        resp = model.invoke(messages)

    return {"messages": [resp]}


workflow = StateGraph(State)
workflow.add_node("agent", _call_model)
workflow.add_node("tools", ToolNode(TOOLS))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")
graph = workflow.compile()


@traceable(name="run_statistics_agent")
def run_statistics_agent(user_message: str) -> str:
    initial = {"messages": [HumanMessage(content=user_message)]}

    final = None
    if LANGFUSE_CALLBACKS:
        stream_iter = graph.stream(initial, config={"callbacks": LANGFUSE_CALLBACKS})
    else:
        stream_iter = graph.stream(initial)

    for event in stream_iter:
        final = event

    if final is None:
        return "Keine Antwort gefunden."

    msgs = None
    if "agent" in final:
        msgs = final["agent"]["messages"]
    elif "__end__" in final:
        msgs = final["__end__"]["messages"]

    if not msgs:
        return "Keine Antwort gefunden."

    for m in reversed(msgs):
        if getattr(m, "content", None):
            return m.content

    return "Keine Antwort gefunden."