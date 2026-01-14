# agents/physics_agent.py
import os
import json
import pandas as pd
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from functools import wraps

from langsmith import traceable
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from orchestrator.observability import setup_observability, get_langfuse_callbacks, get_langfuse_observe

load_dotenv()
setup_observability("agent_fzi_test")
LANGFUSE_CALLBACKS = get_langfuse_callbacks()
observe = get_langfuse_observe()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY fehlt in .env")


# -----------------------------------------------------------------------------
# Helper: observe decorator that preserves __name__/__doc__ (important for @tool)
# -----------------------------------------------------------------------------
def observe_keep_meta(name: str):
    """
    Wraps langfuse observe decorator, but preserves function metadata so
    langchain @tool does not complain about missing docstrings.
    """
    def deco(fn):
        decorated = observe(name=name)(fn)

        @wraps(fn)
        def _wrapped(*args, **kwargs):
            return decorated(*args, **kwargs)

        # ensure original docstring/name survive
        _wrapped.__doc__ = fn.__doc__
        _wrapped.__name__ = fn.__name__
        return _wrapped

    return deco


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    data_json: str


_current_data_json: str | None = None


def _df_from_json(data_json: str) -> pd.DataFrame:
    try:
        return pd.read_json(data_json)
    except ValueError:
        from io import StringIO
        return pd.read_json(StringIO(data_json))


@tool(description="Berechnet Pearson-Korrelationen für ausgewählte Spalten (leer = alle numerischen).")
@observe_keep_meta(name="physics.calculate_correlations")
def calculate_correlations(columns: list[str]) -> str:
    """Berechnet Pearson-Korrelationen für ausgewählte Spalten (leer = alle numerischen)."""
    global _current_data_json
    if not _current_data_json:
        return json.dumps({"error": "No data loaded."}, indent=2)

    try:
        df = _df_from_json(_current_data_json)

        if not columns:
            corr_matrix = df.corr(numeric_only=True)
        else:
            available = [c for c in columns if c in df.columns]
            if not available:
                return json.dumps(
                    {"error": "No valid columns", "available": df.columns.tolist()},
                    indent=2,
                )
            corr_matrix = df[available].corr()

        result = {}
        for c1 in corr_matrix.columns:
            result[c1] = {}
            for c2 in corr_matrix.columns:
                if c1 != c2:
                    result[c1][c2] = round(float(corr_matrix.loc[c1, c2]), 4)

        return json.dumps({"correlations": result}, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@tool(description="Sucht passende Physik-Formeln zu den Feldern (kleine Wissensbasis).")
@observe_keep_meta(name="physics.lookup_physics_formula")
def lookup_physics_formula(fields: list[str]) -> str:
    """Sucht passende Physik-Formeln zu den Feldern (kleine Wissensbasis)."""
    formula_db = {
        ("mass_flow", "volume_flow", "density"): {
            "formula": "mass_flow = density × volume_flow",
            "context": "Fundamentaler Zusammenhang: Massenstrom = Dichte * Volumenstrom.",
            "units": "kg/s = (kg/m³) × (m³/s)",
        },
        ("temperature", "density"): {
            "formula": "density ≈ ρ₀ × (1 - β × ΔT)",
            "context": "Thermische Ausdehnung: Dichte sinkt oft mit steigender Temperatur.",
            "units": "β in K⁻¹",
        },
    }

    norm = tuple(sorted([f.lower() for f in fields]))
    hits = []
    for k, v in formula_db.items():
        if set(norm).issubset(set(k)):
            hits.append(v)

    if hits:
        return json.dumps({"formulas": hits}, indent=2)

    return json.dumps({"formulas": [{"formula": "No direct formula found", "fields": fields}]}, indent=2)


TOOLS = [calculate_correlations, lookup_physics_formula]


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def call_model(state: AgentState):
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY,
    ).bind_tools(TOOLS)

    if LANGFUSE_CALLBACKS:
        resp = model.invoke(state["messages"], config={"callbacks": LANGFUSE_CALLBACKS})
    else:
        resp = model.invoke(state["messages"])

    return {"messages": [resp]}


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(TOOLS))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")
graph = workflow.compile()


@traceable(name="run_physics_agent")
@observe(name="physics.run_physics_agent")
def run_physics_agent(user_message: str, df: pd.DataFrame, max_iterations: int = 10) -> str:
    global _current_data_json
    _current_data_json = df.to_json()

    system_prompt = f"""\
You are a physics-aware data analysis assistant for industrial sensor data.

Available columns: {', '.join(df.columns.tolist())}

CRITICAL:
- If the user asks about relationships/dependencies/consistency:
  1) call calculate_correlations on relevant columns
  2) call lookup_physics_formula on relevant fields
- Do not invent numeric values. Use tool output.
- Keep final answer concise.
"""

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ],
        "data_json": _current_data_json,
    }

    final = None

    # ✅ LangGraph: recursion_limit kommt in config (nicht als 2. positional dict!)
    config = {"recursion_limit": max_iterations}
    if LANGFUSE_CALLBACKS:
        config["callbacks"] = LANGFUSE_CALLBACKS

    for event in graph.stream(initial_state, config=config):
        final = event

    if final is None:
        return "No response."

    msgs = None
    if "agent" in final:
        msgs = final["agent"]["messages"]
    elif "__end__" in final:
        msgs = final["__end__"]["messages"]

    if not msgs:
        return "No response."

    for m in reversed(msgs):
        if getattr(m, "content", None):
            return m.content

    return "No response."