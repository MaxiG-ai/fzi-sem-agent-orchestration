# agents/plot_agent.py
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from langsmith import traceable
from langsmith.wrappers import wrap_openai

from orchestrator.observability import setup_observability, get_langfuse_observe, log_generation

load_dotenv()
setup_observability("agent_fzi_test")
observe = get_langfuse_observe()

DATA_PATH = "data/sample_sensor_data.csv"

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY fehlt in .env")

# LangSmith wrapper (traces OpenAI calls for LangSmith)
LLM = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))


def load_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def ensure_plots_dir():
    plot_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def normalize_column(df: pd.DataFrame, col: str) -> str:
    col_lower = col.lower().strip()
    for c in df.columns:
        if col_lower == c.lower() or col_lower in c.lower():
            return c
    raise ValueError(f"Spalte '{col}' nicht gefunden. Verfügbar: {list(df.columns)}")


def get_timestamp_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "time" in c.lower():
            return c
    raise ValueError("Keine Zeitspalte gefunden (Spaltenname enthält kein 'time').")


def is_valid_time(value):
    if value is None:
        return False
    cleaned = str(value).strip()
    return cleaned not in ["...", "…", "null", "None", "", " "]


def filter_df(df: pd.DataFrame, args: dict) -> pd.DataFrame:
    df_f = df.copy()
    ts_col = get_timestamp_column(df_f)

    if is_valid_time(args.get("start_time")):
        df_f = df_f[df_f[ts_col] >= pd.to_datetime(args["start_time"], errors="coerce")]

    if is_valid_time(args.get("end_time")):
        df_f = df_f[df_f[ts_col] <= pd.to_datetime(args["end_time"], errors="coerce")]

    if args.get("limit"):
        df_f = df_f.tail(int(args["limit"]))

    return df_f


def plot_time_series(df, column):
    column = normalize_column(df, column)
    ts = get_timestamp_column(df)

    plt.figure(figsize=(10, 5))
    plt.plot(df[ts], df[column])
    plt.title(f"{column} über Zeit")

    path = os.path.join(ensure_plots_dir(), f"time_{column}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return {"path": path}


def plot_histogram(df, column, bins=30):
    column = normalize_column(df, column)

    plt.figure(figsize=(8, 5))
    plt.hist(df[column], bins=bins)
    plt.title(f"Histogramm: {column}")

    path = os.path.join(ensure_plots_dir(), f"hist_{column}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return {"path": path}


def plot_scatter(df, x_col, y_col):
    x_col = normalize_column(df, x_col)
    y_col = normalize_column(df, y_col)

    plt.figure(figsize=(8, 5))
    plt.scatter(df[x_col], df[y_col], s=10)
    plt.title(f"{y_col} vs {x_col}")

    path = os.path.join(ensure_plots_dir(), f"scatter_{x_col}_{y_col}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return {"path": path}


def plot_corr(df):
    numeric = df.select_dtypes(include=["float64", "int64", "float32", "int32"])
    corr = numeric.corr()

    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.title("Korrelationsmatrix")

    path = os.path.join(ensure_plots_dir(), "corr.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return {"path": path}


SYSTEM_PROMPT = """
Du bist ein Plot-Agent für CSV-Sensordaten.

Du gibst IMMER folgendes JSON zurück:

{
  "tool": "<tool>",
  "args": {
    "column": "...",
    "x_col": "...",
    "y_col": "...",
    "limit": <int|null>,
    "start_time": <str|null>,
    "end_time": <str|null>
  }
}

Tool-Namen:
- time_series  (requires: column)
- histogram    (requires: column)
- scatter      (requires: x_col, y_col)
- corr         (requires: none)

Regeln:
- Wenn Nutzer keine Zeit nennt → start_time/end_time = null
- Niemals '...' als Zeit setzen.
"""


@observe(name="plot_agent.llm_decide")
def llm_decide(user_msg: str) -> dict:
    resp = LLM.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    )

    # Langfuse manual generation log (because OpenAI SDK, not LangChain)
    try:
        usage = getattr(resp, "usage", None)
        usage_dict = None
        if usage:
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        out_text = resp.choices[0].message.content or ""
        log_generation(
            name="plot_agent.llm_decide",
            model="gpt-4o-mini",
            input={"system": SYSTEM_PROMPT, "user": user_msg},
            output=out_text,
            usage=usage_dict,
            metadata={"component": "plot_agent"},
        )
    except Exception:
        pass

    content = resp.choices[0].message.content or "{}"
    return json.loads(content)


@traceable(name="run_plot_agent")
@observe(name="plot_agent.run_plot_agent")
def run_plot_agent(user_msg: str) -> dict:
    try:
        plan = llm_decide(user_msg)
        tool = plan.get("tool")
        args = plan.get("args", {}) or {}

        df = load_df()
        df_f = filter_df(df, args)

        if tool == "time_series":
            result = plot_time_series(df_f, args["column"])
        elif tool == "histogram":
            result = plot_histogram(df_f, args["column"], args.get("bins", 30))
        elif tool == "scatter":
            result = plot_scatter(df_f, args["x_col"], args["y_col"])
        elif tool == "corr":
            result = plot_corr(df_f)
        else:
            return {"error": f"Unbekanntes Tool: {tool}", "plan": plan}

        return {"tool": tool, "args": args, "result": result}

    except Exception as e:
        return {"error": str(e)}