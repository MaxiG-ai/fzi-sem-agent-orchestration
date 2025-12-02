import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool

from data.sp_data import load_sensor_data_from_csv

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

LLM = AzureChatOpenAI(
    azure_deployment=AZURE_AI_DEPLOYMENT,
    model_name=AZURE_AI_MODEL_NAME,
    api_version=AZURE_AI_API_VERSION,
    azure_endpoint=AZURE_AI_ENDPOINT,
    api_key=AZURE_AI_CREDENTIAL,
)

DEFAULT_MEASURE = "sensor_data"  # bleibt für spätere Erweiterungen


def ensure_plots_dir():
    plot_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

# COLUMN NORMALIZATION

def normalize_column(df: pd.DataFrame, col: str) -> str:
    col_lower = col.lower()
    for c in df.columns:
        if col_lower == c.lower() or col_lower in c.lower():
            return c
    raise ValueError(f"Spalte '{col}' nicht gefunden. Verfügbar: {list(df.columns)}")


def get_timestamp_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "time" in c.lower():
            return c
    raise ValueError("Keine Zeitspalte gefunden.")

# TIME VALIDATION

def is_valid_time(value):
    if not value:
        return False
    cleaned = str(value).strip()
    if cleaned in ["...", "…", "null", "None", "", " "]:
        return False
    return True

# Filtern

def filter_df(df: pd.DataFrame, args: dict) -> pd.DataFrame:
    df_f = df.copy()

    if is_valid_time(args.get("start_time")):
        df_f = df_f[df_f["timestamp"] >= pd.to_datetime(args["start_time"])]

    if is_valid_time(args.get("end_time")):
        df_f = df_f[df_f["timestamp"] <= pd.to_datetime(args["end_time"])]

    if args.get("limit"):
        df_f = df_f.tail(int(args["limit"]))

    return df_f

# PLOT FUNCTIONS

@tool
def plot_time_series(df, column):
    """Plot time series from a df for a column

    Args:
        df (pd.DataFrame): DataFrame containing the data
        column (str): Column to plot

    Returns:
        dict: Dictionary with the path to the saved plot image
    """
    column = normalize_column(df, column)
    ts = get_timestamp_column(df)

    plt.figure(figsize=(10, 5))
    plt.plot(df[ts], df[column])
    plt.title(f"{column} über Zeit")

    path = os.path.join(ensure_plots_dir(), f"time_{column}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return {"path": path}


@tool
def plot_histogram(df: pd.DataFrame, column: str, bins: int = 30):
    """Plot histogram for a specific column from a df.

    Args:
        df (pd.DataFrame): DataFrame containing the data
        column (str): Column to plot
        bins (int, optional): Number of bins for the histogram. Defaults to 30.

    Returns:
        dict: Dictionary with the path to the saved plot image
    """
    column = normalize_column(df, column)

    plt.figure(figsize=(8, 5))
    plt.hist(df[column], bins=bins)
    plt.title(f"Histogramm für {column}")

    path = os.path.join(ensure_plots_dir(), f"hist_{column}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return {"path": path}


@tool
def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str):
    """Plot scatter plot for two columns from a df.
    Args:
        df (pd.DataFrame): DataFrame containing the data
        x_col (str): Column for x-axis
        y_col (str): Column for y-axis
        
    Returns: 
        dict: Dictionary with the path to the saved plot image
    """
    x_col = normalize_column(df, x_col)
    y_col = normalize_column(df, y_col)

    plt.figure(figsize=(8, 5))
    plt.scatter(df[x_col], df[y_col], s=10)
    plt.title(f"{y_col} vs {x_col}")

    path = os.path.join(ensure_plots_dir(), f"scatter_{x_col}_{y_col}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return {"path": path}


@tool
def plot_corr(df: pd.DataFrame):
    """Plot correlation matrix for numeric columns in a df.

    Args:
        df (pd.DataFrame): DataFrame containing the data

    Returns:
        dict: Dictionary with the path to the saved plot image
    """
    numeric = df.select_dtypes(include=["float64", "int64"])
    corr = numeric.corr()

    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.title("Korrelationsmatrix")

    path = os.path.join(ensure_plots_dir(), "corr_matrix.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return {"path": path}

# LLM SYSTEM PROMPT

SYSTEM_PROMPT = """
Du bist ein Datenanalyse-Agent für CSV-Sensordaten.

FILTEROPTIONEN:
- limit
- start_time
- end_time

Du gibst IMMER folgendes JSON zurück:

{
  "tool": "<tool>",
  "args": {
    "column": "...",
    "limit": <int>,
    "start_time": "...",
    "end_time": "..."
  }
}

Regeln:
- column = Spalte im CSV
- Valid tools: time_series, histogram, scatter, corr
- Nutzer muss keine Zeiten angeben
- Niemals Dummy-Zeichen wie "..." als Zeit interpretieren
"""

def llm_decide(user_msg: str) -> dict:
    model = AzureChatOpenAI(
        azure_deployment=AZURE_AI_DEPLOYMENT,
        model_name=AZURE_AI_MODEL_NAME,
        api_version=AZURE_AI_API_VERSION,
        azure_endpoint=AZURE_AI_ENDPOINT,
        api_key=AZURE_AI_CREDENTIAL,
    )
    
    model = model.bind_tools([
        plot_time_series,
        plot_histogram,
        plot_scatter,
        plot_corr,
    ])
    
    response = model.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ])
    
    return json.loads(response)

# ROUTER

def route_tool(tool: str, args: dict):
    df = load_sensor_data_from_csv("data/sample_sensor_data.csv")
    df_filtered = filter_df(df, args)

    if tool == "time_series":
        return plot_time_series(df_filtered, args["column"])

    if tool == "histogram":
        return plot_histogram(df_filtered, args["column"], args.get("bins", 30))

    if tool == "scatter":
        return plot_scatter(df_filtered, args["x_col"], args["y_col"])

    if tool == "corr":
        return plot_corr(df_filtered)

    raise ValueError(f"Unbekanntes Tool: {tool}")

# PUBLIC ENTRY — für rouer


def run_plot_agent(user_message: str):
    """
    Führt den Plot-Agent einmalig aus und gibt ein Ergebnis-Dict zurück.
    """
    try:
        plan = llm_decide(user_message)
        tool_name = plan["tool"]
        args = plan["args"]

        result = route_tool(tool_name, args)

        return {
            "tool": tool_name,
            "args": args,
            "result": result
        }

    except Exception as e:
        return {"error": str(e)}