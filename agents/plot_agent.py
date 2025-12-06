# agents/plot_agent.py
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from agents.utils import get_azure_llm

from data.sp_data import load_sensor_data_from_csv


def ensure_plots_dir():
    plot_dir = os.path.join("plots")
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
def plot_time_series(column: str):
    """Plot time series from a df for a column

    Args:
        column (str): Column to plot

    Returns:
        dict: Dictionary with the path to the saved plot image
    """
    df = load_sensor_data_from_csv()
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
def plot_histogram(column: str, bins: int = 30):
    """Plot histogram for a specific column from a df.

    Args:
        column (str): Column to plot
        bins (int, optional): Number of bins for the histogram. Defaults to 30.

    Returns:
        dict: Dictionary with the path to the saved plot image
    """
    df = load_sensor_data_from_csv()
    column = normalize_column(df, column)

    plt.figure(figsize=(8, 5))
    plt.hist(df[column], bins=bins)
    plt.title(f"Histogramm für {column}")

    path = os.path.join(ensure_plots_dir(), f"hist_{column}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return {"path": path}


@tool
def plot_scatter(x_col: str, y_col: str):
    """Plot scatter plot for two columns from a df.
    Args:
        x_col (str): Column for x-axis
        y_col (str): Column for y-axis
        
    Returns: 
        dict: Dictionary with the path to the saved plot image
    """
    df = load_sensor_data_from_csv()
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
def plot_corr():
    """Plot correlation matrix for numeric columns in a df.

    Returns:
        dict: Dictionary with the path to the saved plot image
    """
    df = load_sensor_data_from_csv()
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

# plot agent runner

def run_plot_agent(user_query: str) -> str:
    llm = get_azure_llm()

    # Define the list of tools
    tools = [plot_time_series, plot_histogram, plot_scatter, plot_corr]

    system_prompt = (
        "You are a data visualization agent. Create plots based on user requests. "
        "Always return the file path of the created plot."
    )

    prompt = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query),
        ],
        "placeholder": "{agent_scratchpad}",
    }

    agent = create_agent(
        model=llm, 
        tools=tools, 
    )
    result = agent.invoke(prompt)
    return result["messages"][-1].content