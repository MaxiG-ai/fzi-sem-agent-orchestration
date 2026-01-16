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
from agents.langfuse_config import get_langfuse_handler
from langfuse import observe, get_client

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
    raise ValueError(f"Column '{col}' not found. Available columns: {list(df.columns)}")


def get_timestamp_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "time" in c.lower():
            return c
    raise ValueError("No timestamp column found in dataframe.")

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

@observe()
def run_plot_agent(user_query: str) -> str:
    """
    Run the plot agent with Langfuse tracking.
    
    Args:
        user_query: The user's query for plotting
        
    Returns:
        The agent's response
    """
    # 1. Setup Langfuse handler with metadata and tags
    langfuse_handler = get_langfuse_handler(
        trace_name="plot_agent",
        metadata={
            "agent_type": "visualization",
            "query": user_query,
            "tools": ["plot_time_series", "plot_histogram", "plot_scatter", "plot_corr"]
        },
        tags=["agent", "visualization", "plotting"],
    )
    
    # 2. Setup LLM with Langfuse callback
    callbacks = [langfuse_handler] if langfuse_handler else None
    llm = get_azure_llm(callbacks=callbacks)

    # Define the list of tools
    tools = [plot_time_series, plot_histogram, plot_scatter, plot_corr]

    # Get prompt from Langfuse
    langfuse = get_client()
    plot_prompt = langfuse.get_prompt("plot_agent")
    system_prompt = plot_prompt.compile()

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
    result = agent.invoke(prompt, config={"callbacks": callbacks} if callbacks else {})
    
    return result["messages"][-1].content