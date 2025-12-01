import matplotlib
matplotlib.use("Agg")   # verhindert MacOS GUI Fehler
import os
import pandas as pd
import matplotlib.pyplot as plt


from langchain.tools import tool

from streampipes_client import load_measure_as_df




def normalize_column(df: pd.DataFrame, raw_col: str) -> str:
    valid_cols = list(df.columns)

    normalized = (
        raw_col.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    exact = [c for c in valid_cols if c.lower() == normalized]
    if exact:
        return exact[0]

    partial = [c for c in valid_cols if normalized in c.lower()]
    if partial:
        return partial[0]

    raise ValueError(
        f"Spalte '{raw_col}' nicht gefunden. Verfügbare Spalten: {valid_cols}"
    )


def get_time_column(df: pd.DataFrame) -> str:
    ts_candidates = [
        c for c in df.columns
        if "time" in c.lower() or "timestamp" in c.lower()
    ]
    if not ts_candidates:
        raise ValueError("Keine Zeitspalte gefunden.")
    return ts_candidates[0]


def ensure_plots_dir() -> str:
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


#  @tool


@tool
def time_series_plot(measure: str, column: str, limit: int = 2000) -> str:
    """Erstellt einen Zeitreihenplot über eine Spalte."""
    df = load_measure_as_df(measure_id=measure, limit=limit)
    column = normalize_column(df, column)
    time_col = get_time_column(df)

    plt.figure(figsize=(10, 5))
    plt.plot(df[time_col], df[column])
    plt.title(f"Time Series: {column}")
    plt.xlabel(time_col)
    plt.ylabel(column)

    plots_dir = ensure_plots_dir()
    path = os.path.join(plots_dir, f"{measure}_{column}_timeseries.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return f"Saved time series plot to {path}"


@tool
def histogram_plot(measure: str, column: str, bins: int = 30, limit: int = 2000) -> str:
    """Erstellt ein Histogramm."""
    df = load_measure_as_df(measure_id=measure, limit=limit)
    column = normalize_column(df, column)

    plt.figure(figsize=(8, 5))
    plt.hist(df[column], bins=bins)
    plt.title(f"Histogram: {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plots_dir = ensure_plots_dir()
    path = os.path.join(plots_dir, f"{measure}_{column}_hist.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return f"Saved histogram to {path}"


@tool
def scatter_plot(measure: str, x_column: str, y_column: str, limit: int = 2000) -> str:
    """Erstellt einen Scatter Plot."""
    df = load_measure_as_df(measure_id=measure, limit=limit)
    x_column = normalize_column(df, x_column)
    y_column = normalize_column(df, y_column)

    plt.figure(figsize=(8, 5))
    plt.scatter(df[x_column], df[y_column], s=10)
    plt.title(f"Scatter: {y_column} vs {x_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)

    plots_dir = ensure_plots_dir()
    path = os.path.join(plots_dir, f"{measure}_{x_column}_vs_{y_column}_scatter.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return f"Saved scatter plot to {path}"


@tool
def boxplot_plot(measure: str, column: str, by: str | None = None, limit: int = 2000) -> str:
    """Erstellt einen Boxplot."""
    df = load_measure_as_df(measure_id=measure, limit=limit)
    column = normalize_column(df, column)

    plt.figure(figsize=(8, 5))
    if by is not None:
        by = normalize_column(df, by)
        df.boxplot(column=column, by=by)
        plt.title(f"Boxplot: {column} by {by}")
        plt.suptitle("")
        plt.xlabel(by)
        plt.ylabel(column)
    else:
        plt.boxplot(df[column].dropna())
        plt.title(f"Boxplot: {column}")
        plt.ylabel(column)

    plots_dir = ensure_plots_dir()
    suffix = f"_by_{by}" if by is not None else ""
    path = os.path.join(plots_dir, f"{measure}_{column}_boxplot{suffix}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return f"Saved boxplot to {path}"


@tool
def correlation_matrix_plot(measure: str, limit: int = 2000) -> str:
    """Erstellt eine Korrelationsmatrix."""
    df = load_measure_as_df(measure_id=measure, limit=limit)
    numeric_df = df.select_dtypes(include=["float64", "int64"])

    if numeric_df.empty:
        raise ValueError("Keine numerischen Spalten für die Korrelation gefunden.")

    corr = numeric_df.corr()

    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr, interpolation="nearest")
    plt.colorbar(im)
    plt.title(f"Correlation Matrix: {measure}")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

    plots_dir = ensure_plots_dir()
    path = os.path.join(plots_dir, f"{measure}_corr_matrix.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return f"Saved correlation matrix plot to {path}"
