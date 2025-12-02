#was ich geändert habe: die Daten werden aus der orchestrator datei genommen und nicht mehr in der datei hier
#der ein und ausgabe loop wurde raus gemacht 
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

#from streampipes_data import load_measure_df

# SETUP
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY fehlt in .env")

LLM = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#DEFAULT_MEASURE = "sensor_data"
#VALID_MEASURES = ["sensor_data"]


def ensure_plots_dir():
    plot_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def normalize_column(df: pd.DataFrame, col: str) -> str:
    """Versucht, eine Spalte tolerant zu finden (case-insensitive + substring)."""
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



# Prüfung, ob Zeitwert gültig ist

def is_valid_time(value):
    """Erlaubt nur echte Zeitstrings – ignoriert '.', '...', None etc."""
    if not value:
        return False
    cleaned = str(value).strip()
    if cleaned in ["...", "…", "null", "None", "", " "]:
        return False
    return True

# DATAFRAME FILTERUNG

def filter_df(df: pd.DataFrame, args: dict) -> pd.DataFrame:
    df_f = df.copy()

    # Zeitfilter robust anwenden
    if is_valid_time(args.get("start_time")):
        df_f = df_f[df_f["timestamp"] >= pd.to_datetime(args["start_time"])]

    if is_valid_time(args.get("end_time")):
        df_f = df_f[df_f["timestamp"] <= pd.to_datetime(args["end_time"])]

    # Limit (immer tail → sinnvoller für Sensor-Daten)
    if args.get("limit"):
        df_f = df_f.tail(int(args["limit"]))

    return df_f

# PLOT-FUNKTIONEN


def plot_time_series(df, measure, column):
    column = normalize_column(df, column)
    ts = get_timestamp_column(df)

    plt.figure(figsize=(10, 5))
    plt.plot(df[ts], df[column])
    plt.title(f"{measure}: {column} über Zeit")

    path = os.path.join(ensure_plots_dir(), f"{measure}_{column}_timeseries.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return {"path": path}


def plot_histogram(df, measure, column, bins=30):
    column = normalize_column(df, column)

    plt.figure(figsize=(8, 5))
    plt.hist(df[column], bins=bins)
    plt.title(f"Histogramm: {column}")

    path = os.path.join(ensure_plots_dir(), f"{measure}_{column}_hist.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return {"path": path}


def plot_scatter(df, measure, x_col, y_col):
    x_col = normalize_column(df, x_col)
    y_col = normalize_column(df, y_col)

    plt.figure(figsize=(8, 5))
    plt.scatter(df[x_col], df[y_col], s=10)
    plt.title(f"{y_col} vs {x_col}")

    path = os.path.join(ensure_plots_dir(), f"{measure}_{x_col}_{y_col}_scatter.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return {"path": path}


def plot_corr(df, measure):
    numeric = df.select_dtypes(include=["float64", "int64"])
    corr = numeric.corr()

    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.title("Korrelationsmatrix")

    path = os.path.join(ensure_plots_dir(), f"{measure}_corr.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return {"path": path}

# LLM SYSTEM PROMPT


SYSTEM_PROMPT = """
Du bist ein Datenanalyse-Agent für StreamPipes.

FILTEROPTIONEN:
- limit: Anzahl der Datenpunkte (z.B. 300)
- start_time: Startzeit (YYYY-MM-DD HH:MM)
- end_time: Endzeit (YYYY-MM-DD HH:MM)

Du gibst IMMER folgendes JSON zurück:

{
  "tool": "<tool>",
  "args": {
    "measure": "sensor_data",
    "column": "...",
    "limit": <int>,
    "start_time": "...",
    "end_time": "..."
  }
}

Regeln:
- Zeitreihe → tool = "time_series"
- Histogramm → tool = "histogram"
- Scatter → tool = "scatter"
- Korrelationsmatrix → tool = "corr"
- measure bleibt IMMER 'sensor_data' (außer explizit genannt).
- Zeitangaben → start_time / end_time.
- Limit ist ein Integer.
- Niemals '...' oder sonstige Dummy-Zeichen als gültige Zeiten setzen.
- Wenn Nutzer keine Zeit nennt → KEINE start_time / end_time erzeugen.
"""

def llm_decide(user_msg: str) -> dict:
    response = LLM.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)

# ROUTER


def route_tool(df_sensor: pd.DataFrame, tool: str, args: dict):
    df_filtered = filter_df(df_sensor, args)

    if tool == "time_series":
        return plot_time_series(df_filtered, args.get("measure", "sensor_data"), args["column"])

    if tool == "histogram":
        return plot_histogram(df_filtered, args.get("measure", "sensor_data"), args["column"], args.get("bins", 30))

    if tool == "scatter":
        return plot_scatter(df_filtered, args.get("measure", "sensor_data"), args["x_col"], args["y_col"])

    if tool == "corr":
        return plot_corr(df_filtered, args.get("measure", "sensor_data"))

    raise ValueError(f"Unbekanntes Tool: {tool}")

#def run_plot_agent(user_input: str, df_sensor: pd.DataFrame) -> str:
    plan = llm_decide(user_input)
    result = route_tool(df_sensor, plan["tool"], plan["args"])
    return f"Plot gespeichert unter: {result['path']}"
# ---------------------------

# NEU: run_plot_agent
def run_plot_agent(user_input: str, df_sensor: pd.DataFrame) -> str:
    plan = llm_decide(user_input)
    result = route_tool(df_sensor, plan["tool"], plan["args"])
    return f"Plot gespeichert unter: {result['path']}"
# ---------------------------

# CLI LOOP den kommentiere ich aus damit der loop nicht drin is 


#def main():
    #print("🚀 Plot-Agent gestartet — StreamPipes angebunden")
    #print("exit zum Beenden.\n")

    #while True:
        #user = input("> ").strip()
        #if user.lower() in ("exit", "quit"):
         #   break

        #try:
         #   plan = llm_decide(user)
          #  print("🔧 Tool:", plan["tool"])
           # print("🔧 Args:", plan["args"])

            #result = route_tool(plan["tool"], plan["args"])
            #print("\n📁 Plot gespeichert unter:", result["path"])

#        except Exception as e:
 #           print("❌ Fehler:", e)


#if __name__ == "__main__":
  #  main()