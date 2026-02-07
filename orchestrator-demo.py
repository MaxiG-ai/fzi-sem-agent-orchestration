"""
StreamPipes Physics Agent Demo

This Marimo notebook demonstrates:
1. Connecting to Apache StreamPipes running on localhost
2. Fetching sensor data from a data lake measure
3. Analyzing the data using a physics-aware AI agent with LangGraph

Prerequisites:
- Apache StreamPipes running at localhost:80
- Valid API credentials configured below
- sensor_data measure available in the data lake
"""

import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell
def _():
    """Import required libraries for StreamPipes and data analysis."""
    from orchestrator import run_router
    import marimo as mo
    import uuid
    import os
    return mo, os, run_router, uuid


@app.cell
def _(uuid):
    session_id = str(uuid.uuid4())
    return session_id,


@app.cell
def _(mo):
    get_plot_path, set_plot_path = mo.state(None)
    return get_plot_path, set_plot_path


@app.cell
def _(mo):
    mo.md(r"""
    # Sensor Data Analysis Agent

    Nutze die untenstehende Chat-Oberfläche, um mit dem Sensordatenanalyse-Agenten zu chatten.

Du kannst Fragen zum Zusammenhang zwischen Temperatur und Dichte stellen, Datenvisualisierungen anfordern oder physikalische Gesetze erklären lassen.    """)
    return


@app.cell
def _(mo, os, run_router, session_id, set_plot_path):
    
    local_system_promt = """Du bist ein intelligenter Router-Agent.

Wähle basierend auf der Nutzeranfrage einen der folgenden Agents:
1) Statistik-Agent → Max, Min, Mittelwert, Ausreißer, Kennzahlen
2) Plot-Agent → Zeitreihen, Histogramm, Scatter, Korrelationsmatrix
3) Physik-Agent → datenbasierte Korrelationen + physikalische Beziehung/Interpretation

Nutze IMMER ein Tool."""
    
    def agent_router_wrapper_model(messages, config):
        # Reset plot path
        if "LATEST_PLOT_PATH" in os.environ:
            del os.environ["LATEST_PLOT_PATH"]

        query = messages[-1].content
        history = [{"role": m.role, "content": m.content} for m in messages[:-1]]
        response = run_router(query, local_system_promt, session_id=session_id, chat_history=history)

        # Check for new plot
        new_plot = os.environ.get("LATEST_PLOT_PATH")
        set_plot_path(new_plot)

        return response

    mo.ui.chat(
        agent_router_wrapper_model,
        prompts=[
            "Was ist der Durchschnitt der Temperaturwerte im oberen Quartil der Messwerte?",
            "Can you show me how pressure and volume are related in the data?", # This results in an error since there is no pressure data. 
            "Erstelle ein Histogramm der Temperatur.",
        ],
        show_configuration_controls=False,
        allow_attachments=False,
    )
    return

@app.cell
def _(get_plot_path, mo, os):
    path = get_plot_path()
    path = os.environ.pop("LATEST_PLOT_PATH", None) if path is None else path
    mo.image(
        src=path,
        alt="Plot image",
        width="600px",
        rounded=True,
        caption=f"Letzter generierter Plot aus pfad: {path} ",
    ) if path else mo.md("Noch kein Plot generiert.")
    return

@app.cell
def _(mo):
    mo.md("""
## Hier ein paar Links zu Beispiel-Traces in Langfuse:

- [Frage nach nicht vorhandenem Tool: Antwort mit Frage](https://cloud.langfuse.com/project/cmj452rvf004fad078fbgnbpo/traces/3b2a081aac8ddffcd0ec08d4b16f7be3?observation=acdf45ce562afca3&timestamp=2026-01-16T16:12:56.200Z)
- [Frage nach nicht vorhandenem Tool: Trotzdem gelöst](https://cloud.langfuse.com/project/cmj452rvf004fad078fbgnbpo/traces/4d305729032fa15414b5d4578b172958?timestamp=2026-01-16T16%3A12%3A56.198Z)
- [Tool Error](https://cloud.langfuse.com/project/cmj452rvf004fad078fbgnbpo/traces/0366af11c2bc610ca487270c45110596?observation=0aa02f533a5a1928&timestamp=2026-01-16T10%3A33%3A09.916Z)
- [LLM Error](https://cloud.langfuse.com/project/cmj452rvf004fad078fbgnbpo/traces?dateRange=30d&pageIndex=1&pageSize=50&search=&peek=1017170e43c4b7c69b3ee73bd6b89508&timestamp=2026-01-16T16%3A17%3A26.923Z)
- [unklare Nutzeranfrage](https://cloud.langfuse.com/project/cmj452rvf004fad078fbgnbpo/traces/fa19fc8b163898a245cf92abf2613975?timestamp=2026-01-16T15:52:32.082Z)
- [hier klappt alles](https://cloud.langfuse.com/project/cmj452rvf004fad078fbgnbpo/traces/0bcb46b66056f921599b4ff8b8b2ee85?timestamp=2026-01-16T15:52:32.081Z)
    """)
    return

if __name__ == "__main__":
    app.run()
