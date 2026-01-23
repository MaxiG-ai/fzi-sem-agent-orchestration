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
    return mo, run_router, uuid


@app.cell
def _(uuid):
    session_id = str(uuid.uuid4())
    return session_id,


@app.cell
def _(mo):
    mo.md(r"""
    # Sensor Data Analysis Agent

    Use the chat interface below to interact with the Sensor Data Analysis Agent.

    You can ask questions about the relationship between temperature and density, request data visualizations or let it explain physics.
    """)
    return


@app.cell
def _(mo, run_router, session_id):
    
    local_system_promt = """Du bist ein intelligenter Router-Agent.

Wähle basierend auf der Nutzeranfrage einen der folgenden Agents:
1) Statistik-Agent → Max, Min, Mittelwert, Ausreißer, Kennzahlen
2) Plot-Agent → Zeitreihen, Histogramm, Scatter, Korrelationsmatrix
3) Physik-Agent → datenbasierte Korrelationen + physikalische Beziehung/Interpretation

Nutze IMMER ein Tool."""
    
    def agent_router_wrapper_model(messages, config):
        query = messages[-1].content
        history = [{"role": m.role, "content": m.content} for m in messages[:-1]]
        return run_router(query, local_system_promt, session_id=session_id, chat_history=history)

    mo.ui.chat(
        agent_router_wrapper_model,
        prompts=[
            "Can you show me how pressure and volume are related in the data?", # This results in an error since there is no pressure data. 
            "Explain the relationship between temperature and density.",
            "Plot a graph showing how temperature affects density.",
        ],
        show_configuration_controls=False,
        allow_attachments=False,
    )
    return


if __name__ == "__main__":
    app.run()
