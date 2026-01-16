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
    return mo, run_router


@app.cell
def _(mo):
    mo.md(r"""
    # Sensor Data Analysis Agent

    Use the chat interface below to interact with the Sensor Data Analysis Agent.

    You can ask questions about the relationship between temperature and density, request data visualizations or let it explain physics.
    """)
    return


@app.cell
def _(mo, run_router):
    def agent_router_wrapper_model(messages, config):
        return run_router(messages[-1].content)

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
