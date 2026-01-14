# /// script
# requires-python = ">=3.13"
# dependencies = [
# "langchain==1.1.0",
# "langchain-core==1.1.0",
# "langchain-openai==1.1.0",
# "langsmith==0.4.49",
# "langgraph==1.0.4",
# "marimo>=0.19.2",
# "pandas==2.3.3",
# "python-dotenv==1.2.1",
# "streampipes==0.97.0",
# "utils==1.0.2",
# "numpy==2.3.5",
# "matplotlib==3.10.7",
# "orjson==3.11.4",
# "requests==2.32.5",
# "langfuse>=0.9.0",
# ]
# ///

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
import os
from dotenv import load_dotenv
from langfuse import get_client


__generated_with = "0.18.1"
app = marimo.App(width="full")

@app.cell
def create_lf_client():
    import os
    from dotenv import load_dotenv
    from langfuse import get_client

    # .env laden (falls noch nicht geladen)
    load_dotenv()

    # ENV setzen, damit get_client() sie sehen kann
    os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
    os.environ["LANGFUSE_BASE_URL"] = os.getenv("LANGFUSE_BASE_URL")

    # Client erzeugen (ohne Argumente!)
    lf_client = get_client()

    return lf_client

@app.cell
def import_agent():
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
def _(mo, run_router, lf_client):
    def agent_router_wrapper_model(messages, config):
        import time
        user_message = messages[-1].content
        start_time = time.time()

        # Hauptspan für die User-Anfrage
        with lf_client.start_as_current_observation(
            as_type="span",
            name="process-request"
        ) as span:
            # Input loggen
            span.update(input=user_message)
            span.update(metrics={"input_length": len(user_message)})

            try:
                # Nested LLM Span
                generation_start = time.time()
                with lf_client.start_as_current_observation(
                    as_type="generation",
                    name="llm-response",
                    model="gpt-3.5-turbo"
                ) as generation:
                    response = run_router(user_message)
                    generation_end = time.time()

                    # Output und Metriken loggen im Nested Span
                    generation.update(output=response)
                    generation.update(metrics={
                        "output_length": len(response),
                        "execution_time_ms": int((generation_end - generation_start) * 1000)
                    })

                    # **Output und Metriken auch im Parent-Span loggen**
                    span.update(metrics={
                        "output_length": len(response),
                        "generation_time_ms": int((generation_end - generation_start) * 1000)
                    })

            except Exception as e:
                span.update(status="error", output=str(e))
                raise

            # Gesamtdauer für die Anfrage
            total_time_ms = int((time.time() - start_time) * 1000)
            span.update(metrics={"total_execution_ms": total_time_ms})
            span.update(status="success")

        return response

    mo.ui.chat(
        agent_router_wrapper_model,
        prompts=[
            "Can you show me how pressure and volume are related in the data?",
            "Explain the relationship between temperature and density.",
            "Plot a graph showing how temperature affects density.",
        ],
        show_configuration_controls=False,
        allow_attachments=False,
    )


if __name__ == "__main__":
    app.run()