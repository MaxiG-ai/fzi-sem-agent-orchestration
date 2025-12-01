# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "langchain-core==1.1.0",
#     "langchain-openai==1.1.0",
#     "langgraph==1.0.4",
#     "pandas==2.3.3",
#     "python-dotenv==1.2.1",
#     "streampipes==0.97.0",
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

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell
def _():
    """Import required libraries for StreamPipes and data analysis."""
    from streampipes.client import StreamPipesClient
    from streampipes.client.config import StreamPipesClientConfig
    from streampipes.client.credential_provider import StreamPipesApiKeyCredentials
    from agents.physics_agent import run_agent
    import pandas as pd
    import marimo as mo
    return (
        StreamPipesApiKeyCredentials,
        StreamPipesClient,
        StreamPipesClientConfig,
        mo,
        pd,
        run_agent,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Connect to Apache StreamPipes.

    Configure your StreamPipes credentials here:
    - username: Your StreamPipes user email
    - api_key: Your StreamPipes API key (found in user settings)
    - host_address: StreamPipes server address
    - port: StreamPipes server port
    """)
    return


@app.cell
def _(
    StreamPipesApiKeyCredentials,
    StreamPipesClient,
    StreamPipesClientConfig,
):
    config = StreamPipesClientConfig(
        credential_provider=StreamPipesApiKeyCredentials(
            username="admin@streampipes.apache.org",
            api_key="v@wGP8Dy_o,1GXCE-00y.WBw",
        ),
        host_address="localhost",
        https_disabled=True,  # Set to False if using HTTPS
        port=80
    )

    # Initialize the StreamPipes client
    client = StreamPipesClient(client_config=config)

    print("✓ Connected to StreamPipes")
    client
    return (client,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    List all available data lake measures.

    Data lake measures are time-series datasets stored in StreamPipes.
    This cell helps you identify which measures are available for analysis.
    """)
    return


@app.cell
def _(client):
    # Fetch all available measures
    measures = client.dataLakeMeasureApi.all()

    print(f"Found {len(measures)} data lake measures:")
    for measure in measures:
        print(f"  - {measure.measure_name}")

    # Display as a pandas DataFrame for easy viewing
    measures.to_pandas()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Fetch sensor data from the 'sensor_data' measure.

    This retrieves all available data points and converts them to a pandas DataFrame.
    The timestamp column is converted to proper datetime format for time-series analysis.
    """)
    return


@app.cell
def _(client, pd):
    # Fetch the specific measure
    datalake_measure = client.dataLakeMeasureApi.get(identifier="sensor_data")

    # Convert to pandas DataFrame
    df = datalake_measure.to_pandas()

    # Convert timestamp to datetime for proper time-series handling
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    print(f"✓ Loaded {len(df)} data points")
    print(f"✓ Columns: {', '.join(df.columns.tolist())}")
    return (df,)


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    print("Sample sensor data:")
    df.head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run the physics-aware AI agent on the sensor data.

    The agent can:
    - Calculate correlations between sensor measurements
    - Look up relevant physics formulas
    - Provide insights based on physical relationships

    Modify the user_message below to ask different questions!
    """)
    return


@app.cell
def _(df, run_agent):
    # Define your analysis question here
    user_message = (
        "Analyze the sensor data and identify the key correlations between measurements. "
        "Then explain what physics formulas govern these relationships and whether "
        "the observed correlations match the expected physical behavior."
    )

    # Run the agent
    result = run_agent(user_message, df)
    return (result,)


@app.cell
def _(mo, result):
    mo.md(result)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Additional analysis examples:

    Example 1: Temperature-Density relationship
    ```python
    run_agent(
        "How does temperature affect density in this dataset? "
        "Is this consistent with thermal expansion principles?",
        df
    )
    ```

    Example 2: Flow rate analysis
    ```python
    run_agent(
        "Analyze the relationship between mass_flow and volume_flow. "
        "What does this tell us about the fluid density?",
        df
    )
    ```

    Example 3: Anomaly detection
    ```python
    run_agent(
        "Are there any measurements that seem inconsistent with expected physics? "
        "Check all correlations and highlight any anomalies.",
        df
    )
    ```
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
