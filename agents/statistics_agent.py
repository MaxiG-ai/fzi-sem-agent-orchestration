from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from data.sp_data import load_sensor_data_from_csv
from agents.utils import get_azure_llm
from agents.langfuse_config import get_langfuse_handler
from langfuse import observe, get_client

# --- TOOLS (Docstrings translated to English for the LLM) ---


@tool
def get_max_value(column_name: str):
    """Returns the maximum value of a specified column."""
    df = load_sensor_data_from_csv()
    if column_name not in df.columns:
        return f"Column '{column_name}' does not exist."
    return float(df[column_name].max())


@tool
def get_min_value(column_name: str):
    """Returns the minimum value of a specified column."""
    df = load_sensor_data_from_csv()
    if column_name not in df.columns:
        return f"Column '{column_name}' does not exist."
    return float(df[column_name].min())


@tool
def detect_outliers_above(column_name: str, threshold_value: float):
    """Finds outliers above a certain threshold value."""
    df = load_sensor_data_from_csv()
    if column_name not in df.columns:
        return f"Column '{column_name}' does not exist."
    outliers = df[df[column_name] > threshold_value]
    return outliers.to_dict(orient="records")


@tool
def detect_outliers_below(column_name: str, threshold_value: float):
    """Finds outliers below a certain threshold value."""
    df = load_sensor_data_from_csv()
    if column_name not in df.columns:
        return f"Column '{column_name}' does not exist."
    outliers = df[df[column_name] < threshold_value]
    return outliers.to_dict(orient="records")


# --- AGENT SETUP ---


@observe()
def run_statistics_agent(user_query: str) -> str:
    """
    Run the statistics agent with Langfuse tracking.
    
    Args:
        user_query: The user's query for statistical analysis
        
    Returns:
        The agent's response
    """
    # 1. Setup Langfuse handler with metadata and tags
    langfuse_handler = get_langfuse_handler(
        trace_name="statistics_agent",
        metadata={
            "agent_type": "statistics",
            "query": user_query,
            "tools": ["get_max_value", "get_min_value", "detect_outliers_above", "detect_outliers_below"]
        },
        tags=["agent", "statistics", "data-analysis"],
    )
    
    # 2. Setup LLM with Langfuse callback
    callbacks = [langfuse_handler] if langfuse_handler else None
    llm = get_azure_llm(callbacks=callbacks)

    # 3. Define Tools
    tools = [get_max_value, get_min_value, detect_outliers_above, detect_outliers_below]

    # 4. Create Prompt (English)
    # We load the dataframe just to get column names for the prompt context
    df = load_sensor_data_from_csv()
    columns_list = ", ".join(df.columns)

    # Get prompt from Langfuse
    langfuse = get_client()
    stats_prompt = langfuse.get_prompt("statistics_agent")
    system_prompt = stats_prompt.compile(columns=columns_list)

    prompt = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query),
        ],
        "placeholder": "{agent_scratchpad}",
    }

    # 5. Create Agent with callbacks
    agent = create_agent(
        model=llm, 
        tools=tools,
        )

    # 6. Execute
    result = agent.invoke(prompt, config={"callbacks": callbacks} if callbacks else {})
    
    return result["messages"][-1].content