from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from data.sp_data import load_sensor_data_from_csv
from agents.utils import get_azure_llm  # Import the shared config

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


def run_statistics_agent(user_query: str) -> str:
    # 1. Setup LLM
    llm = get_azure_llm()

    # 2. Define Tools
    tools = [get_max_value, get_min_value, detect_outliers_above, detect_outliers_below]

    # 3. Create Prompt (English)
    # We load the dataframe just to get column names for the prompt context
    df = load_sensor_data_from_csv()
    columns_list = ", ".join(df.columns)

    system_prompt = (
        f"You are a statistical analysis assistant. Available columns: {columns_list}. "
        "Use the provided tools to answer questions."
    )

    prompt = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query),
        ],
        "placeholder": "{agent_scratchpad}",
    }

    # 4. Create Agent
    agent = create_agent(
        model=llm, 
        tools=tools,
        )

    # 6. Execute
    result = agent.invoke(prompt)
    return result["messages"][-1].content