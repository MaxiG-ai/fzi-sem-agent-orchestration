"""
Physics-Aware Data Analysis Agent

This module provides an AI agent that analyzes sensor data from Apache StreamPipes
with physics domain knowledge. The agent can:
- Calculate correlations between sensor measurements
- Look up relevant physics formulas
- Provide insights based on physical relationships

The agent uses LangGraph for orchestration and LangSmith for optional observability.
"""

import json
import pandas as pd
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from data.sp_data import load_sensor_data_from_csv

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load environment variables from .env file
load_dotenv()

AZURE_AI_CREDENTIAL = os.getenv("AZURE_AI_CREDENTIAL")
AZURE_AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT")
AZURE_AI_MODEL_NAME = os.getenv("AZURE_AI_MODEL_NAME")
AZURE_AI_DEPLOYMENT = os.getenv("AZURE_AI_DEPLOYMENT", "o4-mini")
AZURE_AI_API_VERSION = os.getenv("AZURE_AI_API_VERSION", "2024-12-01-preview")

if not AZURE_AI_CREDENTIAL:
    raise ValueError(
        "AZURE_AI_CREDENTIAL not found in .env file. Please create a .env file with your API key."
    )

# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """State schema for the agent graph."""
    messages: Annotated[list, add_messages]
    data_json: str

# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

# Global variable to hold data_json during execution
_current_data_json = None

@tool
def calculate_correlations(columns: list[str]) -> str:
    """
    Calculate Pearson correlation coefficients between specified data columns.
    
    Args:
        columns: List of column names to correlate. Empty list = correlate all numeric columns.
        
    Returns:
        JSON string containing correlation matrix or error message
    """
    global _current_data_json
    
    try:
        # Deserialize the dataframe from global state
        from io import StringIO
        df = pd.read_json(StringIO(_current_data_json))
        
        if not columns:
            # Correlate all numeric columns if none specified
            corr_matrix = df.corr(numeric_only=True)
        else:
            # Filter to only requested columns that exist in the dataframe
            available = [c for c in columns if c in df.columns]
            if not available:
                return json.dumps({"error": "No valid columns found in dataframe"})
            corr_matrix = df[available].corr()
        
        # Format correlations as nested dictionary (excluding self-correlations)
        result = {}
        for col1 in corr_matrix.columns:
            result[col1] = {}
            for col2 in corr_matrix.columns:
                if col1 != col2:  # Skip diagonal (self-correlation = 1.0)
                    result[col1][col2] = round(corr_matrix.loc[col1, col2], 4)
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Correlation calculation failed: {str(e)}"})


@tool
def lookup_physics_formula(fields: list[str]) -> str:
    """
    Look up physics formulas and relationships for specified sensor fields.
    
    This function contains a knowledge base of common fluid mechanics and thermodynamics
    relationships relevant to industrial sensor data.
    
    Args:
        fields: List of field names (e.g., ['mass_flow', 'density', 'temperature'])
        
    Returns:
        JSON string containing relevant formulas with context and units
    """
    try:
        # Knowledge base of physics formulas
        # Each entry maps field combinations to their physical relationships
        formula_db = {
            ("mass_flow", "volume_flow", "density"): {
                "formula": "mass_flow = density × volume_flow",
                "latex": "\\dot{m} = \\rho \\times \\dot{V}",
                "context": "Fundamental relationship in fluid mechanics. Mass flow rate equals fluid density multiplied by volumetric flow rate.",
                "units": "kg/s = (kg/m³) × (m³/s)",
            },
            ("mass_flow", "density"): {
                "formula": "mass_flow = density × volume_flow",
                "latex": "\\dot{m} = \\rho \\times \\dot{V}",
                "context": "Mass flow rate is proportional to density. Higher density fluids carry more mass at same volumetric flow.",
                "units": "kg/s ∝ kg/m³",
            },
            ("temperature", "density"): {
                "formula": "density ≈ ρ₀ × (1 - β × ΔT)",
                "latex": "\\rho \\approx \\rho_0 (1 - \\beta \\Delta T)",
                "context": "Thermal expansion relationship. Most fluids decrease in density as temperature increases (volumetric expansion).",
                "units": "β is thermal expansion coefficient (K⁻¹)",
            },
            ("level", "volume_flow"): {
                "formula": "dLevel/dt ∝ volume_flow",
                "latex": "\\frac{d(Level)}{dt} \\propto \\dot{V}",
                "context": "Container level changes proportionally to net volumetric flow rate. Positive flow increases level, negative decreases it.",
                "units": "m/s ∝ m³/s (depends on container cross-section)",
            },
            ("mass_flow", "volume_flow"): {
                "formula": "mass_flow = density × volume_flow",
                "latex": "\\dot{m} = \\rho \\times \\dot{V}",
                "context": "These should be highly correlated if density is relatively constant.",
                "units": "kg/s = (kg/m³) × (m³/s)",
            },
        }
        
        # Normalize field names to lowercase and sort for matching
        normalized = tuple(sorted([f.lower() for f in fields]))
        
        # Find all formulas that include the requested fields
        results = []
        for formula_fields, formula_data in formula_db.items():
            if set(normalized).issubset(set(formula_fields)):
                results.append(formula_data)
        
        if results:
            return json.dumps({"formulas": results}, indent=2)
        else:
            return json.dumps({
                "formulas": [{
                    "formula": "No direct formula found",
                    "context": f"No built-in formula for: {', '.join(fields)}",
                    "suggestion": "Try querying with different field combinations or individual field pairs",
                }]
            }, indent=2)
            
    except Exception as e:
        return json.dumps({"error": f"Formula lookup failed: {str(e)}"})


# ============================================================================
# AGENT GRAPH
# ============================================================================

# Collect tools in a list for binding
tools = [calculate_correlations, lookup_physics_formula]

def should_continue(state: AgentState) -> str:
    """Determine whether to continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there are tool calls, route to tools node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise, end the graph
    return END


def call_model(state: AgentState):
    """Call the language model with tools bound."""
    messages = state["messages"]
    
    # Initialize ChatOpenAI with Azure configuration
    model = AzureChatOpenAI(
        azure_deployment=AZURE_AI_DEPLOYMENT,
        model_name=AZURE_AI_MODEL_NAME,
        api_version=AZURE_AI_API_VERSION,
        azure_endpoint=AZURE_AI_ENDPOINT,
        api_key=AZURE_AI_CREDENTIAL,
    )
    
    # Bind tools to the model
    model_with_tools = model.bind_tools(tools)
    
    # Invoke the model
    response = model_with_tools.invoke(messages)
    
    # Return updated state
    return {"messages": [response]}


# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

# Add edges
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    }
)
workflow.add_edge("tools", "agent")

# Compile the graph
graph = workflow.compile()

# ============================================================================
# AGENT RUNNER
# ============================================================================

def run_physics_agent(
    user_message: str, 
    max_iterations: int = 10
) -> str:
    """
    Run the physics analysis agent on sensor data.
    
    This implements an agentic loop using LangGraph where the LLM can:
    1. Receive the user's question
    2. Decide which tools to call (if any)
    3. Execute those tools automatically via ToolNode
    4. Synthesize results into a final answer
    
    Args:
        user_message: The user's question or analysis request
        df: Pandas DataFrame containing sensor data
        max_iterations: Maximum number of agent loop iterations (safety limit)
        
    Returns:
        The agent's final response as a string
    """
    global _current_data_json
    
    df = load_sensor_data_from_csv("data/sample_sensor_data.csv")
    
    # Serialize dataframe and store in global variable
    _current_data_json = df.to_json()
    
    # System prompt to guide the agent's behavior
    system_prompt = f"""You are a physics-aware data analysis assistant. You help users understand sensor data 
from industrial systems by analyzing correlations and explaining physical relationships.

You have access to sensor data with these columns: {', '.join(df.columns.tolist())}

Use the available tools to:
1. Calculate correlations between measurements
2. Look up relevant physics formulas
3. Provide insights based on physical principles

Be concise and focus on actionable insights."""

    # Initialize state
    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ],
        "data_json": _current_data_json,
    }
    
    print(f"\n{'=' * 80}")
    print(f"USER: {user_message}")
    print(f"{'=' * 80}\n")
    
    # Track iteration count
    iteration = 0
    
    # Invoke the graph with streaming to observe tool calls
    final_state = None
    for event in graph.stream(initial_state, {"recursion_limit": max_iterations}):
        iteration += 1
        
        # Print tool execution details
        if "tools" in event:
            tool_messages = event["tools"]["messages"]
            print(
                f"\n[Iteration {iteration}] 🤖 Agent is calling {len(tool_messages)} tool(s)...\n"
            )
            for msg in tool_messages:
                print("\t✅ Tool result received")
        
        # Check for agent responses
        if "agent" in event:
            agent_messages = event["agent"]["messages"]
            for msg in agent_messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        print(f"🔧 Tool: {tool_call['name']}")
                        print(f"   Args: {json.dumps(tool_call['args'], indent=6)}")
        
        final_state = event
    
    # Extract final response
    # Get the last event which should contain the final state
    if final_state:
        # Find the last message from the agent
        all_messages = None
        if "agent" in final_state:
            all_messages = final_state["agent"]["messages"]
        elif "__end__" in final_state:
            all_messages = final_state["__end__"]["messages"]
        
        for msg in reversed(all_messages):
            if hasattr(msg, "content") and msg.content:
                final_response = msg.content
                print(f"\n{'=' * 80}")
                print(f"AGENT: {final_response}")
                print(f"{'=' * 80}\n")
                return final_response
            else:
                raise ValueError("No final response found from agent.")
    

# ============================================================================
# STANDALONE TESTING
# ============================================================================

def main():
    """
    Standalone test with store sensor data.
    Run this file directly to test the agent without StreamPipes.
    """
    # Load using sample CSV data
    df = load_sensor_data_from_csv("data/sample_sensor_data.csv")
    
    # Test query 1: Correlation and physics relationship
    run_physics_agent(
        "Analyze correlations between mass_flow, volume_flow, and density. "
        "Then check what physics formula should relate these fields.",
        df,
    )
    
    print("\n" + "="*80 + "\n")
    
    # Test query 2: Temperature-density relationship
    run_physics_agent(
        "I'm seeing temperature and density changing together. "
        "What physics explains this relationship?",
        df,
    )


if __name__ == "__main__":
    main()