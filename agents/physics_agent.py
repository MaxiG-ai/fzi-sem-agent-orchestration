import json
import pandas as pd

from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from agents.utils import get_azure_llm
from agents.langfuse_config import get_langfuse_handler, flush_langfuse_handler
from langfuse.decorators import observe

from data.sp_data import load_sensor_data_from_csv

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

# --- AGENT SETUP ---

@observe()
def run_physics_agent(user_query: str) -> str:
    """
    Run the physics agent with Langfuse tracking.
    
    Args:
        user_query: The user's query for physics analysis
        
    Returns:
        The agent's response
    """
    global _current_data_json
    df = load_sensor_data_from_csv()
    _current_data_json = df.to_json()

    # 1. Setup Langfuse handler with metadata and tags
    langfuse_handler = get_langfuse_handler(
        trace_name="physics_agent",
        metadata={
            "agent_type": "physics",
            "query": user_query,
            "tools": ["calculate_correlations", "lookup_physics_formula"],
            "data_columns": df.columns.tolist()
        },
        tags=["agent", "physics", "data-analysis", "correlation"],
    )
    
    # 2. Setup LLM with Langfuse callback
    callbacks = [langfuse_handler] if langfuse_handler else None
    llm = get_azure_llm(callbacks=callbacks)
    tools = [calculate_correlations, lookup_physics_formula]

    system_message = f"""You are a physics-aware data analysis assistant. 
    Available sensor columns: {", ".join(df.columns.tolist())}
    Use the tools to analyze correlations and explain physical relationships.
    """

    prompt = {
        "messages": [
            SystemMessage(content=system_message),
            HumanMessage(content=user_query),
        ],
        "placeholder": "{agent_scratchpad}",
    }

    agent = create_agent(
        model=llm, 
        tools=tools, 
        )

    result = agent.invoke(prompt, config={"callbacks": callbacks} if callbacks else {})
    
    # Flush Langfuse handler
    flush_langfuse_handler(langfuse_handler)
    
    return result["messages"][-1].content