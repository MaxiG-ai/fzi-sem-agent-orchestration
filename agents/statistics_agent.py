import os
import json
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI

from data.sp_data import load_sensor_data_from_csv

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

DATA_PATH = "data/sample_sensor_data.csv"

# TOOL FUNCTIONS (wie vorher, nur auf CSV)

@tool
def get_max_value(column_name: str):
    """Gibt den Maximalwert einer angegebenen Spalte zurück.

    Args:
        column_name (str): Name der Spalte

    Returns:
        float: Maximalwert der Spalte
    """
    df = load_sensor_data_from_csv(DATA_PATH)
    if column_name not in df.columns:
        return f"Spalte '{column_name}' existiert nicht."
    return float(df[column_name].max())

@tool
def get_min_value(column_name: str):
    """Gibt den Minimalwert einer angegebenen Spalte zurück

    Args:
        column_name (str): Name der Spalte

    Returns:
        float: Minimalwert der Spalte
    """
    df = load_sensor_data_from_csv(DATA_PATH)
    if column_name not in df.columns:
        return f"Spalte '{column_name}' existiert nicht."
    return float(df[column_name].min())

@tool
def detect_outliers_above(column_name: str, threshold_value: float):
    """ Finde Ausreißer oberhalb eines Schwellenwerts.

    Args:
        column_name (str): Name der Spalte
        threshold_value (float): Schwellenwert

    Returns:
        list: Liste der Ausreißer oberhalb des Schwellenwerts
    """
    df = load_sensor_data_from_csv(DATA_PATH)
    if column_name not in df.columns:
        return f"Spalte '{column_name}' existiert nicht."
    outliers = df[df[column_name] > threshold_value]
    return outliers.to_dict(orient="records")

@tool
def detect_outliers_below(column_name: str, threshold_value: float):
    """Finde Ausreißer unterhalb eines Schwellenwertes.

    Args:
        column_name (str): Name der Spalte
        threshold_value (float): Schwellenwert
    Returns:
        list: Liste der Ausreißer unterhalb des Schwellenwerts
    """
    df = load_sensor_data_from_csv(DATA_PATH)
    if column_name not in df.columns:
        return f"Spalte '{column_name}' existiert nicht."
    outliers = df[df[column_name] < threshold_value]
    return outliers.to_dict(orient="records")

# OPENAI FUNCTION DEFINITIONS

FUNCTIONS = [
    {
        "name": "get_max_value",
        "description": "Gibt das Maximum einer Sensorspalte zurück.",
        "parameters": {
            "type": "object",
            "properties": {
                "column_name": {"type": "string"}
            },
            "required": ["column_name"]
        }
    },
    {
        "name": "get_min_value",
        "description": "Gibt das Minimum einer Sensorspalte zurück.",
        "parameters": {
            "type": "object",
            "properties": {
                "column_name": {"type": "string"}
            },
            "required": ["column_name"]
        }
    },
    {
        "name": "detect_outliers_above",
        "description": "Finde Ausreißer oberhalb eines Schwellenwerts.",
        "parameters": {
            "type": "object",
            "properties": {
                "column_name": {"type": "string"},
                "threshold_value": {"type": "number"}
            },
            "required": ["column_name", "threshold_value"]
        }
    },
    {
        "name": "detect_outliers_below",
        "description": "Finde Ausreißer unterhalb eines Schwellenwerts.",
        "parameters": {
            "type": "object",
            "properties": {
                "column_name": {"type": "string"},
                "threshold_value": {"type": "number"}
            },
                "required": ["column_name", "threshold_value"]
            }
        }
    ]


def run_statistics_agent(user_query: str):
    df = load_sensor_data_from_csv(DATA_PATH)

    system_prompt = f"""
Du bist ein statistischer Analyse-Assistent.

WICHTIG:
- Nutze IMMER eine Funktion.
- KEINE reinen Textantworten.
- Die verfügbaren Spalten sind: {', '.join(df.columns)}
"""

    model = AzureChatOpenAI(
        azure_deployment=AZURE_AI_DEPLOYMENT,
        model_name=AZURE_AI_MODEL_NAME,
        api_version=AZURE_AI_API_VERSION,
        azure_endpoint=AZURE_AI_ENDPOINT,
        api_key=AZURE_AI_CREDENTIAL,
    )
    
    model.bind_tools([
        get_max_value,
        get_min_value,
        detect_outliers_above,
        detect_outliers_below
    ])
    
    response = model.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ])

    return response