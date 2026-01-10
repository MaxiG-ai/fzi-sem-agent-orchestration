import os
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain_core.tools.structured import StructuredTool
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import ChatMessageHistory

# =============================
#  .env laden
# =============================
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key is None:
    raise ValueError("OPENAI_API_KEY nicht gefunden! Bitte in .env eintragen.")

# =============================
#  Funktionen für Datenanalyse
# =============================
def get_max_value(df: pd.DataFrame, column_name: str):
    """Gibt den höchsten Wert der angegebenen Spalte zurück."""
    if column_name not in df.columns:
        return f"Spalte '{column_name}' existiert nicht."
    value = df[column_name].max()
    return f"Der höchste Wert in '{column_name}' beträgt {float(value):.2f}."


def get_min_value(df: pd.DataFrame, column_name: str):
    """Gibt den niedrigsten Wert der angegebenen Spalte zurück."""
    if column_name not in df.columns:
        return f"Spalte '{column_name}' existiert nicht."
    value = df[column_name].min()
    return f"Der niedrigste Wert in '{column_name}' beträgt {float(value):.2f}."


def detect_outliers_above(df: pd.DataFrame, column_name: str, threshold_value: float):
    """Findet alle Werte in der Spalte, die über dem angegebenen Schwellenwert liegen."""
    if column_name not in df.columns:
        return f"Spalte '{column_name}' existiert nicht."
    outliers = df[df[column_name] > threshold_value]
    if outliers.empty:
        return f"Keine Ausreißer über {threshold_value} gefunden."
    return outliers[[column_name, "timestamp"]].to_dict(orient="records")


def detect_outliers_below(df: pd.DataFrame, column_name: str, threshold_value: float):
    """Findet alle Werte in der Spalte, die unter dem angegebenen Schwellenwert liegen."""
    if column_name not in df.columns:
        return f"Spalte '{column_name}' existiert nicht."
    outliers = df[df[column_name] < threshold_value]
    if outliers.empty:
        return f"Keine Ausreißer unter {threshold_value} gefunden."
    return outliers[[column_name, "timestamp"]].to_dict(orient="records")


# =============================
#  Agent-Konstruktion
# =============================
def create_ano_agent(df: pd.DataFrame) -> AgentExecutor:
    # Tools als normale Funktionen mit Docstrings definieren
    def max_value_tool(column_name: str):
        """Gibt den höchsten Wert einer Spalte zurück."""
        return get_max_value(df, column_name)

    def min_value_tool(column_name: str):
        """Gibt den niedrigsten Wert einer Spalte zurück."""
        return get_min_value(df, column_name)

    def outliers_above_tool(column_name: str, threshold_value: float):
        """Gibt alle Werte über einem Schwellenwert zurück."""
        return detect_outliers_above(df, column_name, threshold_value)

    def outliers_below_tool(column_name: str, threshold_value: float):
        """Gibt alle Werte unter einem Schwellenwert zurück."""
        return detect_outliers_below(df, column_name, threshold_value)

    # Tools für LangChain vorbereiten
    tools = [
        StructuredTool.from_function(max_value_tool, name="get_max_value"),
        StructuredTool.from_function(min_value_tool, name="get_min_value"),
        StructuredTool.from_function(outliers_above_tool, name="detect_outliers_above"),
        StructuredTool.from_function(outliers_below_tool, name="detect_outliers_below"),
    ]

    # System-Prompt erstellen
    column_list = ", ".join(df.columns)
    system_prompt = SystemMessage(content=f"""
Du bist ein statistischer Analyse-Assistent.

Der DataFrame 'sensor_data' hat folgende Spalten:
{column_list}

Spaltennamen können auf Deutsch oder Englisch vorkommen (z.B. 'temperature' ↔ 'Temperatur').

Deine Aufgaben:
- Erkenne, ob der Benutzer Maximum, Minimum oder Ausreißer möchte
- Nutze IMMER die passenden Tools
- Wenn der Nutzer einen Spaltennamen nennt, der nicht exakt existiert:
    → suche nach dem ähnlichsten passenden Spaltennamen
""")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        system_prompt,
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    chat_history = ChatMessageHistory()
    memory = ConversationBufferMemory(chat_memory=chat_history, memory_key="chat_history", return_messages=True)

    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
    return agent_executor


__all__ = ["create_ano_agent"]



