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



from data.sp_data import load_sensor_data_from_csv

CSV_FILE = "data/sample_sensor_data.csv"
df_sensor = load_sensor_data_from_csv(CSV_FILE)
# =============================
#  .env laden
# =============================
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
#sp_user = os.getenv("SP_USER")
#sp_api_key = os.getenv("SP_API_KEY")

if openai_key is None:
    raise ValueError("OPENAI_API_KEY nicht gefunden! Bitte in .env eintragen.")


print(df_sensor)
def get_max_value(column_name: str):
    """Gibt den höchsten Wert einer Spalte in df_sensor zurück."""
    if column_name not in df_sensor.columns:
        return f"Spalte '{column_name}' existiert nicht."
    value = df_sensor[column_name].max()
    return f"Der höchste Wert in '{column_name}' beträgt {float(value):.2f}."


def get_min_value(column_name: str):
    """Gibt den niedrigsten Wert einer Spalte in df_sensor zurück."""
    if column_name not in df_sensor.columns:
        return f"Spalte '{column_name}' existiert nicht."
    value = df_sensor[column_name].min()
    return f"Der niedrigste Wert in '{column_name}' beträgt {float(value):.2f}."


def detect_outliers_above(column_name: str, threshold_value: float):
    """Findet alle Werte in df_sensor, die über einem Schwellenwert liegen."""
    if column_name not in df_sensor.columns:
        return f"Spalte '{column_name}' existiert nicht."
    outliers = df_sensor[df_sensor[column_name] > threshold_value]
    if outliers.empty:
        return f"Keine Ausreißer über {threshold_value} gefunden."
    return outliers[[column_name, "timestamp"]].to_dict(orient="records")


def detect_outliers_below(column_name: str, threshold_value: float):
    """Findet alle Werte in df_sensor, die unter einem Schwellenwert liegen."""
    if column_name not in df_sensor.columns:
        return f"Spalte '{column_name}' existiert nicht."
    outliers = df_sensor[df_sensor[column_name] < threshold_value]
    if outliers.empty:
        return f"Keine Ausreißer unter {threshold_value} gefunden."
    return outliers[[column_name, "timestamp"]].to_dict(orient="records")



# =============================
#  Tools
# =============================
tools = [
    StructuredTool.from_function(get_max_value, name="get_max_value"),
    StructuredTool.from_function(get_min_value, name="get_min_value"),
    StructuredTool.from_function(detect_outliers_above, name="detect_outliers_above"),
    StructuredTool.from_function(detect_outliers_below, name="detect_outliers_below"),
]


# =============================
#  System-Prompt
# =============================
#system_prompt = SystemMessage(content="""
#Du bist ein statistischer Analyse-Assistent.
#Erkenne automatisch die Spalten im DataFrame 'sensor_data'.
#Spaltennamen können auf Deutsch oder Englisch sein (z.B. 'temperature', 'Temperatur').
#Finde heraus, was der Benutzer möchte (Maximum, Minimum, Ausreißer)
#und nutze IMMER die passenden Tools.
#Wenn der Nutzer einen Spaltennamen nennt, der nicht exakt existiert, suche nach dem ähnlichsten passenden Spaltennamen.
#""")

# Automatisch Spaltenliste erzeugen
column_list = ", ".join(df_sensor.columns)

system_prompt = SystemMessage(content=f"""
Du bist ein statistischer Analyse-Assistent.

Der DataFrame 'sensor_data' hat folgende Spalten:
{column_list}

Spaltennamen können auf Deutsch oder Englisch vorkommen
(z.B. 'temperature' ↔ 'Temperatur').

Deine Aufgaben:
- Erkenne, ob der Benutzer Maximum, Minimum oder Ausreißer möchte
- Nutze IMMER die passenden Tools
- Wenn der Nutzer einen Spaltennamen nennt, der nicht exakt existiert:
    → suche nach dem ähnlichsten passenden Spaltennamen
""")



# =============================
#  LLM & Prompt
# =============================
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


# =============================
#  Memory
# =============================
chat_history = ChatMessageHistory()
memory = ConversationBufferMemory(chat_memory=chat_history, memory_key="chat_history", return_messages=True)

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)


__all__ = ["agent_executor"]



