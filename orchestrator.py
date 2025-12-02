
import os
import pandas as pd
from dotenv import load_dotenv
from streampipes.client import StreamPipesClient
from streampipes.client.config import StreamPipesClientConfig
from streampipes.client.credential_provider import StreamPipesApiKeyCredentials

# =============================
# .env laden
# =============================
load_dotenv()

SP_USER = os.getenv("SP_USER")
SP_API_KEY = os.getenv("SP_API_KEY")

if not SP_USER or not SP_API_KEY:
    raise ValueError("SP_USER oder SP_API_KEY fehlt in .env!")

# =============================
# StreamPipes Client
# =============================
config = StreamPipesClientConfig(
    credential_provider=StreamPipesApiKeyCredentials.from_env(
        username_env="SP_USER",
        api_key_env="SP_API_KEY"
    ),
    host_address="localhost",
    https_disabled=True,
    port=80
)

sp_client = StreamPipesClient(client_config=config)

# =============================
# Agenten importieren
# ============================= #das funktioniert natürlich nicht wenn die anderen beiden Dateien nicht da sind 
from agents.AnoAgent import agent_executor as data_agent

#from agents.data_analysis_agent import agent_executor as data_agent
from agents.physics_agent import run_agent as physics_agent
#from agents.plot_agent import route_tool as plot_agent

# =============================
# Orchestrator Klasse
# =============================
class OrchestratorAgent:
    def __init__(self, measure: str = "sensor_data", limit: int = 1000):
        # Daten aus StreamPipes laden
        self.df = sp_client.dataLakeMeasureApi.get(measure, limit=limit).to_pandas()
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')
        print(f"Orchestrator gestartet: {len(self.df)} Datenpunkte geladen.")

    def run(self, user_input: str):
        user_input_lower = user_input.lower()

        if any(k in user_input_lower for k in ["max", "min", "outlier"]):
            return data_agent.invoke({"input": user_input})["output"]

        elif any(k in user_input_lower for k in ["correlation", "physics", "formula"]):
            return physics_agent(user_input, self.df)

        #elif any(k in user_input_lower for k in ["plot", "chart", "graph"]):
         #   plan = plot_agent.llm_decide(user_input)
          #  return plot_agent.route_tool(plan["tool"], plan["args"])["path"]

        else:
            return "Kein passender Agent gefunden."

# =============================
# Interaktive CLI
# =============================
if __name__ == "__main__":
    orchestrator = OrchestratorAgent(measure="sensor_data", limit=1000)

    while True:
        user_input = input(">> ").strip()
        if user_input.lower() in ["exit", "quit", "stop"]:
            print("Orchestrator beendet.")
            break
        output = orchestrator.run(user_input)
        print("Antwort des Orchestrators:\n", output)

##############
#__init__.py brauche ich die Datei?


##############
#angeblich unsafed changes vom Anno Agenten wahrscheinlich weil ich versucht habe das an den Orchestrator anzupassen 
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
sp_user = os.getenv("SP_USER")
sp_api_key = os.getenv("SP_API_KEY")

if openai_key is None:
    raise ValueError("OPENAI_API_KEY nicht gefunden! Bitte in .env eintragen.")

if sp_user is None:
    raise ValueError("SP_USER nicht gefunden! Bitte in .env eintragen.")

if sp_api_key is None:
    raise ValueError("SP_API_KEY nicht gefunden! Bitte in .env eintragen.")


# =============================
#  StreamPipes Zugang
# =============================
from streampipes.client import StreamPipesClient
from streampipes.client.config import StreamPipesClientConfig
from streampipes.client.credential_provider import StreamPipesApiKeyCredentials

# ONLY these two env vars are used now
os.environ["SP_USER"] = sp_user
os.environ["SP_API_KEY"] = sp_api_key

config = StreamPipesClientConfig(
    credential_provider=StreamPipesApiKeyCredentials.from_env(
        username_env="SP_USER",
        api_key_env="SP_API_KEY"
    ),
    host_address="localhost",
    https_disabled=True,
    port=80
)

sp_client = StreamPipesClient(client_config=config)


# =============================
#  Daten abrufen
# =============================
df_pressure = sp_client.dataLakeMeasureApi.get("pressure-1", limit=1000).to_pandas()
df_pressure['timestamp'] = pd.to_datetime(df_pressure['timestamp'], unit='ms')

df_flow = sp_client.dataLakeMeasureApi.get("flow-rate", limit=1000).to_pandas()
df_flow['timestamp'] = pd.to_datetime(df_flow['timestamp'], unit='ms')

print(df_pressure.head())
print(df_flow.head())


# =============================
#  Funktionen (Tools)
# =============================
def get_max_value(df_name: str, column_name: str):
    """Gibt den höchsten Wert einer Spalte in einem DataFrame zurück."""
    df_map = {"pressure": df_pressure, "flow": df_flow}
    df = df_map.get(df_name.lower())
    if df is None:
        return f"DataFrame {df_name} existiert nicht."
    if column_name not in df.columns:
        return f"Spalte {column_name} existiert nicht in DataFrame {df_name}."
    value = df[column_name].max()
    return f"Der höchste Wert in '{column_name}' von '{df_name}' beträgt {float(value):.2f}."

def get_min_value(df_name: str, column_name: str):
    """Gibt den niedrigsten Wert einer Spalte in einem DataFrame zurück."""
    df_map = {"pressure": df_pressure, "flow": df_flow}
    df = df_map.get(df_name.lower())
    if df is None:
        return f"DataFrame {df_name} existiert nicht."
    if column_name not in df.columns:
        return f"Spalte {column_name} existiert nicht in DataFrame {df_name}."
    value = df[column_name].min()
    return f"Der niedrigste Wert in '{column_name}' von '{df_name}' beträgt {float(value):.2f}."

def detect_outliers_above(df_name: str, column_name: str, threshold_value: float):
    """Findet alle Werte über einem Schwellenwert in einer Spalte."""
    df_map = {"pressure": df_pressure, "flow": df_flow}
    df = df_map.get(df_name.lower())
    if df is None:
        return f"DataFrame {df_name} existiert nicht."
    if column_name not in df.columns:
        return f"Spalte {column_name} existiert nicht in DataFrame {df_name}."
    outliers = df[df[column_name] > threshold_value]
    if outliers.empty:
        return f"Keine Ausreißer über {threshold_value} gefunden."
    return outliers[[column_name, "timestamp"]].to_dict(orient="records")

def detect_outliers_below(df_name: str, column_name: str, threshold_value: float):
    """Findet alle Werte unter einem Schwellenwert in einer Spalte."""
    df_map = {"pressure": df_pressure, "flow": df_flow}
    df = df_map.get(df_name.lower())
    if df is None:
        return f"DataFrame {df_name} existiert nicht."
    if column_name not in df.columns:
        return f"Spalte {column_name} existiert nicht in DataFrame {df_name}."
    outliers = df[df[column_name] < threshold_value]
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
system_prompt = SystemMessage(content="""
Du bist ein statistischer Analyse-Assistent.
Erkenne:
- DataFrames automatisch ('pressure', 'flow')
- Spalten automatisch
- was der Benutzer möchte (Maximum, Minimum, Ausreißer)
Nutze IMMER die passenden Tools.
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


# =============================
#  Interaktive Schleife
# =============================
print("Chat-Agent bereit! Tippe 'exit' zum Beenden.\n")

while True:
    user_input = input(">> ").strip()
    if user_input.lower() in ["exit", "quit", "stop"]:
        print("Chat beendet.")
        break
    response = agent_executor.invoke({"input": user_input})
    print("\nAntwort des Agents:\n", response["output"], "\n")