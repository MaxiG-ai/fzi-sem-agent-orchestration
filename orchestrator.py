import os
import pandas as pd
from dotenv import load_dotenv

# =============================
# Environment laden
# =============================
load_dotenv()

if os.getenv("OPENAI_API_KEY") is None:
    raise ValueError("OPENAI_API_KEY fehlt in .env")

# =============================
# Agenten importieren
# =============================
from agents.AnoAgent import create_ano_agent
from agents.physics_agent import run_agent as physics_agent  # physics_agent erwartet DataFrame

# =============================
# CSV laden
# =============================
CSV_FILE = "data/sample_sensor_data.csv"
df_sensor = pd.read_csv(CSV_FILE)
print(f"Geladene Spalten: {', '.join(df_sensor.columns)}")



# =============================
# Orchestrator
# =============================
class OrchestratorAgent:
    def __init__(self):
        print("Orchestrator gestartet – aktuell aktiv: AnoAgent.")
        
        # Erstelle Agent-Instanzen und übergebe df_sensor
        self.agents = {
            "statistics": {
                "keywords": ["max", "min", "outlier", "ausreißer", "median", "mittelwert"],
                "agent": create_ano_agent(df_sensor),  # Übergabe des DataFrames
            },
            "physics": {
                "keywords": ["force", "energie", "physik", "formula", "correlation", "relationship"],
                "agent": lambda user_input: physics_agent(user_input, df_sensor),  # Übergabe des DataFrames
            },
        }

    # ----------------------------------------------
    def route_to_agent(self, user_input: str):
        """Durchsucht alle Agenten nach passenden Keywords"""
        user_text = user_input.lower()
        for agent_name, entry in self.agents.items():
            if any(keyword in user_text for keyword in entry["keywords"]):
                return entry["agent"]
        return None

    # ----------------------------------------------
    def run(self, user_input: str):
        """Hauptlogik: wählt Agent aus und führt ihn aus"""
        agent = self.route_to_agent(user_input)

        if agent is None:
            return "❌ Kein passender Agent gefunden."

        # Prüfe, ob es ein AgentExecutor (AnoAgent) oder Lambda (Physics-Agent) ist
        if hasattr(agent, "invoke"):
            result = agent.invoke({"input": user_input})
            return result["output"]
        else:
            # Physics-Agent ist eine Funktion
            return agent(user_input)


# =============================
# CLI zum Testen
# =============================
if __name__ == "__main__":
    orchestrator = OrchestratorAgent()

    while True:
        user_input = input(">> ").strip()
        if user_input.lower() in ["exit", "quit", "stop"]:
            print("Orchestrator beendet.")
            break

        output = orchestrator.run(user_input)
        print("\nAntwort:\n", output, "\n")

