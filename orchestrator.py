import os
import pandas as pd
from dotenv import load_dotenv
import json

from openai import OpenAI
#LLM = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =============================
# Environment laden
# =============================
load_dotenv()

if os.getenv("OPENAI_API_KEY") is None:
    raise ValueError("OPENAI_API_KEY fehlt in .env")

# OpenAI Client erstellen
LLM = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# =============================
# Agenten importieren
# =============================
from agents.AnoAgent import create_ano_agent
from agents.physics_agent import run_agent as physics_agent  # physics_agent erwartet DataFrame
from plot_agent.plot_agent import run_plot_agent  # unser neuer Plot-Agent


# =============================
# CSV laden
# =============================
CSV_FILE = "data/sample_sensor_data.csv"
df_sensor = pd.read_csv(CSV_FILE)
print(f"Geladene Spalten: {', '.join(df_sensor.columns)}")

SYSTEM_ROUTING_PROMPT = """
Du bist ein Orchestrator-Agent, der entscheidet, welcher Agent eine Anfrage bearbeitet.

Regeln:
- statistics → für Max, Min, Ausreißer, Median
- physics → für physikalische Berechnungen, Korrelationen, Beziehungen
- plot → für Diagramme, Histogramme, Scatterplots, Zeitreihen

Gib IMMER folgendes JSON zurück:
{
  "agent": "<statistics|physics|plot>"
}

Beispiele:
User: "Zeige die Korrelation zwischen density und level"
Antwort: { "agent": "physics" }

User: "Erstelle ein Histogramm der Temperatur"
Antwort: { "agent": "plot" }

User: "Finde den maximalen Wert der mass_flow"
Antwort: { "agent": "statistics" }
"""



# =============================
# Orchestrator
# =============================
class OrchestratorAgent:
    def __init__(self):
        print("Orchestrator gestartet – aktuell aktiv: AnoAgent.")
        
        # Erstelle Agent-Instanzen und übergebe df_sensor
        self.agents = {
            "statistics": {
                "agent": create_ano_agent(df_sensor),  # Statistik-Agent
            },
            "physics": {
                "agent": lambda user_input: physics_agent(user_input, df_sensor),  # Physics-Agent
            },
            "plot": {
                "agent": lambda user_input: run_plot_agent(user_input, df_sensor),  # Plot-Agent
            },
        }

    # ----------------------------------------------
    def route_to_agent(self, user_input: str):
        """LLM entscheidet, welcher Agent die Anfrage bearbeiten soll."""
        try:
            response = LLM.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_ROUTING_PROMPT},
                    {"role": "user", "content": user_input},
                ],
                temperature=0,
            )
            agent_name = json.loads(response.choices[0].message.content)["agent"]
            return self.agents.get(agent_name)["agent"]
        except Exception as e:
            print("⚠️ Fehler beim LLM-Routing:", e)
            return None
    
    #print(json.loads('{"agent":"plot"}')["agent"])


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

