import os
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
from agents.AnoAgent import agent_executor as ano_agent
# Platz für spätere Imports:
# from agents.physics_agent import agent_executor as physics_agent
# from agents.plot_agent import agent_executor as plot_agent


# =============================
# Modularer Orchestrator
# =============================
class OrchestratorAgent:
    def __init__(self):
        print("Orchestrator gestartet – aktuell aktiv: AnoAgent.")

        # Registry der Agenten (später einfach erweitern)
        self.agents = {
            "statistics": {
                "keywords": ["max", "min", "outlier", "ausreißer", "median", "mittelwert"],
                "agent": ano_agent,
            },
            # "physics": {
            #     "keywords": ["force", "energie", "physik", "formula", "correlation"],
            #     "agent": physics_agent,
            # },
            # "plot": {
            #     "keywords": ["plot", "diagramm", "chart", "graph"],
            #     "agent": plot_agent,
            # },
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

        # AnoAgent nutzt .invoke()
        result = agent.invoke({"input": user_input})
        return result["output"]


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
