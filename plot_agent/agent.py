import os
import subprocess
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# Tools
from plot_tools import (
    time_series_plot,
    histogram_plot,
    scatter_plot,
    boxplot_plot,
    correlation_matrix_plot,
)

# Load environment + LangSmith config

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY fehlt in .env")

if not os.getenv("LANGSMITH_API_KEY"):
    raise RuntimeError("LANGSMITH_API_KEY fehlt in .env – LangSmith benötigt ihn.")

os.environ.setdefault("LANGSMITH_PROJECT", "pipeline_plot_agent")

# Tools definieren

tools = [
    time_series_plot,
    histogram_plot,
    scatter_plot,
    boxplot_plot,
    correlation_matrix_plot,
]

# Agent erstellen 
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=tools,
    system_prompt="""
Du bist ein Datenanalyse-Agent für StreamPipes.

REGELN:
- Das Measure ist IMMER 'sensor_data', außer der Nutzer nennt explizit ein anderes.
- Du verwendest IMMER eines der Tools.
- Tools speichern PNG-Dateien ab. Erfinde keine Werte.
- Gib in deiner finalen Antwort IMMER den Dateipfad an.
"""
)


# PNG-Dateipfad aus ToolMessage extrahieren

def extract_png_path(messages):
    """Durchsuche alle ToolMessages nach einem Pfad zu einer PNG-Datei."""
    for m in messages:
        if hasattr(m, "content") and isinstance(m.content, str):
            if m.content.endswith(".png") or ".png" in m.content:
                # Pfad extrahieren
                start = m.content.find("/")   # erster Slash
                if start != -1:
                    return m.content[start:].strip()
    return None

# CLI Loop

def main():
    print("🚀 Plot-Agent gestartet (LangChain 1.x + LangSmith Tracking aktiv)")
    print("Beispiele:")
    print("- 'Zeig mir die Temperatur über die Zeit'")
    print("- 'Histogramm density'")
    print("- 'Korrelationsmatrix'")
    print("exit zum Beenden.\n")

    while True:
        user_request = input("> ").strip()

        if user_request.lower() in {"exit", "quit"}:
            print("Bye 👋")
            break

        try:
            # Agent ausführen
            result = agent.invoke({
                "messages": [
                    {"role": "user", "content": user_request}
                ]
            })

            # Finale AI-Antwort extrahieren
            final_answer = result["messages"][-1].content

            # Pfad zum PNG extrahieren
            png_path = extract_png_path(result["messages"])

            print("\n🟢 Ausgabe:")
            print(final_answer)

            if png_path:
                print(f"\n📁 Plot gespeichert unter:\n{png_path}")

                # PNG automatisch im Finder öffnen
                if os.path.exists(png_path):
                    print("\n🔍 Öffne Bild im Finder…")
                    subprocess.run(["open", png_path])
                else:
                    print("\n⚠️ PNG nicht gefunden – wurde evtl. verschoben?")

        except Exception as e:
            print("\n❌ Fehler:")
            print(e)


if __name__ == "__main__":
    main()

def invoke_agent(input_str: str):
    """
    Wrapper für die LangSmith-Evaluation.
    Nimmt eine Nutzereingabe (String),
    ruft den Agenten auf,
    und gibt das vollständige Message-Format zurück.
    """
    result = agent.invoke({"messages": [{"role": "user", "content": input_str}]})
    return result