from agents.AnoAgent import agent_executor

# Beispiel: Maximum einer Spalte abfragen
response = agent_executor.invoke({"input": "Zeige mir das Maximum der Temperatur"})
print(response["output"])

# Beispiel: Minimum einer Spalte abfragen
response = agent_executor.invoke({"input": "Zeige mir das Minimum der Luftfeuchtigkeit"})
print(response["output"])

# Beispiel: Ausreißer über einem Wert
response = agent_executor.invoke({"input": "Zeige Ausreißer über 30 für die Temperatur"})
print(response["output"])
