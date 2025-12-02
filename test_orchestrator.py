from orchestrator import OrchestratorAgent

orc = OrchestratorAgent()

print(orc.run("Was ist das Maximum der temperature?"))
print(orc.run("Finde Ausreißer über 70 bei volume_flow"))
print(orc.run("Minimum von density"))
