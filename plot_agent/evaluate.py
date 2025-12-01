from langsmith import Client
from langsmith.evaluation import evaluate, StringEvaluator
from agent import invoke_agent   # Agent-Wrapper für einen einzelnen Prompt



DATASET_ID = "2191eee3-2b2d-4f25-9a8e-42df6e343f55"
client = Client()



# EVALUATOR: prüft, ob der Agent das richtige Tool gewählt hat

def check_tool_match(output: dict, expected: str, *args, **kwargs):
    """
    output: das vollständige LLM-Output (inkl. Messages)
    expected: erwarteter Toolname (z.B. 'time_series_plot')
    """

    # Suche alle Nachrichten
    msgs = output.get("messages", [])
    for msg in msgs:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            predicted = tool_calls[0]["name"]
            return {
                "score": 1 if predicted == expected else 0,
                "comment": f"expected={expected}, got={predicted}",
            }

    return {"score": 0, "comment": "no tool call found"}


tool_match_eval = StringEvaluator(
    name="tool_match",
    grading_function=check_tool_match,
    input_key="input",
    expected_output_key="expected_tool",
)

# EVALUATION STARTEN

run = evaluate(
    target=invoke_agent,                         # WICHTIG: "target", nicht "model"
    data=client.read_dataset(dataset_id=DATASET_ID),
    evaluators=[tool_match_eval],
    experiment_name="plot-agent-test-v1",
)

print("Evaluation gestartet:")
print(run)