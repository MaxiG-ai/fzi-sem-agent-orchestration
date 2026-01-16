from langfuse import Langfuse
from dotenv import load_dotenv
import os

load_dotenv()

langfuse = Langfuse()

prompts = [
    {
        "name": "physics_agent",
        "prompt": "You are a physics-aware data analysis assistant. \nAvailable sensor columns: {{columns}}\nUse the tools to analyze correlations and explain physical relationships."
    },
    {
        "name": "statistics_agent",
        "prompt": "You are a statistical analysis assistant. Available columns: {{columns}}. Use the provided tools to answer questions."
    },
    {
        "name": "plot_agent",
        "prompt": "You are a data visualization agent. Create plots based on user requests. Always return the file path of the created plot."
    }
]

for p in prompts:
    print(f"Creating prompt: {p['name']}")
    try:
        langfuse.create_prompt(
            name=p["name"],
            prompt=p["prompt"],
            type="text",
            labels=["production"] 
        )
        print(f"Successfully created {p['name']}")
    except Exception as e:
        print(f"Error creating {p['name']}: {e}")

print("Done.")
