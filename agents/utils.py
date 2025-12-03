# utils.py
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()


def get_azure_llm(temperature: float = 0):
    """
    Returns a configured AzureChatOpenAI instance.
    """
    credential = os.getenv("AZURE_AI_CREDENTIAL")
    endpoint = os.getenv("AZURE_AI_ENDPOINT")
    model_name = os.getenv("AZURE_AI_MODEL_NAME")
    if not credential:
        raise ValueError("AZURE_AI_CREDENTIAL not found in .env file.")
    if not endpoint:
        raise ValueError("AZURE_AI_ENDPOINT not found in .env file.")
    if not model_name:
        raise ValueError("AZURE_AI_MODEL_NAME not found in .env file.")

    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_AI_DEPLOYMENT", "o4-mini"),
        model_name=model_name,
        api_version=os.getenv("AZURE_AI_API_VERSION", "2024-12-01-preview"),
        azure_endpoint=endpoint,
        api_key=credential,
    )
