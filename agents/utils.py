# utils.py
import os
from typing import Optional, List
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()


def get_azure_llm(temperature: Optional[float] = None, callbacks: Optional[List[BaseCallbackHandler]] = None):
    """
    Returns a configured AzureChatOpenAI instance with optional callbacks.

    Args:
        temperature: Model temperature (default: None, uses model's default)
        callbacks: Optional list of callback handlers (e.g., Langfuse)
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

    llm_kwargs = {
        "azure_deployment": os.getenv("AZURE_AI_DEPLOYMENT", "o4-mini"),
        "model_name": model_name,
        "api_version": os.getenv("AZURE_AI_API_VERSION", "2024-12-01-preview"),
        "azure_endpoint": endpoint,
        "api_key": credential,
    }

    if temperature is not None:
        llm_kwargs["temperature"] = temperature

    if callbacks:
        llm_kwargs["callbacks"] = callbacks

    return AzureChatOpenAI(**llm_kwargs)
