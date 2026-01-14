# agents/langfuse_config.py
"""
Centralized Langfuse configuration and callback handler management.
"""
import os
from typing import Optional
from dotenv import load_dotenv
from langfuse.callback import CallbackHandler

load_dotenv()


def get_langfuse_handler(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    trace_name: Optional[str] = None,
) -> Optional[CallbackHandler]:
    """
    Returns a configured Langfuse CallbackHandler if credentials are available.
    
    Args:
        session_id: Optional session identifier for grouping related traces
        user_id: Optional user identifier
        trace_name: Optional name for the trace
        
    Returns:
        CallbackHandler instance if Langfuse is configured, None otherwise
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    # Return None if Langfuse is not configured (optional dependency)
    if not public_key or not secret_key:
        return None
    
    try:
        handler = CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            session_id=session_id,
            user_id=user_id,
            trace_name=trace_name,
        )
        return handler
    except Exception as e:
        print(f"Warning: Failed to initialize Langfuse handler: {e}")
        return None


def flush_langfuse_handler(handler: Optional[CallbackHandler]):
    """
    Flushes the Langfuse handler to ensure all data is sent.
    
    Args:
        handler: The CallbackHandler to flush
    """
    if handler:
        try:
            handler.flush()
        except Exception as e:
            print(f"Warning: Failed to flush Langfuse handler: {e}")
