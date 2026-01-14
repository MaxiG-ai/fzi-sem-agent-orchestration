# agents/langfuse_config.py
"""
Centralized Langfuse configuration and callback handler management.
Follows best practices from https://langfuse.com/docs/observability/get-started
and https://python.reference.langfuse.com/langfuse
"""
import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from langfuse.callback import CallbackHandler
from langfuse import Langfuse

load_dotenv()


def _get_sample_rate() -> float:
    """
    Helper function to parse LANGFUSE_SAMPLE_RATE from environment.
    
    Returns:
        Sample rate as float (default: 1.0)
    
    Raises:
        ValueError: If LANGFUSE_SAMPLE_RATE is not a valid float
    """
    sample_rate_str = os.getenv("LANGFUSE_SAMPLE_RATE", "1.0")
    try:
        return float(sample_rate_str)
    except ValueError:
        raise ValueError(f"LANGFUSE_SAMPLE_RATE must be a valid float, got: {sample_rate_str}")


def get_langfuse_client() -> Optional[Langfuse]:
    """
    Returns a configured Langfuse client if credentials are available.
    The client is used for direct API interactions and provides more control.
    
    Returns:
        Langfuse client instance if configured, None otherwise
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    # Return None if Langfuse is not configured (optional dependency)
    if not public_key or not secret_key:
        return None
    
    try:
        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            release=os.getenv("LANGFUSE_RELEASE"),
            debug=os.getenv("LANGFUSE_DEBUG", "false").lower() == "true",
            enabled=os.getenv("LANGFUSE_ENABLED", "true").lower() == "true",
            sample_rate=_get_sample_rate(),
            environment=os.getenv("LANGFUSE_ENVIRONMENT", "production"),
        )
        return client
    except ValueError as e:
        print(f"Warning: Invalid Langfuse configuration value: {e}")
        return None
    except Exception as e:
        print(f"Warning: Failed to initialize Langfuse client: {e}")
        return None


def get_langfuse_handler(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    trace_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    version: Optional[str] = None,
) -> Optional[CallbackHandler]:
    """
    Returns a configured Langfuse CallbackHandler for LangChain integration.
    
    The CallbackHandler automatically captures:
    - All LLM calls with prompts, completions, and token usage
    - Tool invocations with inputs and outputs
    - Agent reasoning steps and decisions
    - Chain executions with intermediate steps
    
    Args:
        session_id: Optional session identifier for grouping related traces
        user_id: Optional user identifier for tracking per-user metrics
        trace_name: Optional name for the trace (e.g., "router_agent", "statistics_agent")
        metadata: Optional dictionary of custom metadata to attach to the trace
        tags: Optional list of tags for filtering and organizing traces
        version: Optional version identifier for A/B testing and deployment tracking
        
    Returns:
        CallbackHandler instance if Langfuse is configured, None otherwise
        
    Example:
        ```python
        handler = get_langfuse_handler(
            trace_name="my_agent",
            metadata={"query_type": "statistical_analysis"},
            tags=["production", "v1.0"]
        )
        llm.invoke(messages, config={"callbacks": [handler]})
        ```
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
            metadata=metadata,
            tags=tags,
            version=version,
            release=os.getenv("LANGFUSE_RELEASE"),
            debug=os.getenv("LANGFUSE_DEBUG", "false").lower() == "true",
            enabled=os.getenv("LANGFUSE_ENABLED", "true").lower() == "true",
            sample_rate=_get_sample_rate(),
            environment=os.getenv("LANGFUSE_ENVIRONMENT", "production"),
        )
        return handler
    except ValueError as e:
        print(f"Warning: Invalid Langfuse configuration value: {e}")
        return None
    except Exception as e:
        print(f"Warning: Failed to initialize Langfuse handler: {e}")
        return None


def flush_langfuse_handler(handler: Optional[CallbackHandler]):
    """
    Flushes the Langfuse handler to ensure all data is sent to the backend.
    
    Should be called at the end of a trace to ensure all events are uploaded.
    The handler uses background threads for performance, so flushing ensures
    data is sent before the program exits.
    
    Args:
        handler: The CallbackHandler to flush
        
    Example:
        ```python
        handler = get_langfuse_handler(trace_name="my_trace")
        # ... run your agent/chain ...
        flush_langfuse_handler(handler)  # Ensure data is sent
        ```
    """
    if handler:
        try:
            handler.flush()
        except Exception as e:
            print(f"Warning: Failed to flush Langfuse handler: {e}")


def shutdown_langfuse():
    """
    Shutdown the Langfuse client gracefully.
    
    This should be called when the application is shutting down to ensure
    all pending traces are sent to the backend.
    """
    client = get_langfuse_client()
    if client:
        try:
            client.flush()
        except Exception as e:
            print(f"Warning: Failed to shutdown Langfuse client: {e}")
