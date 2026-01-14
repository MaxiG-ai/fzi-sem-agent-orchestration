# orchestrator/observability.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Callable

from dotenv import load_dotenv, find_dotenv

_LANGFUSE_CLIENT = None
_LANGFUSE_CB = None


def setup_observability(project_name: str | None = None) -> None:
    """
    Loads env + configures LangSmith/Langfuse indirectly via env vars.
    Safe to call multiple times.
    """
    # make dotenv robust when running "python -m orchestrator.router"
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
    else:
        load_dotenv(override=False)

    # Optional: you can set LangSmith project here if you want
    # (LangSmith is usually controlled via env vars already)
    if project_name and not os.getenv("LANGSMITH_PROJECT"):
        # only set if not present
        os.environ["LANGSMITH_PROJECT"] = project_name


def _langfuse_env_ok() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(os.getenv("LANGFUSE_SECRET_KEY"))


def get_langfuse_client():
    """
    Returns Langfuse client if configured, else None.
    """
    global _LANGFUSE_CLIENT

    if _LANGFUSE_CLIENT is not None:
        return _LANGFUSE_CLIENT

    if not _langfuse_env_ok():
        return None

    try:
        # preferred client creation in newer SDKs
        from langfuse import get_client

        lf = get_client()  # reads env: LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST
        # verify auth (if available)
        try:
            if not lf.auth_check():
                return None
        except Exception:
            # auth_check might not exist in older versions
            pass

        _LANGFUSE_CLIENT = lf
        return _LANGFUSE_CLIENT
    except Exception:
        return None


def get_langfuse_callbacks() -> Optional[List[object]]:
    """
    LangChain/LangGraph callback handler for Langfuse.
    Returns None if Langfuse is not installed or keys missing.
    """
    global _LANGFUSE_CB

    if _LANGFUSE_CB is not None:
        return [_LANGFUSE_CB]

    if not _langfuse_env_ok():
        return None

    try:
        # IMPORTANT: this is the LangChain integration path
        from langfuse.langchain import CallbackHandler  # type: ignore

        _LANGFUSE_CB = CallbackHandler()  # reads env automatically
        return [_LANGFUSE_CB]
    except Exception:
        return None


def get_langfuse_observe():
    """
    Returns Langfuse 'observe' decorator if available, else a no-op decorator.
    """
    if _langfuse_env_ok():
        try:
            from langfuse.decorators import observe  # type: ignore
            return observe
        except Exception:
            pass

    # no-op decorator fallback
    def _noop_decorator(*dargs, **dkwargs):
        def _wrap(fn):
            return fn
        if dargs and callable(dargs[0]) and not dkwargs:
            return dargs[0]
        return _wrap

    return _noop_decorator


def flush_langfuse() -> None:
    """
    Flushes pending events (best-effort).
    """
    lf = get_langfuse_client()
    if not lf:
        return
    try:
        lf.flush()
    except Exception:
        pass
    # --- add to orchestrator/observability.py ---

from typing import Any, Dict, Optional

def log_generation(
    trace_or_span,
    name: str,
    model: str,
    input: Any,
    output: Any,
    usage: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Log an LLM call as a Langfuse 'generation'.
    Works for both trace or span objects (Langfuse SDK).
    """
    if not trace_or_span:
        return None
    try:
        return trace_or_span.generation(
            name=name,
            model=model,
            input=input,
            output=output,
            usage=usage or {},
            metadata=metadata or {},
        )
    except Exception:
        return None