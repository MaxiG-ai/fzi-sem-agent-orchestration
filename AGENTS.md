# Agent Guidelines for FZI SEM Agent Orchestration

This document defines the engineering standards, workflows, and best practices for the `fzi-sem-agent-orchestration` repository. It serves as the authoritative guide for both autonomous coding agents and human developers working in this codebase.

## 1. Project Management & Environment

We utilize **uv** for fast, reliable, and deterministic dependency management.

### Commands

- **Setup & Sync**: Install dependencies defined in `pyproject.toml` and `uv.lock`.

    ```bash
    uv sync
    ```

- **Adding Dependencies**:
  - Runtime: `uv add <package_name>`
  - Dev: `uv add --dev <package_name>`
- **Running Scripts**: Execute python scripts within the isolated environment.

    ```bash
    uv run python orchestrator.py
    ```

- **Dependency Updates**: To upgrade packages.

    ```bash
    uv lock --upgrade
    ```

### Environment Variables

- Configuration is managed via `.env`.
- **Rule**: Never commit `.env` to version control.
- **Setup**: Copy `.env.example` to `.env` and populate required keys.
  - `AZURE_OPENAI_*`: For LLM access.
  - `LANGFUSE_*`: For observability and evaluation.
  - `SP_*`: For StreamPipes connectivity (optional).

## 2. Quality Assurance & Testing

We enforce strict testing and linting protocols to maintain code quality.

### Running Tests (Pytest)

Ensure all new logic is covered by tests.

- **Run All Tests**:

    ```bash
    uv run pytest
    ```

- **Run a Single Test File**:

    ```bash
    uv run pytest tests/path/to/test_file.py
    ```

- **Run a Single Test Case**:

    ```bash
    uv run pytest tests/path/to/test_file.py::test_class_name::test_method_name
    ```

- **Debug Output**:

    ```bash
    uv run pytest -s -v
    ```

### Linting & Formatting (Ruff)

We use **Ruff** for its speed and comprehensive rule set.

- **Check**: Report issues without modifying files.

    ```bash
    uv run ruff check .
    ```

- **Fix**: Automatically fix fixable issues.

    ```bash
    uv run ruff check --fix .
    ```

- **Format**: Enforce code style.

    ```bash
    uv run ruff format .
    ```

## 3. Code Style & Conventions

Adherence to these conventions is mandatory to ensure readability and maintainability.

### Imports

1. **Structure**: Standard Lib -> Third-Party -> Local App.
2. **Absolute Imports**: Always use absolute paths for internal modules.
    - *Correct*: `from agents.utils import get_azure_llm`
    - *Incorrect*: `from ..utils import get_azure_llm`
3. **No Wildcards**: Avoid `from module import *`.

### Naming Conventions

- **Variables/Functions**: `snake_case` (e.g., `calculate_metrics`).
- **Classes/Exceptions**: `PascalCase` (e.g., `RouterState`).
- **Constants**: `UPPER_CASE` (e.g., `DEFAULT_TIMEOUT`).
- **Private Members**: `_leading_underscore` (e.g., `_internal_helper`).
- **Agents**: Entry points should be named `run_<name>_agent`.

### Typing

- Use **Python 3.10+** syntax (e.g., `list[str]` instead of `List[str]`).
- **Mandatory**: Type hints for all function arguments and return values.
- **LangGraph**: Use `typing.TypedDict` for defining graph state schemas.

    ```python
    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]
    ```

### Docstrings

- Use triple double-quotes `"""`.
- Follow Google-style or NumPy-style.
- **Required**: `Args`, `Returns`, and `Raises` sections for complex functions.
- **Tools**: Tool docstrings must be descriptive as they are consumed by the LLM.

## 4. Architecture & Library Usage

### LangChain & LangGraph

- **Orchestration**: Use LangGraph for control flow (`StateGraph`, `START`, `END`).
- **Tools**: Decorate functions with `@tool`. Return simple strings or JSON.
- **State**: Keep state objects immutable where possible; use `Annotated` for reducers.

### Data Handling (Pandas)

- **Efficiency**: Prefer vectorized operations over `iterrows`.
- **Validation**: Check for `df.empty` and handle `NaN` values explicitly.
- **Types**: Ensure correct dtype casting before processing (e.g., `pd.to_datetime`).

### Visualization (Matplotlib)

- **Headless Mode**: **MUST** set backend to `Agg` before importing `pyplot`.

    ```python
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ```

- **Output**: Save plots to a dedicated directory (e.g., `plots/`) and return the absolute path.

## 5. Observability & Evaluations (Langfuse)

This project relies on **Langfuse** for full-stack LLM engineering. Integration is required.

### 1. Observability

- **Tracing**: Decorate all main agent functions with `@observe()`.
- **Context**: Pass `langfuse_handler` to all LLM/Chain invokes via `callbacks`.
- **Metadata**: Add rich metadata (query, tools used) to traces for debugging.

### 2. Prompt Management

- **Single Source of Truth**: Manage prompts in the Langfuse UI, not in code.
- **Fetch Pattern**:

    ```python
    langfuse = get_client()
    prompt_obj = langfuse.get_prompt("prompt_name")
    system_message = prompt_obj.compile(variable_name=value)
    ```

- **Versioning**: Langfuse handles versioning automatically. Always fetch the latest or a pinned production version.

### 3. Experiments & Evals

- **Datasets**: Curate golden datasets in Langfuse (`fzi_sem_ds`).
- **Runner**: Use `langfuse.run_experiment()` to test agent performance.
- **Evaluators**: Implement deterministic (code-based) or model-based (LLM-as-a-judge) evaluators.
  - *Example*: Check if output contains a specific keyword or if JSON is valid.

## 6. Error Handling Strategy

- **Graceful Degradation**: Agents should not crash on tool errors.
- **Tool Errors**: Return a descriptive error string (e.g., `"Error: Column not found"`) to the LLM so it can attempt a correction.
- **Exceptions**: Use specific `try...except` blocks. Log errors using `logging` (or Langfuse trace levels), not `print`.

## 7. Version Control & Git

- **Branching**: Use feature branches (`feat/...`, `fix/...`).
- **Commits**: Follow Conventional Commits.
  - `feat: integrate langfuse prompts`
  - `fix: resolve matplotlib backend issue`
  - `docs: update agents.md`
- **PRs**: Provide context and link to tickets/issues.

## 8. Repository Structure Reference

- `orchestrator.py`: Main entry point. Defines the router agent and graph.
- `agents/`:
  - `physics_agent.py`: Physics analysis tools and agent logic.
  - `plot_agent.py`: Visualization tools (Matplotlib).
  - `statistics_agent.py`: Statistical calculations (Pandas).
  - `utils.py`: Shared utilities (LLM setup).
  - `langfuse_config.py`: Centralized Langfuse configuration/handler.
- `data/`:
  - `sp_data.py`: Data loading logic (StreamPipes/CSV).
- `scripts/`:
  - `upload_subagent_prompts.py`: Utility to sync prompts to Langfuse.
- `AGENTS.md`: This file.

---
Generated by opencode for the FZI SEM Agent Orchestration Team
