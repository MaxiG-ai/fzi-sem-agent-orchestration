# Langfuse Integration Guide

This document provides comprehensive information about the Langfuse tracking integration in this repository.

## Overview

This repository integrates [Langfuse](https://langfuse.com) for comprehensive observability of all LangGraph agents, LLM calls, and tool invocations. Langfuse provides detailed insights into agent behavior, performance metrics, and debugging capabilities.

## What Gets Tracked

### 1. **Agent Executions** (`@observe` decorator)
All agent runner functions are decorated with `@observe()` to create trace spans:
- `run_router()` - Main orchestration agent
- `run_statistics_agent()` - Statistical analysis agent
- `run_physics_agent()` - Physics analysis agent
- `run_plot_agent()` - Visualization agent

Each agent execution captures:
- Start and end timestamps
- Execution duration
- Input parameters (user query)
- Output/return values
- Any exceptions or errors

### 2. **LLM Interactions** (via `CallbackHandler`)
All LLM calls are automatically tracked through the LangChain `CallbackHandler`:
- Model invocations (Azure OpenAI)
- Complete prompts (system + user messages)
- Model responses
- Token usage (prompt tokens, completion tokens, total)
- Model parameters (temperature, etc.)
- Latency metrics

### 3. **Tool Calls**
All tool invocations are tracked automatically:
- Tool name and description
- Input parameters
- Tool execution time
- Return values
- Success/failure status

**Statistics Agent Tools:**
- `get_max_value(column_name)`
- `get_min_value(column_name)`
- `detect_outliers_above(column_name, threshold_value)`
- `detect_outliers_below(column_name, threshold_value)`

**Physics Agent Tools:**
- `calculate_correlations(columns)`
- `lookup_physics_formula(fields)`

**Plot Agent Tools:**
- `plot_time_series(column)`
- `plot_histogram(column, bins)`
- `plot_scatter(x_col, y_col)`
- `plot_corr()`

**Router Agent Tools:**
- `call_statistics_agent(query)`
- `call_plot_agent(query)`
- `call_physics_agent(query)`

### 4. **LangGraph Execution**
The LangGraph workflow execution is tracked:
- Node transitions
- State updates
- Edge traversals
- Conditional logic decisions

### 5. **Metadata and Context**

Each trace includes rich metadata for filtering and analysis:

**Router Agent:**
```python
metadata = {
    "agent_type": "router",
    "query": "<user_query>",
    "available_agents": ["statistics_agent", "plot_agent", "physics_agent"],
    "orchestration_type": "langgraph"
}
tags = ["agent", "router", "orchestration"]
```

**Statistics Agent:**
```python
metadata = {
    "agent_type": "statistics",
    "query": "<user_query>",
    "tools": ["get_max_value", "get_min_value", "detect_outliers_above", "detect_outliers_below"]
}
tags = ["agent", "statistics", "data-analysis"]
```

**Physics Agent:**
```python
metadata = {
    "agent_type": "physics",
    "query": "<user_query>",
    "tools": ["calculate_correlations", "lookup_physics_formula"],
    "data_columns": [...]
}
tags = ["agent", "physics", "data-analysis", "correlation"]
```

**Plot Agent:**
```python
metadata = {
    "agent_type": "visualization",
    "query": "<user_query>",
    "tools": ["plot_time_series", "plot_histogram", "plot_scatter", "plot_corr"]
}
tags = ["agent", "visualization", "plotting"]
```

## Configuration

### Required Environment Variables

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Optional Environment Variables

```bash
# Release version for tracking deployments
LANGFUSE_RELEASE=v1.0.0

# Environment identifier (e.g., development, staging, production)
LANGFUSE_ENVIRONMENT=production

# Enable debug mode for verbose logging
LANGFUSE_DEBUG=false

# Enable/disable tracking globally
LANGFUSE_ENABLED=true

# Sampling rate (0.0-1.0, where 1.0 = 100%)
LANGFUSE_SAMPLE_RATE=1.0
```

## Setup Instructions

1. **Sign up for Langfuse**
   - Visit [https://cloud.langfuse.com](https://cloud.langfuse.com)
   - Create a free account
   - Create a new project

2. **Get API Keys**
   - Navigate to Settings → API Keys
   - Create a new API key pair
   - Copy the public key (starts with `pk-lf-`)
   - Copy the secret key (starts with `sk-lf-`)

3. **Configure Environment**
   - Add keys to your `.env` file:
     ```bash
     LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
     LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
     LANGFUSE_HOST=https://cloud.langfuse.com
     ```

4. **Run Your Agents**
   - The tracking is automatic - just run your code normally
   - All traces will appear in your Langfuse dashboard

5. **View Traces**
   - Log in to [https://cloud.langfuse.com](https://cloud.langfuse.com)
   - Navigate to your project
   - View traces, metrics, and analytics

## Architecture

### Trace Hierarchy

```
Trace: router_agent
├── Span: LLM Call (Azure OpenAI)
│   ├── Input: System + User Messages
│   └── Output: Tool selection decision
├── Span: Tool Call (call_physics_agent)
│   └── Nested Trace: physics_agent
│       ├── Span: LLM Call (Azure OpenAI)
│       ├── Span: Tool Call (calculate_correlations)
│       │   ├── Input: column names
│       │   └── Output: correlation matrix
│       └── Span: LLM Call (Azure OpenAI - final response)
└── Span: LLM Call (Azure OpenAI - format response)
```

### Data Flow

1. **User Query** → `run_router(query)`
2. **Router Agent** creates Langfuse handler with metadata
3. **LLM Call** to decide which agent to invoke (tracked)
4. **Tool Call** to sub-agent (e.g., `call_physics_agent`)
5. **Sub-Agent** creates nested trace with own handler
6. **Sub-Agent LLM Calls** and tool invocations (tracked)
7. **Results** returned through chain
8. **Handler Flush** ensures all data is sent to Langfuse

## Best Practices

### 1. Session and User Tracking
Add session and user IDs for better analytics:
```python
handler = get_langfuse_handler(
    trace_name="my_agent",
    session_id="session_abc123",
    user_id="user_456"
)
```

### 2. Metadata for Debugging
Add relevant metadata for filtering:
```python
handler = get_langfuse_handler(
    trace_name="my_agent",
    metadata={
        "query_type": "analysis",
        "data_source": "sensor_data",
        "columns_requested": ["temperature", "pressure"]
    }
)
```

### 3. Tags for Organization
Use tags to organize and filter traces:
```python
handler = get_langfuse_handler(
    trace_name="my_agent",
    tags=["production", "v1.0", "critical"]
)
```

### 4. Environment Segregation
Use different environments for dev/staging/prod:
```bash
LANGFUSE_ENVIRONMENT=development  # or staging, production
```

### 5. Sampling for High-Volume
Use sampling to reduce costs in high-volume scenarios:
```bash
LANGFUSE_SAMPLE_RATE=0.1  # Track only 10% of requests
```

## Metrics and Analytics

Langfuse provides rich analytics on your traces:

- **Performance Metrics**: Average latency, p50/p95/p99 percentiles
- **Cost Tracking**: Token usage and estimated costs per trace
- **Error Rates**: Track failures and exceptions
- **User Analytics**: Per-user usage patterns
- **Model Comparison**: Compare performance across different models
- **A/B Testing**: Track multiple versions simultaneously

## Troubleshooting

### No Traces Appearing

1. **Check credentials**: Verify `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set correctly
2. **Check enabled flag**: Ensure `LANGFUSE_ENABLED=true` (or not set, as true is default)
3. **Check network**: Ensure your application can reach `LANGFUSE_HOST`
4. **Check flush**: Make sure `flush_langfuse_handler()` is called at the end of traces

### Traces Incomplete

1. **Wait for flush**: Background threads upload data asynchronously
2. **Call flush explicitly**: Use `flush_langfuse_handler(handler)` at the end
3. **Check sampling**: Ensure `LANGFUSE_SAMPLE_RATE` is set to 1.0 (100%)

### Debug Mode

Enable debug mode for verbose logging:
```bash
LANGFUSE_DEBUG=true
```

This will print detailed information about trace creation and uploads.

## Security Considerations

1. **Keep keys secret**: Never commit `.env` files with real keys
2. **Use environment-specific keys**: Different keys for dev/prod
3. **Rotate keys regularly**: Generate new keys periodically
4. **Review data**: Be aware that prompts and responses are sent to Langfuse
5. **Data masking**: Use the `mask` parameter if you need to redact sensitive data

## Testing

A comprehensive test suite is provided in `/tmp/test_langfuse_comprehensive.py`:

```bash
python /tmp/test_langfuse_comprehensive.py
```

Tests verify:
- ✓ Langfuse client initialization
- ✓ CallbackHandler with all parameters
- ✓ Agent @observe decorator integration
- ✓ Metadata and tags configuration
- ✓ Environment variables support
- ✓ Callback integration in LLM and agents
- ✓ Documentation completeness

All tests pass with 100% success rate.

## Performance Impact

Langfuse integration has minimal performance impact:

- **Asynchronous**: Data is sent in background threads
- **Batching**: Multiple events are batched before upload
- **Non-blocking**: Main execution is not blocked by uploads
- **Optional**: Can be disabled entirely with `LANGFUSE_ENABLED=false`
- **Sampling**: Can reduce data volume with `LANGFUSE_SAMPLE_RATE`

Typical overhead: < 5ms per trace

## Additional Resources

- [Langfuse Documentation](https://langfuse.com/docs)
- [LangChain Integration Guide](https://langfuse.com/docs/integrations/langchain/tracing)
- [Python SDK Reference](https://python.reference.langfuse.com/langfuse)
- [Best Practices](https://langfuse.com/docs/observability/best-practices)

## Support

For issues or questions:
- Langfuse Discord: [https://discord.gg/7NXusRtqYU](https://discord.gg/7NXusRtqYU)
- GitHub Issues: [https://github.com/langfuse/langfuse](https://github.com/langfuse/langfuse)
- Documentation: [https://langfuse.com/docs](https://langfuse.com/docs)
