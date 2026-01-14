# fzi-sem-agent-orchestration
Repository for FZI Seminar on Agent Orchestration

## Setup

1. Install dependencies:
```bash
pip install -e .
```

2. Configure environment variables by copying `.env.example` to `.env`:
```bash
cp .env.example .env
```

3. Fill in your credentials in the `.env` file:
   - Azure OpenAI credentials (required)
   - StreamPipes credentials (optional, for live data)
   - Langfuse credentials (optional, for tracking)

## Langfuse Tracking

This repository integrates [Langfuse](https://langfuse.com) for comprehensive tracking of all agent executions, tool calls, and LLM interactions.

### Setup Langfuse

1. Sign up for a free account at [https://cloud.langfuse.com](https://cloud.langfuse.com)
2. Create a new project
3. Copy your API keys and add them to `.env`:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

### What is Tracked

Langfuse tracking provides detailed insights into:

- **Router Agent**: Tracks the main orchestration logic and agent selection
- **Statistics Agent**: Tracks statistical calculations (max, min, outliers)
- **Physics Agent**: Tracks correlation calculations and physics formula lookups
- **Plot Agent**: Tracks visualization generation
- **All Tool Calls**: Each individual tool invocation is tracked with inputs and outputs
- **LLM Interactions**: All model calls with prompts, completions, and token usage

### Viewing Traces

After running the agents, visit your Langfuse dashboard to see:
- Complete execution traces
- Performance metrics
- Token usage statistics
- Error tracking
- Request/response details

### Optional Usage

Langfuse is completely optional. If the credentials are not configured, the agents will work normally without tracking.

## Get data from streampipes

Data can be loaded live from streampipes (when local docker container is running) or from a stored version of the data.

### Load Live data

Live data needs the docker container running and ``SP_USERNAME`` & `SP_API_KEY` set in `.env` .

```python
from data.sp_data import get_live_sensor_data

df = get_live_sensor_data()
```

### Load data from .csv file

A stored sensor data version can also be loaded from a .csv file using:

```python
from data.sp_data import load_sensor_data_from_csv

df = load_sensor_data_from_csv()
```

This file can be updated by running `data/sp_data.py` as a script.
```bash
python data/sp_data.py
```
