# fzi-sem-agent-orchestration
Repository for FZI Seminar on Agent Orchestration

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
