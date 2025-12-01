import os
from typing import Dict, Any
import pandas as pd
import requests

SP_HOST = "http://localhost:8030"
SP_BASE_API = f"{SP_HOST}/streampipes-backend/api/v4"

SP_USERNAME = "admin@streampipes.apache.org"
SP_API_KEY = "eIIMPM)@xljkKZZba+Pj4Yx6"


def _make_headers() -> dict:
    return {
        "X-API-USER": SP_USERNAME,
        "X-API-KEY": SP_API_KEY,
        "Accept": "application/json",
    }


def _extract_query_result_v97(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Speziell für StreamPipes 0.97.x:
    Das Data-Lake-Format lautet:

    {
      "headers": [...],
      "allDataSeries": [
        {
          "rows": [...],
          "tags": {...}
        }
      ],
      "total": ...,
      ...
    }
    """
    if "headers" not in data:
        raise ValueError("Feld 'headers' fehlt in Antwort.")

    if "allDataSeries" not in data:
        raise ValueError("Feld 'allDataSeries' fehlt in Antwort.")

    if not data["allDataSeries"]:
        raise ValueError("allDataSeries ist leer – keine Daten vorhanden.")

    first_series = data["allDataSeries"][0]

    if "rows" not in first_series:
        raise ValueError("In allDataSeries[0] fehlt das Feld 'rows'.")

    return {
        "headers": data["headers"],
        "rows": first_series["rows"],
    }


def load_measure_as_df(measure_id: str, limit: int = 10000) -> pd.DataFrame:
    url = f"{SP_BASE_API}/datalake/measurements/{measure_id}"
    params = {"limit": limit}

    resp = requests.get(url, headers=_make_headers(), params=params)
    resp.raise_for_status()

    data = resp.json()

    # Spezielles Format für StreamPipes 0.97.x extrahieren
    qr = _extract_query_result_v97(data)

    headers = qr["headers"]
    rows = qr["rows"]

    df = pd.DataFrame(rows, columns=headers)

    # Timestamp-Spalte finden und konvertieren
    ts_cols = [c for c in df.columns if "timestamp" in c.lower() or c.lower() == "time"]
    if ts_cols:
        ts = ts_cols[0]
        try:
            df[ts] = pd.to_datetime(df[ts], unit="ms", errors="ignore")
        except Exception:
            pass

    return df


if __name__ == "__main__":
    df = load_measure_as_df("sensor_data", limit=10000)
    print(df.head())
    print("Shape:", df.shape)