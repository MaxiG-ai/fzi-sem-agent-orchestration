# streampipes_data.py

from streampipes.client import StreamPipesClient
from streampipes.client.config import StreamPipesClientConfig
from streampipes.client.credential_provider import StreamPipesApiKeyCredentials
import pandas as pd

# ==========================================================
# KONFIGURATION – bitte API KEY prüfen
# ==========================================================

SP_USERNAME = "admin@streampipes.apache.org"
SP_API_KEY = "eIIMPM)@xljkKZZba+Pj4Yx6"
SP_HOST = "localhost"
SP_PORT = 80
SP_HTTPS_DISABLED = True

# ==========================================================
# StreamPipes Client erstellen
# ==========================================================

_config = StreamPipesClientConfig(
    credential_provider=StreamPipesApiKeyCredentials(
        username=SP_USERNAME,
        api_key=SP_API_KEY
    ),
    host_address=SP_HOST,
    https_disabled=SP_HTTPS_DISABLED,
    port=SP_PORT
)

_client = StreamPipesClient(client_config=_config)


# ==========================================================
# DataLake → DataFrame (ALLE Daten)
# ==========================================================

def load_measure_df(measure_id: str) -> pd.DataFrame:
    """
    Lädt das vollständige DataFrame eines DataLake-Measure aus StreamPipes
    und wandelt timestamp zu datetime um.
    """
    measure = _client.dataLakeMeasureApi.get(identifier=measure_id)
    df = measure.to_pandas()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")

    return df