import os
import pandas as pd
from dotenv import load_dotenv

from streampipes.client import StreamPipesClient
from streampipes.client.config import StreamPipesClientConfig
from streampipes.client.credential_provider import StreamPipesApiKeyCredentials

# Load .env
load_dotenv()

# Output path for the unified CSV file
DATA_PATH = "data/sample_sensor_data.csv"

#  Create StreamPipes Client

def get_local_streampipes_client() -> StreamPipesClient:
    username = os.getenv("SP_USERNAME")
    api_key = os.getenv("SP_API_KEY")

    if not username or not api_key:
        raise ValueError("❌ SP_USERNAME oder SP_API_KEY fehlt in .env")

    config = StreamPipesClientConfig(
        credential_provider=StreamPipesApiKeyCredentials(
            username=username,
            api_key=api_key
        ),
        host_address="localhost",
        https_disabled=True,
        port=80,
    )

    return StreamPipesClient(client_config=config)

#  Load sensor data from StreamPipes

def load_sensor_data_from_sp(client: StreamPipesClient) -> pd.DataFrame:
    print("⏳ Lade Sensordaten aus StreamPipes...")

    measure = client.dataLakeMeasureApi.get(identifier="sensor_data")
    df = measure.to_pandas()

    # Normalize timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    print(f"✓ {len(df)} Datenpunkte geladen")
    print(f"✓ Spalten: {', '.join(df.columns)}")
    return df

#  Save DataFrame to CSV

def save_sensor_data_to_csv(df: pd.DataFrame, file_path: str = DATA_PATH):
    os.makedirs("data", exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"💾 CSV gespeichert unter: {file_path}")


#  Load DataFrame from CSV (Agents nutzen das!)

def load_sensor_data_from_csv(file_path: str = DATA_PATH) -> pd.DataFrame:
    print(f"📂 Lade Sensordaten aus CSV: {file_path}")
    df = pd.read_csv(file_path)

    # Timestamp reconversion
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(f"✓ {len(df)} Datenpunkte geladen")
    return df


if __name__ == "__main__":
    print("🚀 Starte Datenpipeline...")

    client = get_local_streampipes_client()
    df = load_sensor_data_from_sp(client)

    save_sensor_data_to_csv(df, DATA_PATH)

    print("🏁 Pipeline abgeschlossen.")