import pandas as pd
import os
from dotenv import load_dotenv

from streampipes.client import StreamPipesClient
from streampipes.client.config import StreamPipesClientConfig
from streampipes.client.credential_provider import StreamPipesApiKeyCredentials

load_dotenv()

FILE_PATH = "data/sample_sensor_data.csv"

def get_local_streampipes_client():
    config = StreamPipesClientConfig(
        credential_provider=StreamPipesApiKeyCredentials(
            username=os.getenv("SP_USERNAME"),
            api_key=os.getenv("SP_API_KEY"),
        ),
        host_address="localhost",
        https_disabled=True,  # Set to False if using HTTPS
        port=80,
    )


    # Initialize the StreamPipes client
    client = StreamPipesClient(client_config=config)
    return client

def load_sensor_data_from_sp(client: StreamPipesClient) -> pd.DataFrame:
    # Fetch the specific measure
    datalake_measure = client.dataLakeMeasureApi.get(identifier="sensor_data")

    # Convert to pandas DataFrame
    df = datalake_measure.to_pandas()

    # Convert timestamp to datetime for proper time-series handling
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    print(f"✓ Loaded {len(df)} data points")
    print(f"✓ Columns: {', '.join(df.columns.tolist())}")
    return df

def get_live_sensor_data():
    client = get_local_streampipes_client()
    df = load_sensor_data_from_sp(client)
    return df

def save_sensor_data_to_csv(df: pd.DataFrame, file_path: str):
    df.to_csv(file_path, index=False)
    print(f"✓ Sensor data saved to {file_path}")
    
def load_sensor_data_from_csv() -> pd.DataFrame:
    df = pd.read_csv("data/sample_sensor_data.csv")
    print(f"✓ Loaded sensor data from subdir data, {len(df)} data points")
    return df

if __name__ == "__main__":
    # When run as a script, load new data from StreamPipes and save to CSV
    client = get_local_streampipes_client()
    df = load_sensor_data_from_sp(client)   
    save_sensor_data_to_csv(df, FILE_PATH)