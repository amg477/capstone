import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read values from .env
conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("CONTAINER_NAME")
blob_path = os.getenv("BLOB_PATH")

def download_blob(local_path="data/processed/processed_data.csv"):
    """Download a blob from Azure Storage to a local file."""
    blob_service_client = BlobServiceClient.from_connection_string(conn_str)
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_path
    )

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(blob_client.download_blob().readall())

    print(f"✅ Downloaded blob to {local_path}")

if __name__ == "__main__":
    download_blob()