import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from azure.storage.blob import BlobServiceClient, BlobClient
from dotenv import load_dotenv

# Always load .env next to this file
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Two supported modes:
# A) Connection string + container + blob path
CONN_STR   = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER  = os.getenv("CONTAINER_NAME")
BLOB_PATH  = os.getenv("BLOB_PATH")  # e.g. data/processed/processed_data.csv

# B) Container SAS URL + blob path
CONTAINER_SAS_URL = os.getenv("AZURE_CONTAINER_SAS_URL") or os.getenv("AZURE_BLOB_SAS_URL")

def _download_to(local_path, blob_client: BlobClient):
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(blob_client.download_blob().readall())
    print(f"✅ Downloaded blob to {local_path}")

def main(local_path="data/processed/processed_data.csv"):
    if not BLOB_PATH:
        raise SystemExit("❌ Missing BLOB_PATH in .env (e.g. data/processed/processed_data.csv)")

    # MODE B: Container SAS URL present
    if CONTAINER_SAS_URL:
        # CONTAINER_SAS_URL should look like:
        # https://<acct>.blob.core.windows.net/<container>?<sas>
        parts = urlsplit(CONTAINER_SAS_URL)
        if not parts.query or not parts.path.strip("/"):
            raise SystemExit("❌ AZURE_CONTAINER_SAS_URL must be a container URL with a SAS token.")
        container_path = parts.path.rstrip("/")  # /capstone
        # Build full blob URL: container base + / + blob path + ? + sas
        blob_path = BLOB_PATH.lstrip("/")
        new_path = f"{container_path}/{blob_path}"   # /capstone/data/processed/processed_data.csv
        blob_url = urlunsplit((parts.scheme, parts.netloc, new_path, parts.query, ""))
        blob_client = BlobClient.from_blob_url(blob_url)
        return _download_to(local_path, blob_client)

    # MODE A: Connection string
    if not CONN_STR or not CONTAINER:
        raise SystemExit("❌ Set either (AZURE_CONTAINER_SAS_URL + BLOB_PATH) or (AZURE_STORAGE_CONNECTION_STRING + CONTAINER_NAME + BLOB_PATH) in .env")

    svc = BlobServiceClient.from_connection_string(CONN_STR)
    blob_client = svc.get_blob_client(container=CONTAINER, blob=BLOB_PATH)
    return _download_to(local_path, blob_client)

if __name__ == "__main__":
    main()