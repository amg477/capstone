#!/usr/bin/env python3
"""
Script to upload data files to Azure Blob Storage for PolicyPath deployment.
Run this after setting up your Azure Storage Account.
"""

import os
from azure.storage.blob import BlobServiceClient
from pathlib import Path

def upload_to_azure():
    # Configuration - UPDATE THESE VALUES
    CONNECTION_STRING = "your_connection_string_here"  # Get from Azure Portal
    CONTAINER_NAME = "data"
    
    # File paths - adjust if your files are in different locations
    FILES_TO_UPLOAD = {
        "../data/final_model_dataset.csv": "data/final_model_dataset.csv",
        "../data/attribution_all_scored.csv": "data/attribution_all_scored.csv", 
        "penta_logo.png": "data/penta_logo.png"
    }
    
    try:
        # Create blob service client
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        print(f"📦 Uploading files to Azure container: {CONTAINER_NAME}")
        
        for local_path, blob_path in FILES_TO_UPLOAD.items():
            if os.path.exists(local_path):
                print(f"⬆️  Uploading {local_path} → {blob_path}")
                
                with open(local_path, "rb") as data:
                    blob_client = container_client.get_blob_client(blob_path)
                    blob_client.upload_blob(data, overwrite=True)
                
                print(f"✅ Successfully uploaded {blob_path}")
            else:
                print(f"❌ File not found: {local_path}")
        
        print("\n🎉 Upload complete! Your app can now use the full dataset from Azure.")
        print("\nNext steps:")
        print("1. Copy your connection string to Streamlit Cloud secrets")
        print("2. Set mode = 'azure' in your secrets.toml")
        print("3. Redeploy your app")
        
    except Exception as e:
        print(f"❌ Error uploading to Azure: {e}")
        print("\nMake sure you:")
        print("1. Have created an Azure Storage Account")
        print("2. Have created a 'data' container")
        print("3. Have updated the CONNECTION_STRING above")

if __name__ == "__main__":
    print("🚀 PolicyPath Azure Upload Script")
    print("=" * 40)
    upload_to_azure()
