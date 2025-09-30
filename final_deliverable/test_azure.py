#!/usr/bin/env python3
"""
Test Azure Connection for PolicyPath
Run this to verify your Azure setup is working.
"""

from azure.storage.blob import BlobServiceClient

def test_azure_connection():
    print("🧪 Testing Azure Connection...")
    
    # Get connection string from user
    connection_string = input("Enter your Azure connection string: ").strip()
    container_name = "data"
    
    try:
        # Test connection
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        print("✅ Connection successful!")
        
        # List blobs in container
        print(f"\n📁 Files in '{container_name}' container:")
        blobs = container_client.list_blobs()
        blob_count = 0
        for blob in blobs:
            print(f"  - {blob.name} ({blob.size} bytes)")
            blob_count += 1
        
        if blob_count == 0:
            print("  (No files found - you need to upload your data first)")
            print("\n💡 Run: python upload_to_azure.py")
        else:
            print(f"\n✅ Found {blob_count} files in Azure!")
            print("\n🎉 Your Azure setup is ready!")
            print("Now update your Streamlit Cloud secrets with this connection string.")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check your connection string is correct")
        print("2. Make sure the 'data' container exists")
        print("3. Verify your storage account is active")

if __name__ == "__main__":
    test_azure_connection()
