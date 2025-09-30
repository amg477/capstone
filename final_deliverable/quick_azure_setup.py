#!/usr/bin/env python3
"""
Quick Azure Setup for PolicyPath
This script will help you upload your data to Azure and get the connection string.
"""

import os
from pathlib import Path

def main():
    print("🚀 PolicyPath Azure Quick Setup")
    print("=" * 50)
    
    print("\n📋 STEP 1: Create Azure Storage Account")
    print("1. Go to: https://portal.azure.com")
    print("2. Click 'Create a resource' → 'Storage account'")
    print("3. Fill in:")
    print("   - Storage account name: policy-path-storage (or any unique name)")
    print("   - Resource group: Create new")
    print("   - Location: Choose closest to you")
    print("   - Performance: Standard")
    print("   - Redundancy: LRS")
    print("4. Click 'Review + create' → 'Create'")
    
    print("\n📦 STEP 2: Create Container")
    print("1. Go to your storage account")
    print("2. Click 'Containers' → '+ Container'")
    print("3. Name: 'data'")
    print("4. Public access: 'Private'")
    print("5. Click 'Create'")
    
    print("\n🔑 STEP 3: Get Connection String")
    print("1. Go to 'Access keys' in your storage account")
    print("2. Click 'Show keys'")
    print("3. Copy the 'Connection string' from key1")
    
    print("\n📁 STEP 4: Upload Your Data")
    print("Run this command to upload your files:")
    print("python upload_to_azure.py")
    print("\nMake sure to update the CONNECTION_STRING in upload_to_azure.py first!")
    
    print("\n⚙️ STEP 5: Configure Streamlit Cloud")
    print("1. Go to your Streamlit Cloud app")
    print("2. Click 'Settings' → 'Secrets'")
    print("3. Add this configuration:")
    print("""
[data]
mode = "azure"
AZURE_STORAGE_CONNECTION_STRING = "your_connection_string_here"
container = "data"
parquet_blob = "data/final_model_dataset.parquet"
csv_blob = "data/final_model_dataset.csv"
attr_blob = "data/attribution_all_scored.csv"
logo_blob = "data/penta_logo.png"
    """)
    
    print("\n✅ STEP 6: Deploy!")
    print("Save the secrets and your app will automatically redeploy with the full dataset!")
    
    print("\n💡 Need help? Check AZURE_SETUP.md for detailed instructions.")

if __name__ == "__main__":
    main()
