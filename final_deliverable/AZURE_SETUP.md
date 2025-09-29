# Azure Blob Storage Setup for PolicyPath

## Step 1: Create Azure Storage Account

1. Go to [Azure Portal](https://portal.azure.com)
2. Click "Create a resource" → "Storage account"
3. Fill in:
   - **Subscription**: Your subscription
   - **Resource group**: Create new or use existing
   - **Storage account name**: `policy-path-storage` (must be globally unique)
   - **Region**: Choose closest to you
   - **Performance**: Standard
   - **Redundancy**: LRS (Locally-redundant storage)
4. Click "Review + create" → "Create"

## Step 2: Create Container

1. Go to your storage account
2. Click "Containers" in the left menu
3. Click "+ Container"
4. Name: `data`
5. Public access level: "Private (no anonymous access)"
6. Click "Create"

## Step 3: Upload Your Data Files

1. Go to the `data` container
2. Click "Upload" → "Upload files"
3. Upload these files:
   - `final_model_dataset.csv` (your 1.3GB file)
   - `attribution_all_scored.csv`
   - `penta_logo.png` (optional)

## Step 4: Get Connection String

1. Go to your storage account
2. Click "Access keys" in the left menu
3. Click "Show keys"
4. Copy the "Connection string" from key1 or key2

## Step 5: Configure Streamlit Cloud

1. Go to your Streamlit Cloud app
2. Click "Settings" (gear icon)
3. Go to "Secrets" tab
4. Replace the placeholder values:

```toml
[data]
mode = "azure"
AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=your-storage-account;AccountKey=your-key;EndpointSuffix=core.windows.net"
container = "data"
parquet_blob = "data/final_model_dataset.parquet"
csv_blob = "data/final_model_dataset.csv"
attr_blob = "data/attribution_all_scored.csv"
logo_blob = "data/penta_logo.png"
```

## Step 6: Deploy

1. Save the secrets
2. The app will automatically redeploy
3. It will download your full dataset from Azure!

## Benefits of Azure Setup

- ✅ **Full dataset** (379K rows) instead of sample
- ✅ **Fast loading** from Azure CDN
- ✅ **No GitHub file size limits**
- ✅ **Secure** with private containers
- ✅ **Scalable** for future growth

## Cost Estimate

- Storage: ~$0.02/month for 1.3GB
- Data transfer: ~$0.01/GB
- **Total: <$1/month** for typical usage
