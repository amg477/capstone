import io
import pandas as pd
import streamlit as st
from azure.storage.blob import BlobServiceClient

# ---------- Page Setup ----------
st.set_page_config(page_title="Penta Capstone Explorer",
                   layout="wide",
                   page_icon="📊")

st.title("Penta Capstone Data Explorer")

# ---------- Helper Functions ----------
@st.cache_data(show_spinner="Downloading data from Azure…", ttl=3600)
def load_data() -> pd.DataFrame:
    """Download the CSV from Azure Blob Storage using a SAS token."""
    account   = st.secrets["AZURE_ACCOUNT"]
    container = st.secrets["AZURE_CONTAINER"]
    sas_token = st.secrets["AZURE_SAS"]
    blob_path = st.secrets["AZURE_BLOB_PATH"]

    # Create client using the SAS token
    service_client = BlobServiceClient(
        account_url=f"https://{account}.blob.core.windows.net",
        credential=sas_token
    )
    blob_client = service_client.get_blob_client(container=container, blob=blob_path)

    # Download to memory
    byte_stream = io.BytesIO(blob_client.download_blob().readall())

    # Adjust read method if you switch to Parquet
    df = pd.read_csv(byte_stream)
    return df

# ---------- Load & Display ----------
try:
    df = load_data()
    st.success(f"Loaded {len(df):,} rows from Azure Blob Storage ✅")
    st.dataframe(df.head(50), use_container_width=True)
except Exception as e:
    st.error("⚠️ Failed to load data. Check secrets, SAS token, or blob path.")
    st.exception(e)

# ---------- Optional Filters / UI ----------
st.markdown("### Quick Filters")
if "column_selector" not in st.session_state:
    st.session_state.column_selector = df.columns[0]

col = st.selectbox("Filter column:", df.columns, index=0, key="column_selector")
val = st.text_input("Substring match:")
if val:
    filtered = df[df[col].astype(str).str.contains(val, case=False)]
    st.write(f"Showing {len(filtered):,} matching rows")
    st.dataframe(filtered, use_container_width=True)

# Allow CSV download of filtered data
st.download_button(
    label="Download filtered data as CSV",
    data=filtered.to_csv(index=False).encode("utf-8") if val else df.to_csv(index=False).encode("utf-8"),
    file_name="capstone_filtered.csv",
    mime="text/csv"
)