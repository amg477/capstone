"""
Data Loading Functions - Simplified for speed
Just load parquet files directly, no conversions
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import os

# Get the directory of this file, then navigate to data directory
APP_DIR = Path(__file__).resolve().parent

# Try multiple possible paths (for local and Streamlit Cloud)
possible_paths = [
    APP_DIR.parent / "data_storage" / "streamlit_app_data",  # Relative from streamlit_app/
    Path("data_storage/streamlit_app_data"),  # From project root (current working directory)
    Path("/mount/src/capstone/data_storage/streamlit_app_data"),  # Streamlit Cloud path
    Path("/Users/annaglass/capstone/capstone/data_storage/streamlit_app_data"),  # Absolute fallback
]

DATA_DIR = None
for path in possible_paths:
    if path.exists():
        DATA_DIR = path
        break

if DATA_DIR is None:
    # Last resort: use the first relative path even if it doesn't exist yet
    # (might be created or mounted on Streamlit Cloud)
    DATA_DIR = APP_DIR.parent / "data_storage" / "streamlit_app_data"

@st.cache_data(show_spinner=False, ttl=3600)
def load_influencer_table():
    """Load influencer table - simple and fast"""
    file_path = DATA_DIR / "influencer_table_cleaned.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Influencer table not found at {file_path}")
    return pd.read_parquet(file_path)

@st.cache_data(show_spinner=False, ttl=3600)
def load_final_dataset():
    """Load final dataset - simple and fast"""
    file_path = DATA_DIR / "final_dataset_with_attribution.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Final dataset not found at {file_path}")
    return pd.read_parquet(file_path)

@st.cache_data(show_spinner=False, ttl=3600)
def load_persons_by_row():
    """Load persons by row - simple and fast"""
    file_path = DATA_DIR / "persons_by_row_cleaned.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Persons by row not found at {file_path}")
    return pd.read_parquet(file_path)

