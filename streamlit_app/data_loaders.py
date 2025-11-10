"""
Data Loading Functions - Simplified for speed
Just load parquet files directly, no conversions
"""

import streamlit as st
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/Users/annaglass/capstone/capstone/data_storage/streamlit_app_data")

@st.cache_data(show_spinner=False, ttl=3600)
def load_influencer_table():
    """Load influencer table - simple and fast"""
    return pd.read_parquet(DATA_DIR / "influencer_table_cleaned.parquet")

@st.cache_data(show_spinner=False, ttl=3600)
def load_final_dataset():
    """Load final dataset - simple and fast"""
    return pd.read_parquet(DATA_DIR / "final_dataset_with_attribution.parquet")

@st.cache_data(show_spinner=False, ttl=3600)
def load_persons_by_row():
    """Load persons by row - simple and fast"""
    return pd.read_parquet(DATA_DIR / "persons_by_row_cleaned.parquet")

