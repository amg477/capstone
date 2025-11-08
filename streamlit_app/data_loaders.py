"""
Data Loading Functions
Handles loading of datasets, models, and cached resources
"""

import streamlit as st
import pandas as pd
from pathlib import Path


@st.cache_data(show_spinner=False, ttl=3600)
def load_influencer_table():
    project_root = Path(__file__).parent.parent
    # Prefer cleaned Parquet in streamlit_app_data, fallback to final_data
    preferred = project_root / "data_storage" / "streamlit_app_data" / "influencer_table.parquet"
    fallback = project_root / "data_storage" / "final_data" / "influencer_table.parquet"
    path = preferred if preferred.exists() else fallback
    df = pd.read_parquet(path)
        
    # Optimize data types to reduce memory
    if 'circulation_size' in df.columns:
        df['circulation_size'] = pd.to_numeric(df['circulation_size'], errors='coerce', downcast='float')
    if 'sentiment_score' in df.columns:
        df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce', downcast='float')
    if 'mention_count' in df.columns:
        df['mention_count'] = pd.to_numeric(df['mention_count'], errors='coerce', downcast='integer')
        
    return df


@st.cache_data(show_spinner=False)
def load_final_dataset():

    project_root = Path(__file__).parent.parent
    # Prefer cleaned Parquet in streamlit_app_data, fallback to final_data
    preferred = project_root / "data_storage" / "streamlit_app_data" / "final_dataset_with_attribution.parquet"
    fallback = project_root / "data_storage" / "final_data" / "final_dataset_with_attribution.parquet"
    path = preferred if preferred.exists() else fallback
    df = pd.read_parquet(path)
    
    if 'circulation_size' in df.columns:
        df['circulation_size'] = pd.to_numeric(df['circulation_size'], errors='coerce', downcast='float')
    if 'sentiment_score' in df.columns:
        df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce', downcast='float')
    if 'row_index' in df.columns:
        df['row_index'] = pd.to_numeric(df['row_index'], errors='coerce', downcast='integer')
    
    return df


@st.cache_data(show_spinner=False)
def load_persons_by_row():
    """Load persons by row data - prefer parquet, fallback to CSV"""
    project_root = Path(__file__).parent.parent
    # Prefer cleaned Parquet in streamlit_app_data, fallback to final_data
    preferred = project_root / "data_storage" / "streamlit_app_data" / "persons_by_row.parquet"
    fallback = project_root / "data_storage" / "final_data" / "persons_by_row.parquet"
    path = preferred if preferred.exists() else fallback
    df = pd.read_parquet(path)
    
    # Optimize data types
    if 'row_index' in df.columns:
        df['row_index'] = pd.to_numeric(df['row_index'], errors='coerce', downcast='integer')
    
    return df

