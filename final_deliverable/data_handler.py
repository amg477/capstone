# data_handler.py — Data Loading & Management

from pathlib import Path
from typing import Optional, Tuple, Set
import pandas as pd
import streamlit as st

from config import DATA_CONFIG

@st.cache_data
def load_combined_dataset() -> pd.DataFrame:
    """Load and combine all split dataset files."""
    possible_paths = [
        Path(DATA_CONFIG["file_locations"]["split_data_dir"]),
        Path.cwd() / DATA_CONFIG["file_locations"]["split_data_dir"],
        Path("../data/processed/split"),
        Path("../data/split"),
    ]
    
    # Add more possible paths
    for base in ["data", "data/processed"]:
        possible_paths.extend([
            Path(base),
            Path.cwd() / base,
        ])
    
    split_dir = next((p for p in possible_paths if p.exists()), None)
    if split_dir is None:
        return pd.DataFrame()
    
    combined: list[pd.DataFrame] = []
    file_pattern = "final_model_dataset_part_*.csv"
    
    # Try to find split files
    split_files = list(split_dir.glob(file_pattern))
    if not split_files:
        # Fallback: try parent directories
        for parent in split_dir.parents:
            split_files = list(parent.glob(file_pattern))
            if split_files:
                split_dir = parent
                break
    
    # Load files in order
    for fp in sorted(split_files):
        if fp.exists():
            try:
                df = pd.read_csv(fp, dtype_backend="pyarrow")
                combined.append(df)
                print(f"Loaded: {fp.name} ({len(df):,} rows)")
            except Exception as e:
                print(f"Error loading {fp}: {e}")
                continue
    
    if not combined:
        return pd.DataFrame()
    
    final_df = pd.concat(combined, ignore_index=True)
    combined.clear()
    print(f"Total loaded: {len(final_df):,} rows")
    return final_df

@st.cache_resource
def get_main_dataset() -> pd.DataFrame:
    """Get the main dataset (cached)."""
    return load_combined_dataset()

@st.cache_resource  
def get_attribution_data() -> Optional[pd.DataFrame]:
    """Load attribution analysis data."""
    attr_file = DATA_CONFIG["file_locations"]["attribution_file"]
    
    possible_paths = [
        Path(attr_file),
        Path.cwd() / attr_file,
        Path("../data/processed/attribution_all_scored.csv"),
        Path("../data/processed/attribution_all_scored.csv"),
    ]
    
    attr_path = next((p for p in possible_paths if p.exists()), None)
    if attr_path:
        try:
            return pd.read_csv(attr_path, dtype_backend="pyarrow")
        except Exception as e:
            st.warning(f"Could not load attribution data: {e}")
            return None
    return None

def get_data() -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Set[str]]:
    """Get main data, attribution data, and available columns."""
    df_main = get_main_dataset()
    df_attr = get_attribution_data()
    columns = set(df_main.columns) if not df_main.empty else set()
    
    return df_main, df_attr, columns

def get_column_by_type(df: pd.DataFrame, column_types: list[str]) -> Optional[str]:
    """Find first column matching any of the given types."""
    cols_lower = [c.lower() for c in df.columns]
    for col_type in column_types:
        for col in df.columns:
            if any(k in col.lower() for k in col_type.split()):
                return col
    return None

def sample_data_for_charts(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """Intelligent sampling for charting based on importance."""
    if len(df) <= sample_size:
        return df
    
    # Create importance score
    score = pd.Series(0.0, index=df.index)
    
    # Add influence scores if available
    influence_cols = ["pub_credit_share", "max_term_credit", "credit_share"]
    for col in influence_cols:
        if col in df.columns:
            score += df[col].fillna(0) * 1000
    
    # Add circulation/audience metrics
    circ_cols = ["circulation_size", "reach", "impressions", "audience"]
    for col in circ_cols:
        if col in df.columns:
            try:
                circ = pd.to_numeric(df[col], errors='coerce')
                if circ.notna().any():
                    norm = (circ - circ.min()) / (circ.max() - circ.min() + 1e-9)
                    score += norm.fillna(0) * 100
            except Exception:
                continue
    
    # Sample top items by importance
    top_indices = score.nlargest(sample_size).index
    return df.loc[top_indices]
