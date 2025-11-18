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

def _find_data_dir():
    """Find the data directory by trying multiple possible paths"""
    possible_paths = [
        APP_DIR.parent / "data_storage" / "streamlit_app_data",  # Relative from streamlit_app/
        Path("data_storage/streamlit_app_data"),  # From project root (current working directory)
        Path("/mount/src/capstone/data_storage/streamlit_app_data"),  # Streamlit Cloud path
        Path("/Users/annaglass/capstone/capstone/data_storage/streamlit_app_data"),  # Absolute fallback
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # Last resort: use the first relative path even if it doesn't exist yet
    return APP_DIR.parent / "data_storage" / "streamlit_app_data"

def _get_tried_paths():
    """Get list of paths that were tried"""
    return [
        str(APP_DIR.parent / "data_storage" / "streamlit_app_data"),
        str(Path("data_storage/streamlit_app_data")),
        str(Path("/mount/src/capstone/data_storage/streamlit_app_data")),
        str(Path("/Users/annaglass/capstone/capstone/data_storage/streamlit_app_data")),
    ]

def _find_file(filename: str) -> tuple[Path | None, list[str]]:
    """Find a file by checking all possible paths. Returns (file_path, tried_paths)"""
    tried_paths = []
    # First check streamlit_app_data locations (preferred)
    possible_dirs = [
        APP_DIR.parent / "data_storage" / "streamlit_app_data",
        Path("data_storage/streamlit_app_data"),
        Path("/mount/src/capstone/data_storage/streamlit_app_data"),
    ]
    
    # Also check final_data as fallback (for files like final_dataset_with_attribution.parquet)
    final_data_dirs = [
        APP_DIR.parent / "data_storage" / "final_data",
        Path("data_storage/final_data"),
        Path("/mount/src/capstone/data_storage/final_data")
    ]
    
    # Combine both lists
    all_dirs = possible_dirs + final_data_dirs
    
    for data_dir in all_dirs:
        tried_paths.append(str(data_dir))
        file_path = data_dir / filename
        if file_path.exists():
            return file_path, tried_paths
    
    # Return the most likely path even if it doesn't exist (for error messages)
    most_likely = APP_DIR.parent / "data_storage" / "streamlit_app_data" / filename
    return most_likely, tried_paths

DATA_DIR = _find_data_dir()

@st.cache_data(show_spinner=False, ttl=3600)
def load_influencer_table():
    """Load influencer table - simple and fast"""
    try:
        file_path, tried_paths = _find_file("influencer_table_cleaned.parquet")
        if not file_path.exists():
            error_msg = (
                f"**Influencer table not found**\n\n"
                f"Looking for: `{file_path}`\n\n"
                f"Tried paths:\n" + "\n".join(f"- {p}" for p in tried_paths) + "\n\n"
                f"Current directory: `{os.getcwd()}`\n"
                f"App directory: `{APP_DIR}`\n\n"
                f"Please ensure `influencer_table_cleaned.parquet` is available in one of these locations."
            )
            try:
                st.error(error_msg)
            except:
                pass  # If Streamlit isn't initialized yet, just continue
            return pd.DataFrame()  # Return empty DataFrame instead of raising error
        
        # Try to read the file - this can still raise FileNotFoundError in some cases
        try:
            return pd.read_parquet(file_path)
        except (FileNotFoundError, OSError) as e:
            error_msg = (
                f"**Error reading influencer table**\n\n"
                f"File path: `{file_path}`\n"
                f"Error: {str(e)}\n\n"
                f"Tried paths:\n" + "\n".join(f"- {p}" for p in tried_paths) + "\n\n"
                f"Current directory: `{os.getcwd()}`\n"
                f"App directory: `{APP_DIR}`"
            )
            try:
                st.error(error_msg)
            except:
                pass
            return pd.DataFrame()
    except Exception as e:
        try:
            st.error(f"Unexpected error loading influencer table: {str(e)}")
        except:
            pass
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def load_final_dataset():
    """Load final dataset - simple and fast"""
    try:
        file_path, tried_paths = _find_file("final_dataset_with_attribution.parquet")
        if not file_path.exists():
            error_msg = (
                f"**Final dataset not found**\n\n"
                f"Looking for: `{file_path}`\n\n"
                f"Tried paths:\n" + "\n".join(f"- {p}" for p in tried_paths) + "\n\n"
                f"Current directory: `{os.getcwd()}`\n"
                f"App directory: `{APP_DIR}`\n\n"
                f"Please ensure `final_dataset_with_attribution.parquet` is available in one of these locations."
            )
            try:
                st.error(error_msg)
            except:
                pass  # If Streamlit isn't initialized yet, just continue
            return pd.DataFrame()  # Return empty DataFrame instead of raising error
        
        # Try to read the file - this can still raise FileNotFoundError in some cases
        try:
            return pd.read_parquet(file_path)
        except (FileNotFoundError, OSError) as e:
            error_msg = (
                f"**Error reading final dataset**\n\n"
                f"File path: `{file_path}`\n"
                f"Error: {str(e)}\n\n"
                f"Tried paths:\n" + "\n".join(f"- {p}" for p in tried_paths) + "\n\n"
                f"Current directory: `{os.getcwd()}`\n"
                f"App directory: `{APP_DIR}`"
            )
            try:
                st.error(error_msg)
            except:
                pass
            return pd.DataFrame()
    except Exception as e:
        try:
            st.error(f"Unexpected error loading final dataset: {str(e)}")
        except:
            pass
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def load_persons_by_row():
    """Load persons by row - simple and fast"""
    try:
        file_path, tried_paths = _find_file("persons_by_row_cleaned.parquet")
        if not file_path.exists():
            error_msg = (
                f"**Persons by row not found**\n\n"
                f"Looking for: `{file_path}`\n\n"
                f"Tried paths:\n" + "\n".join(f"- {p}" for p in tried_paths) + "\n\n"
                f"Current directory: `{os.getcwd()}`\n"
                f"App directory: `{APP_DIR}`\n\n"
                f"Please ensure `persons_by_row_cleaned.parquet` is available in one of these locations."
            )
            try:
                st.error(error_msg)
            except:
                pass  # If Streamlit isn't initialized yet, just continue
            return pd.DataFrame()  # Return empty DataFrame instead of raising error
        
        # Try to read the file - this can still raise FileNotFoundError in some cases
        try:
            return pd.read_parquet(file_path)
        except (FileNotFoundError, OSError) as e:
            error_msg = (
                f"**Error reading persons by row**\n\n"
                f"File path: `{file_path}`\n"
                f"Error: {str(e)}\n\n"
                f"Tried paths:\n" + "\n".join(f"- {p}" for p in tried_paths) + "\n\n"
                f"Current directory: `{os.getcwd()}`\n"
                f"App directory: `{APP_DIR}`"
            )
            try:
                st.error(error_msg)
            except:
                pass
            return pd.DataFrame()
    except Exception as e:
        try:
            st.error(f"Unexpected error loading persons by row: {str(e)}")
        except:
            pass
        return pd.DataFrame()

