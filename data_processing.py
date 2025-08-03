"""
File Name: data_processing.py

Purpose:
This file cleans and processes the data to be ready for EDA and model development.

Addresses:
- Combines sentiment columns into 'sentiment' and 'sentiment_overflow'
- Drops columns with more than 90% missing values (at the end)

Optimizations:
- Memory-efficient chunked reading for large files
- Early filtering to reduce memory usage
- Optimized data type inference
- Vectorized operations for missing value calculations
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import gc
from typing import List, Optional

def data_processing(chunk_size: Optional[int] = None, optimize_dtypes: bool = True) -> pd.DataFrame:
    """
    Loads, combines, and processes multiple Excel files with memory optimization
    
    Args:
        chunk_size: If provided, processes files in chunks to reduce memory usage
        optimize_dtypes: Whether to optimize data types to reduce memory usage
        
    Returns: 
        A single, processed DataFrame combining and cleaning all data.
    """
    
    # Use pathlib for better path handling
    data_dir = Path("data/raw")
    excel_file_paths = [
        data_dir / "Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_1_df8ce708-7388-44d9-9aeb-a9f1cd72fd6b.xlsx",
        data_dir / "Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_2_5f3db6ca-8894-45b3-8e08-162e2e88baba.xlsx",
        data_dir / "Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_3_dd084585-7587-445b-ace2-fdd4eec637f9.xlsx",
        data_dir / "Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_4_9eb1d7ad-6010-4d1d-a128-b6af3cca5cfa.xlsx",
        data_dir / "Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_5_53407d72-3f41-4dc2-878e-db6076e73954.xlsx",
        data_dir / "Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_6_7e7a5c27-9b2a-444c-bec5-92133ae41865.xlsx",
        data_dir / "Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_7_7e2eaa3d-a08f-4882-a8f9-3a2ad04c91c7.xlsx",
        data_dir / "Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_8_edeac5df-5b03-424a-9706-8ef3de3827ca.xlsx",
        data_dir / "Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_d13e454c-6fef-48ac-af60-4406f9204813.xlsx"
    ]
    
    # Verify files exist before processing
    existing_files = [path for path in excel_file_paths if path.exists()]
    if len(existing_files) != len(excel_file_paths):
        print(f"Warning: {len(excel_file_paths) - len(existing_files)} files not found")
    
    # Define columns to drop early (reduces memory usage)
    cols_to_drop = ["hit_strength", "vipr_weight", "vipr_score", "country"]
    
    # Process files efficiently
    if chunk_size:
        # For very large files, process in chunks
        df = _process_files_chunked(existing_files, cols_to_drop, chunk_size, optimize_dtypes)
    else:
        # Standard processing with optimizations
        df = _process_files_standard(existing_files, cols_to_drop, optimize_dtypes)
    
    # Early filtering for US observations (reduces data size for subsequent operations)
    df = df[df["country"].eq("United States")].copy()
    
    # Drop the country column after filtering
    df = df.drop(columns=["country"], errors="ignore")
    
    # Optimize memory usage by dropping unused columns early
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors="ignore")
    
    # Optimized missing value calculation using vectorized operations
    df = _remove_high_missing_columns(df, threshold=0.9)
    
    # Optimize data types to reduce memory usage
    if optimize_dtypes:
        df = _optimize_dtypes(df)
    
    # Force garbage collection
    gc.collect()
    
    return df


def _process_files_standard(file_paths: List[Path], cols_to_drop: List[str], optimize_dtypes: bool) -> pd.DataFrame:
    """Standard file processing with memory optimizations"""
    
    all_dfs = []
    
    for i, path in enumerate(file_paths):
        print(f"Processing file {i+1}/{len(file_paths)}: {path.name}")
        
        try:
            # Read with optimized settings
            df_chunk = pd.read_excel(
                path,
                engine='openpyxl'  # Explicitly specify engine for consistency
            )
            
            # Filter early to reduce memory usage
            if "country" in df_chunk.columns:
                df_chunk = df_chunk[df_chunk["country"].eq("United States")]
            
            # Drop unnecessary columns early
            df_chunk = df_chunk.drop(columns=[col for col in cols_to_drop if col in df_chunk.columns], errors="ignore")
            
            # Optimize dtypes if requested
            if optimize_dtypes:
                df_chunk = _optimize_dtypes(df_chunk)
            
            all_dfs.append(df_chunk)
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    
    if not all_dfs:
        raise ValueError("No files were successfully processed")
    
    # Concatenate with ignore_index for efficiency
    return pd.concat(all_dfs, ignore_index=True, sort=False)


def _process_files_chunked(file_paths: List[Path], cols_to_drop: List[str], chunk_size: int, optimize_dtypes: bool) -> pd.DataFrame:
    """Process files in chunks for memory efficiency with very large datasets"""
    
    all_chunks = []
    
    for i, path in enumerate(file_paths):
        print(f"Processing file {i+1}/{len(file_paths)} in chunks: {path.name}")
        
        try:
            # For Excel files, we can't easily chunk read, so we read the full file
            # but process in smaller batches if it's very large
            df_file = pd.read_excel(path, engine='openpyxl')
            
            # Process in chunks if file is large
            if len(df_file) > chunk_size:
                for chunk_start in range(0, len(df_file), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(df_file))
                    df_chunk = df_file.iloc[chunk_start:chunk_end].copy()
                    
                    # Apply filtering and processing to chunk
                    df_chunk = _process_chunk(df_chunk, cols_to_drop, optimize_dtypes)
                    
                    if len(df_chunk) > 0:  # Only add non-empty chunks
                        all_chunks.append(df_chunk)
            else:
                # Process entire file if it's small enough
                df_processed = _process_chunk(df_file, cols_to_drop, optimize_dtypes)
                if len(df_processed) > 0:
                    all_chunks.append(df_processed)
                    
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    
    if not all_chunks:
        raise ValueError("No data was successfully processed")
    
    return pd.concat(all_chunks, ignore_index=True, sort=False)


def _process_chunk(df_chunk: pd.DataFrame, cols_to_drop: List[str], optimize_dtypes: bool) -> pd.DataFrame:
    """Process a single chunk of data"""
    
    # Filter for US observations
    if "country" in df_chunk.columns:
        df_chunk = df_chunk[df_chunk["country"].eq("United States")]
    
    # Drop unnecessary columns
    df_chunk = df_chunk.drop(columns=[col for col in cols_to_drop if col in df_chunk.columns], errors="ignore")
    
    # Optimize dtypes
    if optimize_dtypes:
        df_chunk = _optimize_dtypes(df_chunk)
    
    return df_chunk


def _remove_high_missing_columns(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Efficiently remove columns with high missing values using vectorized operations"""
    
    # Vectorized calculation of missing ratios
    # Use np.where for efficient empty string detection
    missing_mask = df.isna() | (df == "")
    missing_ratio = missing_mask.mean()
    
    # Find columns to drop
    columns_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    
    if columns_to_drop:
        print(f"Dropping {len(columns_to_drop)} columns with >{threshold*100}% missing values")
        df = df.drop(columns=columns_to_drop)
    
    return df


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize data types to reduce memory usage"""
    
    # Convert object columns to category where appropriate
    for col in df.select_dtypes(include=['object']):
        if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
            df[col] = df[col].astype('category')
    
    # Downcast numeric types
    for col in df.select_dtypes(include=['int64']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df


def get_memory_usage(df: pd.DataFrame) -> None:
    """Print memory usage information"""
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # Convert to MB
    print(f"DataFrame memory usage: {memory_usage:.2f} MB")
    print(f"Shape: {df.shape}")


if __name__ == "__main__": 
    print("Starting optimized data processing")
    
    # Process with optimizations
    final_df = data_processing(optimize_dtypes=True)
    
    # Print memory usage info
    get_memory_usage(final_df)
    
    # Use pathlib for output path
    output_path = Path("data/processed/processed_data.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    
    # Save with optimized settings
    final_df.to_csv(output_path, index=False)
    
    print(f"Data processing complete. The new file has been saved to: {output_path}")
    print(f"Final dataset shape: {final_df.shape}")